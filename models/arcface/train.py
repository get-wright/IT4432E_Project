"""Two-phase training: softmax warmup (CE) → ArcFace + SGD + MultiStepLR."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .arcface_head import ArcFaceHead
from .dataset import FaceDataset, PKSampler, train_transform
from .eval_lfw import evaluate_lfw, load_pairs_txt
from .model import ClassifierHead, FaceEmbedding
from .utils import AverageMeter, set_seed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/process-data"))
from lfw_layout import find_lfw_identity_root  # noqa: E402


def _build_optimizer(params_backbone, params_head, lr_backbone, lr_head, wd):
    return torch.optim.AdamW(
        [
            {"params": params_backbone, "lr": lr_backbone},
            {"params": params_head,     "lr": lr_head},
        ],
        weight_decay=wd,
    )


def _split_params(model: FaceEmbedding):
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if "backbone.fc" in name else backbone).append(p)
    return backbone, head


def _cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def _run_lfw_probe(model, pairs, aligned_root, raw_root, device, max_pairs):
    """Non-strict in-loop probe. Returns metrics dict; never aborts training.

    `max_pairs` is honored only when it's a positive int. None / 0 / negative → use all pairs.
    Slicing the LFW pair list with a small N gives an unbalanced prefix because pairs.txt
    orders each fold as N_per_fold positives followed by N_per_fold negatives.
    """
    subset = pairs[:max_pairs] if (max_pairs and max_pairs > 0) else pairs
    return evaluate_lfw(
        model, subset,
        aligned_root=aligned_root, raw_root=raw_root,
        device=device, strict=False,
    )


def run_training(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(cfg["checkpoints_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(cfg["tensorboard_dir"]);   tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    train_ds = FaceDataset(Path(cfg["manifest"]), split="train", transform=train_transform())
    n_identities = len(set(train_ds.labels))

    # LFW probe inputs (shared across both phases).
    pairs = load_pairs_txt(Path(cfg["eval"]["pairs_txt"]))
    aligned_root = ROOT / "shared/process-data/lfw_pairs"
    raw_root = find_lfw_identity_root(ROOT / "preshared/process-data/lfw")

    model = FaceEmbedding(embedding_dim=cfg["train"]["embedding_dim"]).to(device)
    classifier = ClassifierHead(cfg["train"]["embedding_dim"], n_identities).to(device)

    history = {"train_loss": [], "lfw_acc": [], "spread": [], "phase": []}
    best_acc = 0.0

    # ---------- Phase 1: softmax warmup ----------
    phase1_epochs = cfg["train"]["phase1_epochs"]
    phase1_batches = cfg["train"]["batches_per_epoch"]
    phase1_bs = cfg["train"]["phase1_batch_size"]

    p1_sampler = RandomSampler(
        train_ds, replacement=True,
        num_samples=phase1_batches * phase1_bs,
    )
    p1_loader = DataLoader(
        train_ds, sampler=p1_sampler,
        batch_size=phase1_bs,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg["train"]["num_workers"] > 0),
    )

    backbone_params, head_params = _split_params(model)
    p1_optim = _build_optimizer(
        backbone_params, list(classifier.parameters()) + head_params,
        cfg["train"]["lr_backbone"], cfg["train"]["lr_head"], cfg["train"]["weight_decay"],
    )
    ce = torch.nn.CrossEntropyLoss()
    p1_total_steps = phase1_epochs * phase1_batches
    p1_warmup = cfg["train"]["warmup_steps"]
    step = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(1, phase1_epochs + 1):
        model.train(); classifier.train()
        meter = AverageMeter()
        pbar = tqdm(p1_loader, desc=f"P1 epoch {epoch}/{phase1_epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            for pg, base in zip(p1_optim.param_groups, [cfg["train"]["lr_backbone"], cfg["train"]["lr_head"]]):
                pg["lr"] = _cosine_lr(step, p1_total_steps, p1_warmup, base)
            p1_optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                emb = model(imgs)
                logits = classifier(emb)
                loss = ce(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(p1_optim)
            scaler.update()
            meter.update(loss.item(), imgs.size(0))
            pbar.set_postfix(loss=meter.avg)
            writer.add_scalar("p1/train_loss_step", loss.item(), step)
            step += 1
        metrics = _run_lfw_probe(model, pairs, aligned_root, raw_root, device,
                                 cfg["eval"]["max_pairs_inloop"])
        history["train_loss"].append(meter.avg)
        history["lfw_acc"].append(metrics["mean_acc"])
        history["spread"].append(metrics["spread"])
        history["phase"].append(1)
        print(f"[P1 epoch {epoch}] train_loss={meter.avg:.4f}  lfw_acc={metrics['mean_acc']:.4f}  "
              f"spread={metrics['spread']:.4f}  pos={metrics['pos_sim_mean']:.3f}  neg={metrics['neg_sim_mean']:.3f}")
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

    if history["spread"][-1] <= 0.05:
        raise RuntimeError(
            f"phase 1 produced collapsed embeddings (spread={history['spread'][-1]:.4f}). "
            f"Check data, augmentations, learning rate. Aborting before phase 2."
        )

    torch.save(
        {"model": model.state_dict(), "classifier": classifier.state_dict(), "cfg": cfg},
        ckpt_dir / "phase1_end.pt",
    )

    # ---------- Phase 2: semi-hard triplet ----------
    del classifier
    phase2_epochs = cfg["train"]["phase2_epochs"]
    phase2_batches = cfg["train"]["batches_per_epoch"]

    p2_sampler = PKSampler(
        train_ds.labels,
        p=cfg["train"]["p"],
        k=cfg["train"]["k"],
        num_batches=phase2_batches,
        seed=cfg["seed"],
    )
    p2_loader = DataLoader(
        train_ds, batch_sampler=p2_sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg["train"]["num_workers"] > 0),
    )

    # === P2: ArcFace head + SGD m=0.9 + MultiStepLR step decay ===
    arc_head = ArcFaceHead(
        embedding_dim=cfg["train"]["embedding_dim"],
        num_classes=n_identities,
        s=cfg["train"]["arcface_s"],
        m=cfg["train"]["arcface_m"],
    ).to(device)

    # Backbone gets weight decay; ArcFace centers don't (decay collapses center norms,
    # destabilizes the normalized angular distance).
    p2_optim = torch.optim.SGD(
        [
            {"params": model.parameters(), "weight_decay": cfg["train"]["weight_decay"]},
            {"params": arc_head.parameters(), "weight_decay": 0.0},
        ],
        lr=cfg["train"]["p2_lr"],
        momentum=0.9,
        nesterov=False,
    )

    p2_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        p2_optim,
        milestones=cfg["train"]["p2_lr_decay_epochs"],
        gamma=cfg["train"]["p2_lr_decay_gamma"],
    )

    # Linear LR warmup over the first P2 epoch — prevents the geometry shock
    # observed when switching from AdamW/CE warmup to SGD/ArcFace at full LR.
    p2_warmup_steps = phase2_batches
    p2_base_lr = cfg["train"]["p2_lr"]

    step = 0
    for epoch in range(1, phase2_epochs + 1):
        model.train()
        arc_head.train()
        meter = AverageMeter()
        pbar = tqdm(p2_loader, desc=f"P2 epoch {epoch}/{phase2_epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if step < p2_warmup_steps:
                warmup_factor = (step + 1) / p2_warmup_steps
                for pg in p2_optim.param_groups:
                    pg["lr"] = p2_base_lr * warmup_factor
            p2_optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                emb = model.embed_normalized(imgs)
                logits = arc_head(emb, labels)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(p2_optim)
            scaler.update()
            meter.update(loss.item(), imgs.size(0))
            pbar.set_postfix(loss=meter.avg)
            writer.add_scalar("p2/train_loss_step", loss.item(), step)
            step += 1

        p2_scheduler.step()

        metrics = _run_lfw_probe(
            model, pairs, aligned_root, raw_root, device,
            cfg["eval"]["max_pairs_inloop"],
        )
        history["train_loss"].append(meter.avg)
        history["lfw_acc"].append(metrics["mean_acc"])
        history["spread"].append(metrics["spread"])
        history["phase"].append(2)

        center_norms = arc_head.center_norms()
        print(
            f"[P2 epoch {epoch}] train_loss={meter.avg:.4f}  "
            f"lfw_acc={metrics['mean_acc']:.4f}  spread={metrics['spread']:.4f}  "
            f"center_norm mean={center_norms.mean().item():.3f} "
            f"min={center_norms.min().item():.3f} "
            f"max={center_norms.max().item():.3f}"
        )
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

        if metrics["mean_acc"] > best_acc and metrics["spread"] > 0.02:
            best_acc = metrics["mean_acc"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg},
                ckpt_dir / cfg.get("checkpoint_name", "best.pt"),
            )

    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / "last.pt")
    torch.save(arc_head.state_dict(), ckpt_dir / "arcface_head.pt")
    return {"best_acc": best_acc, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path(__file__).resolve().parent.parent / "configs/train.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    result = run_training(cfg)
    print(f"best LFW probe acc: {result['best_acc']:.4f}")


if __name__ == "__main__":
    main()
