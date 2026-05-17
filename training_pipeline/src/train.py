"""Train Siamese face embedding network."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import FaceDataset, PKSampler, train_transform
from .eval_lfw import evaluate_lfw, load_pairs_txt
from .loss import batch_hard_triplet_loss
from .model import FaceEmbedding
from .utils import AverageMeter, set_seed

ROOT = Path(__file__).resolve().parents[2]


def _build_optimizer(model, lr_head: float, lr_backbone: float, wd: float):
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.fc" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ],
        weight_decay=wd,
    )


def _cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def run_training(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(cfg["checkpoints_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(cfg["tensorboard_dir"]);  tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    # FaceDataset reads paths relative to project ROOT. To support absolute paths
    # in tests, monkey-patch a local FaceDataset variant: just call FaceDataset
    # but if the path is absolute, ROOT-prefixing has no effect (Path / abs = abs).
    train_ds = FaceDataset(Path(cfg["manifest"]), split="train", transform=train_transform())
    sampler = PKSampler(
        train_ds.labels,
        p=cfg["train"]["p"],
        k=cfg["train"]["k"],
        num_batches=cfg["train"]["batches_per_epoch"],
        seed=cfg["seed"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg["train"]["num_workers"] > 0),
    )

    model = FaceEmbedding(embedding_dim=cfg["train"]["embedding_dim"]).to(device)
    optim = _build_optimizer(
        model,
        cfg["train"]["lr_head"],
        cfg["train"]["lr_backbone"],
        cfg["train"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    total_steps = cfg["train"]["epochs"] * cfg["train"]["batches_per_epoch"]
    warmup = cfg["train"]["warmup_steps"]
    base_head = cfg["train"]["lr_head"]
    base_back = cfg["train"]["lr_backbone"]

    pairs = []
    if cfg["eval"].get("pairs_txt"):
        pairs_path = ROOT / cfg["eval"]["pairs_txt"]
        if pairs_path.exists() and cfg["eval"]["max_pairs_inloop"] > 0:
            pairs = load_pairs_txt(pairs_path)[: cfg["eval"]["max_pairs_inloop"]]

    history: dict[str, list[float]] = {"train_loss": [], "lfw_acc": []}
    best_acc = -1.0
    step = 0

    try:
        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            meter = AverageMeter()
            for imgs, labels in tqdm(train_loader, desc=f"epoch {epoch}"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lr_h = _cosine_lr(step, total_steps, warmup, base_head)
                lr_b = _cosine_lr(step, total_steps, warmup, base_back)
                optim.param_groups[0]["lr"] = lr_b
                optim.param_groups[1]["lr"] = lr_h

                optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    emb = model(imgs)
                    loss = batch_hard_triplet_loss(emb, labels, margin=cfg["train"]["margin"])
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                meter.update(loss.item(), imgs.size(0))
                writer.add_scalar("train/loss_step", loss.item(), step)
                writer.add_scalar("train/lr_head", lr_h, step)
                step += 1

            history["train_loss"].append(meter.avg)
            writer.add_scalar("train/loss_epoch", meter.avg, epoch)

            acc = -1.0
            if pairs:
                metrics = evaluate_lfw(model, pairs, device, cfg["eval"]["batch_size"])
                acc = metrics["mean_acc"]
                writer.add_scalar("eval/lfw_acc", acc, epoch)
            history["lfw_acc"].append(acc)

            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                       ckpt_dir / "last.pt")
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                           ckpt_dir / "best.pt")

            print(f"epoch {epoch}: loss={meter.avg:.4f} lfw_acc={acc:.4f}")

        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    finally:
        writer.close()
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_training(cfg)


if __name__ == "__main__":
    main()
