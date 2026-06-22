import torch
import torch.nn.functional as F


def _pairwise_dist(emb: torch.Tensor) -> torch.Tensor:
    """Compute pairwise euclidean distances.

    Args:
        emb: (N, D) embedding matrix

    Returns:
        (N, N) distance matrix
    """
    # Using expanded form for numerical stability: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    dot = emb @ emb.t()
    sq = (emb * emb).sum(dim=1)
    d2 = sq.unsqueeze(0) + sq.unsqueeze(1) - 2 * dot
    d2 = d2.clamp(min=0.0)
    return torch.sqrt(d2 + 1e-12)


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """BatchHard triplet loss (Hermans et al., 2017).

    For each anchor, picks the hardest positive (max distance among same-label)
    and the hardest negative (min distance among different-label), then computes
    relu(d_ap - d_an + margin) and averages over the batch.

    Args:
        emb: (N, D) embedding matrix, typically L2-normalized
        labels: (N,) class labels for each embedding
        margin: margin parameter for triplet loss

    Returns:
        scalar loss value
    """
    dist = _pairwise_dist(emb)
    n = labels.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same & ~eye
    neg_mask = ~same

    pos_dist = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg_dist = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    valid = (pos_dist != float("-inf")) & (neg_dist != float("inf"))
    losses = F.relu(pos_dist[valid] - neg_dist[valid] + margin)
    if losses.numel() == 0:
        return torch.tensor(0.0, device=emb.device)
    return losses.mean()


def softplus_loss(d_ap: torch.Tensor, d_an: torch.Tensor) -> torch.Tensor:
    """Soft-margin triplet loss: log(1 + exp(d_ap - d_an))."""
    return F.softplus(d_ap - d_an)


def semi_hard_triplet_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, int]:
    """Semi-hard negative mining triplet loss.

    For each (anchor a, positive p) pair, pick a negative n such that
    d_ap < d_an < d_ap + margin (the "semi-hard" band). Fall back to the
    closest harder-than-positive negative if no in-band one exists.

    Returns (loss, n_triplets). When n_triplets == 0 the loss is
    `emb.sum() * 0.0` — a graph-connected zero so `.backward()` is safe
    but produces no gradients. Callers should skip `optimizer.step()`.

    Input is L2-normalized internally so distances live on the unit
    hypersphere — prevents norm-scale escape.
    """
    emb = F.normalize(emb, p=2, dim=1)
    dist = _pairwise_dist(emb)
    n = labels.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same & ~eye
    neg_mask = ~same

    d_aps: list[torch.Tensor] = []
    d_ans: list[torch.Tensor] = []
    for a in range(n):
        pos_idxs = pos_mask[a].nonzero(as_tuple=False).flatten()
        for p in pos_idxs:
            d_ap = dist[a, p]
            d_an_vec = dist[a]
            in_band = neg_mask[a] & (d_an_vec > d_ap) & (d_an_vec < d_ap + margin)
            in_band_idxs = in_band.nonzero(as_tuple=False).flatten()
            if len(in_band_idxs) > 0:
                pick = in_band_idxs[torch.randint(len(in_band_idxs), (1,), device=emb.device).item()]
                d_ans.append(d_an_vec[pick])
                d_aps.append(d_ap)
                continue
            harder = neg_mask[a] & (d_an_vec > d_ap)
            harder_idxs = harder.nonzero(as_tuple=False).flatten()
            if len(harder_idxs) > 0:
                pick = harder_idxs[d_an_vec[harder_idxs].argmin()]
                d_ans.append(d_an_vec[pick])
                d_aps.append(d_ap)

    if not d_aps:
        return emb.sum() * 0.0, 0

    d_ap_t = torch.stack(d_aps)
    d_an_t = torch.stack(d_ans)
    loss = softplus_loss(d_ap_t, d_an_t).mean()
    return loss, len(d_aps)
