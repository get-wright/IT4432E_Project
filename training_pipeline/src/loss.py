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
