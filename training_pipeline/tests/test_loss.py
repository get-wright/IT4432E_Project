import math

import torch
from training_pipeline.src.loss import batch_hard_triplet_loss, semi_hard_triplet_loss, softplus_loss


def test_loss_zero_when_anchor_equals_positive_and_far_from_neg():
    # 2 identities, 2 samples each. Same-id samples identical; different-id far apart.
    emb = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    # Hardest pos dist = 0, hardest neg dist = sqrt(2) > margin -> loss = 0.
    assert loss.item() == 0.0


def test_loss_positive_when_negatives_closer_than_positives():
    emb = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],  # same id 0, far from anchor
        [0.1, 0.1],  # different id, close to anchor
        [0.1, 0.1],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(emb, labels, margin=0.3)
    assert loss.item() > 0.0


def test_softplus_loss_positive_when_d_ap_greater():
    d_ap = torch.tensor([0.5])
    d_an = torch.tensor([0.4])
    loss = softplus_loss(d_ap, d_an)
    # softplus(0.5 - 0.4) = log(1 + e^0.1) ≈ 0.7444
    assert math.isclose(loss.item(), 0.7444, abs_tol=1e-3)


def test_softplus_loss_smooth_gradient_past_margin():
    """Unlike hard hinge, softplus gives a nonzero gradient even when d_an >> d_ap."""
    d_ap = torch.tensor([0.1], requires_grad=True)
    d_an = torch.tensor([0.9])
    loss = softplus_loss(d_ap, d_an)
    loss.backward()
    assert d_ap.grad is not None
    assert torch.isfinite(d_ap.grad).all()
    assert d_ap.grad.item() > 0


def test_semi_hard_returns_tuple():
    emb = torch.randn(8, 32)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    result = semi_hard_triplet_loss(emb, labels)
    assert isinstance(result, tuple)
    assert len(result) == 2
    loss, n_triplets = result
    assert isinstance(loss, torch.Tensor)
    assert isinstance(n_triplets, int)


def test_semi_hard_normalizes_input():
    """Distances must be computed on unit-normalized vectors regardless of input norm.
    Same embedding pattern scaled by 100x should give identical loss + same n_triplets."""
    base = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])

    # Set a manual seed so the random in-band negative pick is deterministic.
    torch.manual_seed(0)
    loss_small, n_small = semi_hard_triplet_loss(base, labels, margin=0.3)
    torch.manual_seed(0)
    loss_big, n_big = semi_hard_triplet_loss(base * 100, labels, margin=0.3)

    assert n_small == n_big
    assert torch.allclose(loss_small, loss_big, atol=1e-5)


def test_semi_hard_no_valid_returns_graph_connected_zero():
    """All embeddings identical → no negative satisfies d_an > d_ap (both are 0).
    Must return n=0 and a loss that supports .backward() without NaN."""
    emb = torch.ones(4, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss, n_triplets = semi_hard_triplet_loss(emb, labels, margin=0.3)
    assert n_triplets == 0
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(emb.grad).all()


def test_semi_hard_picks_when_in_band_exists():
    """Set up 1 anchor + 1 positive + 1 in-band negative + 1 out-of-band negative.
    Expect n_triplets >= 1 and loss > 0."""
    def from_angle(theta):
        return [math.cos(theta), math.sin(theta)]

    emb = torch.tensor([
        from_angle(0.0),     # anchor, id 0
        from_angle(0.1),     # positive, id 0  -> d_ap ≈ 0.1 after L2-norm
        from_angle(0.25),    # in-band negative, id 1  (band = (0.1, 0.4))
        from_angle(0.8),     # out-of-band negative, id 1
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    loss, n_triplets = semi_hard_triplet_loss(emb, labels, margin=0.3)
    assert n_triplets >= 1
    assert loss.item() > 0
