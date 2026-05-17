import torch
from training_pipeline.src.loss import batch_hard_triplet_loss


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
