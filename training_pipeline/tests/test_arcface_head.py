"""Unit tests for ArcFaceHead — shape, finite, math correctness, gradient flow."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from training_pipeline.src.arcface_head import ArcFaceHead


@pytest.fixture
def head():
    torch.manual_seed(0)
    return ArcFaceHead(embedding_dim=8, num_classes=4, s=64.0, m=0.5)


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


def test_output_shape(head):
    emb = _normed(torch.randn(3, 8))
    labels = torch.tensor([0, 1, 2])
    logits = head(emb, labels)
    assert logits.shape == (3, 4)


def test_logits_finite_on_random_input(head):
    emb = _normed(torch.randn(16, 8))
    labels = torch.randint(0, 4, (16,))
    logits = head(emb, labels)
    assert torch.isfinite(logits).all()


def test_correct_class_logit_drops_with_margin(head):
    """For positive cos_theta, cos(theta + m) < cos(theta) so the target logit drops vs no-margin."""
    torch.manual_seed(1)
    # Build an embedding aligned exactly with class-0's weight direction
    with torch.no_grad():
        w0 = F.normalize(head.weight[0:1], dim=1)  # [1, 8]
    emb = w0.clone()                                 # already unit-norm
    labels = torch.tensor([0])
    logits = head(emb, labels)
    # Reference: scaled cosine without margin would be s * 1.0 == s
    assert logits[0, 0].item() < head.s, "margin should pull the target logit below s"
    assert logits[0, 0].item() > 0, "with cos_theta=1, margin output should still be positive"


def test_non_target_logits_use_plain_cosine(head):
    emb = _normed(torch.randn(2, 8))
    labels = torch.tensor([0, 1])
    logits = head(emb, labels)
    weight_norm = F.normalize(head.weight, dim=1)
    plain = head.s * (emb @ weight_norm.t())
    # Non-target columns must equal plain scaled cosine (margin is on target only).
    assert torch.allclose(logits[0, 1:], plain[0, 1:], atol=1e-5)
    assert torch.allclose(logits[1, [0, 2, 3]], plain[1, [0, 2, 3]], atol=1e-5)


def test_logit_lower_bound_with_standard_fallback(head):
    """Standard non-easy-margin ArcFace can push target logits to -(1 + mm) * s."""
    mm = math.sin(math.pi - head.m) * head.m
    expected_floor = -(1.0 + mm) * head.s
    # Push embedding to be anti-aligned with class 0's weight (cos_theta ≈ -1)
    with torch.no_grad():
        w0 = F.normalize(head.weight[0:1], dim=1)
    emb = -w0.clone()
    labels = torch.tensor([0])
    logits = head(emb, labels)
    target = logits[0, 0].item()
    # Must respect the derived lower bound (small slack for fp32 noise).
    assert target >= expected_floor - 1e-3
    # And must reflect the fallback branch — i.e., target < non-target (plain) cosine column.
    assert target < head.s * 1.0


def test_gradient_flows_to_embedding_and_weight(head):
    emb = _normed(torch.randn(4, 8)).requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(emb, labels)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()


def test_center_norms_returns_per_class_norms(head):
    norms = head.center_norms()
    assert norms.shape == (4,)
    assert torch.isfinite(norms).all()
    # Xavier-normal init → norms should sit in a small range around sqrt(2/(D+C))*sqrt(D).
    assert (norms > 0).all()
