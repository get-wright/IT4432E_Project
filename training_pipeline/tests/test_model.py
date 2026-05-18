import torch
from training_pipeline.src.model import FaceEmbedding, ClassifierHead


def test_embedding_shape_and_norm():
    m = FaceEmbedding(embedding_dim=512, pretrained=False).eval()
    x = torch.randn(4, 3, 160, 160)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (4, 512)
    # forward is un-normalized; unit-norm check moved to test_embed_normalized_unit_vectors
    norms = y.norm(p=2, dim=1)
    assert not torch.allclose(norms, torch.ones(4), atol=0.1)


def test_custom_embedding_dim():
    m = FaceEmbedding(embedding_dim=128, pretrained=False).eval()
    x = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 128)


def test_forward_unnormalized():
    """forward returns un-normalized vectors. The Phase 2 triplet loss normalizes internally;
    the Phase 1 classifier head consumes raw logits."""
    model = FaceEmbedding(embedding_dim=64, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 160, 160)
    with torch.no_grad():
        out = model(x)
    norms = out.norm(p=2, dim=1)
    assert out.shape == (4, 64)
    # If model were L2-normalized, norms would all be ~1.0. We want UN-normalized.
    assert not torch.allclose(norms, torch.ones(4), atol=0.1), (
        f"forward output looks L2-normalized (norms={norms.tolist()}); should be raw."
    )


def test_embed_normalized_unit_vectors():
    model = FaceEmbedding(embedding_dim=64, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 160, 160)
    with torch.no_grad():
        out = model.embed_normalized(x)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"norms={norms.tolist()}"


def test_classifier_head_shapes():
    head = ClassifierHead(embedding_dim=64, n_identities=10)
    emb = torch.randn(4, 64)
    logits = head(emb)
    assert logits.shape == (4, 10)


def test_classifier_head_has_no_bias():
    head = ClassifierHead(embedding_dim=64, n_identities=10)
    assert head.fc.bias is None


def test_embed_tta_unit_norm_and_flip_consistency():
    """embed_tta returns unit-norm and is permutation-invariant to flipping the input."""
    import torch
    import torch.nn.functional as F
    from training_pipeline.src.model import FaceEmbedding

    torch.manual_seed(0)
    model = FaceEmbedding(embedding_dim=32, pretrained=False).eval()
    x = torch.randn(2, 3, 160, 160)

    with torch.no_grad():
        e = model.embed_tta(x)
        e_flipped = model.embed_tta(torch.flip(x, dims=[-1]))

    # Unit norm
    assert torch.allclose(e.norm(dim=1), torch.ones(2), atol=1e-5)
    # Flip-invariant: TTA averages x and flip(x), so embedding of flip(x) is the same set.
    assert torch.allclose(e, e_flipped, atol=1e-5)
