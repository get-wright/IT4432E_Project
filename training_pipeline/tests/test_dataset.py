from collections import Counter
from training_pipeline.src.dataset import PKSampler


def test_pk_sampler_yields_pk_batch_with_p_unique_ids():
    # 10 identities, 5 samples each. Labels at indices = identity_id.
    labels = [i // 5 for i in range(50)]  # 10 IDs x 5 samples.
    sampler = PKSampler(labels, p=4, k=2, num_batches=3, seed=0)
    batches = list(iter(sampler))
    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 8  # P*K
        ids = [labels[i] for i in batch]
        c = Counter(ids)
        assert len(c) == 4
        for v in c.values():
            assert v == 2


def test_pk_sampler_skips_labels_with_too_few_samples():
    # 1 identity has 1 sample (< k=2). Should be excluded.
    labels = [0, 1, 1, 1, 2, 2, 2]
    sampler = PKSampler(labels, p=2, k=2, num_batches=5, seed=0)
    for batch in iter(sampler):
        ids = [labels[i] for i in batch]
        assert 0 not in ids


def test_train_transform_has_no_random_erasing():
    from torchvision import transforms
    from training_pipeline.src.dataset import train_transform

    tf = train_transform()
    assert isinstance(tf, transforms.Compose)
    names = [type(t).__name__ for t in tf.transforms]
    assert "RandomErasing" not in names, (
        f"RandomErasing is in train_transform: {names}. "
        f"It erases identity regions on 160x160 face crops — drop it."
    )
