import torch

from optimizer_resurrection.tasks import OnlineShapeSet, ShapeSetDataset, make_sequence_batch


def test_sequence_tasks_are_reproducible_and_padded():
    for task in ("latch", "two_sequence", "parity"):
        first = make_sequence_batch(task, 8, 20, torch.Generator().manual_seed(7))
        second = make_sequence_batch(task, 8, 20, torch.Generator().manual_seed(7))
        assert torch.equal(first.inputs, second.inputs)
        assert torch.equal(first.targets, second.targets)
        assert first.lengths.min() >= 10
        for row, length in zip(first.inputs, first.lengths):
            assert torch.count_nonzero(row[int(length) :]) == 0


def test_latch_target_is_encoded_at_first_step():
    batch = make_sequence_batch("latch", 16, 12, torch.Generator().manual_seed(3))
    decoded = (batch.inputs[:, 0, 0] > 0).long()
    assert torch.equal(decoded, batch.targets)
    assert torch.all(batch.inputs[:, 0, 1] == 1)


def test_parity_target_matches_sequence():
    batch = make_sequence_batch("parity", 16, 15, torch.Generator().manual_seed(4))
    positives = (batch.inputs[:, :, 0] > 0).sum(dim=1) % 2
    assert torch.equal(positives, batch.targets)


def test_shapeset_has_nine_deterministic_classes():
    dataset = ShapeSetDataset(size=100, seed=11)
    image, label = dataset[5]
    image_again, label_again = dataset[5]
    assert image.shape == (1, 32, 32)
    assert 0 <= int(label) < 9
    assert torch.equal(image, image_again)
    assert torch.equal(label, label_again)


def test_online_shapeset_does_not_recycle_after_nominal_epoch():
    stream = iter(OnlineShapeSet(seed=2))
    first = next(stream)
    for _ in range(8):
        next(stream)
    later = next(stream)
    assert not torch.equal(first[0], later[0])
