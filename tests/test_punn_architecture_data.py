import hashlib
import itertools

import pytest
import torch

from optimizer_resurrection import punn_architecture_data as data


def _fake_source(monkeypatch, task, rows, classes):
    raw = ("\n".join(",".join(map(str, row)) for row in rows) + "\n").encode()
    source = data.RawSource(
        name=f"fixture-{task}", url="https://fixture.invalid/data.csv",
        sha256=hashlib.sha256(raw).hexdigest(), citation="test fixture",
        expected_rows=len(rows), classes=tuple(classes), format=f"{task}_csv",
    )
    monkeypatch.setitem(data.RAW_SOURCES, task, source)
    monkeypatch.setattr(data, "_download_raw", lambda _url: raw)
    return source


@pytest.mark.parametrize(
    ("task", "classes", "target_builder", "expected_outputs"),
    [
        ("iris", ("Iris-setosa", "Iris-versicolor", "Iris-virginica"),
         lambda i: ("Iris-setosa", "Iris-versicolor", "Iris-virginica")[i % 3], 3),
        ("wine", ("1", "2", "3"), lambda i: str(i % 3 + 1), 3),
        ("diabetes", ("0", "1"), lambda i: str(i % 2), 1),
    ],
)
def test_real_classification_sources_parse_split_and_repeat_offline(
    monkeypatch, task, classes, target_builder, expected_outputs
):
    input_dim = data.TASK_SPECS[task][0]
    rows = []
    for index in range(12):
        features = [str(index * (column + 1) + column + 1) for column in range(input_dim)]
        label = target_builder(index)
        rows.append([*features, label] if task != "wine" else [label, *features])
    source = _fake_source(monkeypatch, task, rows, classes)

    first, metadata = data.make_architecture_dataset(task, 37)
    second, metadata_again = data.make_architecture_dataset(task, 37)
    assert first.digest == second.digest == metadata["dataset_digest"]
    assert metadata == metadata_again
    assert (first.train_x.shape, first.test_x.shape) == ((9, input_dim), (3, input_dim))
    assert first.train_y.shape == (9, expected_outputs)
    assert first.test_y.shape == (3, expected_outputs)
    assert metadata["raw_source"]["sha256"] == source.sha256
    assert metadata["raw_source"]["url"] == source.url
    assert metadata["train_indices"] == metadata_again["train_indices"]
    assert bool(torch.isfinite(first.train_x).all() and torch.isfinite(first.test_x).all())
    if expected_outputs == 3:
        torch.testing.assert_close(first.train_y.sum(dim=1), torch.ones(9))
        torch.testing.assert_close(first.test_y.sum(dim=1), torch.ones(3))
    else:
        assert set(first.train_y.flatten().tolist() + first.test_y.flatten().tolist()) <= {0.0, 1.0}


def test_scaler_uses_training_extrema_only_and_zero_policy_is_exact():
    # Find a deterministic split where the outlier is held out. Its scaled
    # value should exceed +1, proving the test set did not fit or refit scale.
    x = torch.tensor([[0.0], [2.0], [4.0], [100.0]])
    y = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    seed = next(
        candidate for candidate in range(100)
        if torch.randperm(4, generator=torch.Generator().manual_seed(candidate))[-1].item() == 3
    )
    dataset, metadata = data._assemble_dataset(
        "diabetes", seed, x, y, classification=True, source=None,
        classes=("0", "1"), label_encoding="scalar binary target 0/1",
        normalize=True,
    )
    assert metadata["normalization"]["training_min"] == [0.0]
    assert metadata["normalization"]["training_max"] == [4.0]
    assert float(dataset.test_x.max()) > 1.0
    assert metadata["normalization"]["exact_zero_replacements_train_by_feature"] == [1]
    assert metadata["normalization"]["exact_zero_replacements_total"] == 1
    assert bool(torch.any(dataset.train_x.flatten() == torch.tensor(1e-6, dtype=torch.float32)))
    assert metadata["normalization"]["exact_zero_policy"].startswith("replace exact zero")


def test_pima_zero_sentinels_are_counted_without_imputation(monkeypatch):
    rows = [
        [0, 90, 60, 20, 0, 24.0, .1, 21, 0],
        [2, 100, 70, 30, 10, 30.0, .2, 25, 1],
        [1, 110, 80, 25, 20, 35.0, .3, 35, 0],
        [4, 120, 90, 40, 30, 40.0, .4, 45, 1],
    ]
    _fake_source(monkeypatch, "diabetes", rows, ("0", "1"))
    dataset, metadata = data.make_architecture_dataset("diabetes", 4)
    assert metadata["raw_zero_values_by_feature"][0] == 1
    assert metadata["raw_zero_values_by_feature"][4] == 1
    assert metadata["potential_pima_zero_sentinel_fields"][3] == {
        "feature": "serum_insulin", "count": 1,
    }
    assert "without missing-value imputation" in metadata["zero_value_policy"]
    assert dataset.train_y.shape[1] == dataset.test_y.shape[1] == 1


def test_checksum_mismatch_is_rejected_before_parsing(monkeypatch):
    monkeypatch.setattr(data, "_download_raw", lambda _url: b"modified pinned content")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data._verified_raw(data.RAW_SOURCES["iris"])


def test_xor_table_is_exact_and_seeded_split_is_paired():
    first, metadata = data.make_architecture_dataset("xor", 18)
    second, _ = data.make_architecture_dataset("xor", 18)
    assert first.digest == second.digest
    assert first.train_x.shape == (3, 2) and first.test_x.shape == (1, 2)
    assert first.train_y.shape == (3, 1) and first.test_y.shape == (1, 1)
    all_x = torch.cat((first.train_x, first.test_x))
    all_y = torch.cat((first.train_y, first.test_y)).flatten()
    assert {tuple(row.tolist()) for row in all_x} == set(itertools.product((-1.0, 1.0), repeat=2))
    assert all(float(label) == float(left != right) for (left, right), label in zip(all_x.tolist(), all_y))
    assert metadata["normalization"]["exact_zero_replacements_total"] == 0


@pytest.mark.parametrize("task", ["f1", "f4"])
def test_regression_datasets_reuse_existing_reconstruction(task):
    existing = data.make_dataset(task, 100037)
    actual, metadata = data.make_architecture_dataset(task, 100037)
    assert actual.digest == existing.digest
    assert metadata["dataset_digest"] == existing.digest
    assert metadata["task_type"] == "regression"
    assert actual.train_y.shape[1] == actual.test_y.shape[1] == 1

