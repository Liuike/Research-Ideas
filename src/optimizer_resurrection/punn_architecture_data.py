"""Datasets for the 2024 PUNN architecture-stability reconstruction.

The paper's author-provided classification splits and preprocessing were not
published.  This module therefore records an explicit, reproducible
reconstruction: public raw data, seeded 75/25 split, and input scaling fit on
the training split only.  It writes no local dataset or cache files.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch

from .punn_landscape import Dataset, make_dataset


# (input dimension, regular/small hidden width, oversized hidden width,
# output dimension).  The widths follow the architecture table used by the
# paper's classification comparison and the repository's f1/f4 controls.
TASK_SPECS: dict[str, tuple[int, int, int, int]] = {
    "f1": (1, 1, 2, 1),
    "f4": (2, 3, 8, 1),
    "xor": (2, 1, 2, 1),
    "iris": (4, 2, 4, 3),
    "wine": (13, 2, 10, 3),
    "diabetes": (8, 1, 8, 1),
}

_IRIS_CLASSES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
_WINE_CLASSES = ("1", "2", "3")
_DIABETES_CLASSES = ("0", "1")
_PIMA_FEATURES = (
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "serum_insulin", "body_mass_index", "diabetes_pedigree_function", "age",
)


@dataclass(frozen=True)
class RawSource:
    name: str
    url: str
    sha256: str
    citation: str
    expected_rows: int
    classes: tuple[str, ...]
    format: str
    provenance_note: str = ""


RAW_SOURCES: dict[str, RawSource] = {
    "iris": RawSource(
        name="Iris",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        sha256="6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0",
        citation="Fisher (1936), Iris, UCI Machine Learning Repository, DOI 10.24432/C56C76.",
        expected_rows=150,
        classes=_IRIS_CLASSES,
        format="iris_csv",
    ),
    "wine": RawSource(
        name="Wine",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        sha256="6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659",
        citation="Aeberhard and Forina (1992), Wine, UCI Machine Learning Repository, DOI 10.24432/C5PC7J.",
        expected_rows=178,
        classes=_WINE_CLASSES,
        format="wine_csv",
    ),
    "diabetes": RawSource(
        name="Pima Indians Diabetes",
        url=("https://raw.githubusercontent.com/nebiljabari/Pima-Indian-Diabetes-Study/"
             "644641bcb55ac6bf7750c4ab54a26b5e8e922925/pima-indians-diabetes.data"),
        sha256="06f5b7c2cd7bca686fda4f92eab5f61e7ff6426a9acefa2e3dda04fc54293cf5",
        citation=("Smith et al. (1988), Using the ADAP learning algorithm to forecast the onset of "
                  "diabetes mellitus, UCI Pima Indians Diabetes dataset."),
        expected_rows=768,
        classes=_DIABETES_CLASSES,
        format="diabetes_csv",
        provenance_note=("The historical UCI raw endpoint currently returns 404 and its old dataset "
                         "identifier is reused. This content-addressed GitHub mirror is pinned to "
                         "an immutable commit and was cross-checked row-for-row against an "
                         "independent public copy; original UCI origin is retained as provenance."),
    ),
}


def _download_raw(url: str) -> bytes:
    """Retrieve bytes in memory with bounded retries; never write a local cache."""
    request = urllib.request.Request(url, headers={"User-Agent": "PUNN-reproduction/1.0"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as error:  # networking errors vary by platform
            last_error = error
            if attempt < 2:
                time.sleep(0.25 * (2**attempt))
    assert last_error is not None
    raise RuntimeError(f"could not retrieve pinned dataset source ({type(last_error).__name__})") from last_error


def _verified_raw(source: RawSource) -> bytes:
    raw = _download_raw(source.url)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != source.sha256:
        raise ValueError(f"raw dataset SHA-256 mismatch for {source.name}: expected {source.sha256}, got {actual}")
    return raw


def _parse_raw(task: str, source: RawSource, raw: bytes) -> tuple[torch.Tensor, list[str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"raw {source.name} dataset is not UTF-8") from error
    rows = [row for row in csv.reader(io.StringIO(text)) if row and any(cell.strip() for cell in row)]
    input_dim = TASK_SPECS[task][0]
    if len(rows) != source.expected_rows:
        raise ValueError(f"{source.name} row-count mismatch: expected {source.expected_rows}, got {len(rows)}")

    values: list[list[float]] = []
    labels: list[str] = []
    expected_columns = input_dim + 1
    for line_no, row in enumerate(rows, start=1):
        if len(row) != expected_columns:
            raise ValueError(f"{source.name} row {line_no} has {len(row)} columns; expected {expected_columns}")
        if task == "iris":
            feature_cells, label = row[:4], row[4].strip()
        elif task == "wine":
            label, feature_cells = row[0].strip(), row[1:]
        else:  # Pima file stores the eight inputs followed by the binary label.
            feature_cells, label = row[:8], row[8].strip()
        try:
            values.append([float(cell) for cell in feature_cells])
        except ValueError as error:
            raise ValueError(f"{source.name} contains a nonnumeric feature at row {line_no}") from error
        labels.append(label)

    x = torch.tensor(values, dtype=torch.float32)
    if x.shape != (source.expected_rows, input_dim) or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{source.name} feature array has invalid shape or nonfinite values")
    if set(labels) != set(source.classes):
        raise ValueError(f"{source.name} labels differ from the pinned class vocabulary")
    return x, labels


def _digest_dataset(train_x: torch.Tensor, train_y: torch.Tensor,
                    test_x: torch.Tensor, test_y: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (train_x, train_y, test_x, test_y):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assemble_dataset(
    task: str,
    data_seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    classification: bool,
    source: RawSource | None,
    classes: tuple[str, ...],
    label_encoding: str,
    normalize: bool,
    zero_sentinel_counts: list[int] | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    count = len(x)
    if x.ndim != 2 or y.ndim != 2 or len(y) != count or count < 2:
        raise ValueError("dataset must have aligned 2-D inputs and targets")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("dataset contains nonfinite input or target values")

    generator = torch.Generator(device="cpu").manual_seed(data_seed)
    order = torch.randperm(count, generator=generator)
    train_count = int(0.75 * count)
    train_indices, test_indices = order[:train_count], order[train_count:]
    train_x = x[train_indices].clone()
    test_x = x[test_indices].clone()
    train_y, test_y = y[train_indices].clone(), y[test_indices].clone()

    scaler: dict[str, Any]
    if normalize:
        minimum = train_x.amin(dim=0)
        maximum = train_x.amax(dim=0)
        span = maximum - minimum
        constant = span == 0
        safe_span = torch.where(constant, torch.ones_like(span), span)
        train_x = 2.0 * (train_x - minimum) / safe_span - 1.0
        test_x = 2.0 * (test_x - minimum) / safe_span - 1.0
        if bool(constant.any()):
            train_x[:, constant] = 0.0
            test_x[:, constant] = 0.0
        zero_mask_train = train_x == 0
        zero_mask_test = test_x == 0
        affected_train = zero_mask_train.sum(dim=0).tolist()
        affected_test = zero_mask_test.sum(dim=0).tolist()
        train_x[zero_mask_train] = 1e-6
        test_x[zero_mask_test] = 1e-6
        scaler = {
            "method": "train-only min-max to [-1, 1]; test values use training extrema without clipping",
            "training_min": minimum.tolist(),
            "training_max": maximum.tolist(),
            "constant_features": constant.nonzero(as_tuple=False).flatten().tolist(),
            "exact_zero_policy": "replace exact zero normalized inputs by +1e-6; leave all nonzero values unchanged",
            "exact_zero_replacements_train_by_feature": affected_train,
            "exact_zero_replacements_test_by_feature": affected_test,
            "exact_zero_replacements_total": int(sum(affected_train) + sum(affected_test)),
        }
    else:
        zero_mask_train = train_x == 0
        zero_mask_test = test_x == 0
        scaler = {
            "method": "none (inputs already defined in [-1, 1])",
            "exact_zero_policy": "none required; XOR inputs are exactly -1 or +1",
            "exact_zero_replacements_train_by_feature": [0] * x.shape[1],
            "exact_zero_replacements_test_by_feature": [0] * x.shape[1],
            "exact_zero_replacements_total": 0,
        }

    digest = _digest_dataset(train_x, train_y, test_x, test_y)
    dataset = Dataset(train_x, train_y, test_x, test_y, digest)
    metadata: dict[str, Any] = {
        "task": task,
        "task_type": "classification" if classification else "regression",
        "data_seed": data_seed,
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "row_count": count,
        "split": "seeded CPU torch.randperm; first floor(0.75*n) rows train, remainder test",
        "train_count": int(len(train_indices)),
        "test_count": int(len(test_indices)),
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "classes": list(classes),
        "label_encoding": label_encoding,
        "normalization": scaler,
        "dataset_digest": digest,
        "paper_preprocessing": "unavailable; public-data preprocessing and split are explicit reconstruction assumptions",
    }
    if zero_sentinel_counts is not None:
        metadata["raw_zero_values_by_feature"] = zero_sentinel_counts
        metadata["raw_zero_value_feature_names"] = list(_PIMA_FEATURES)
        metadata["potential_pima_zero_sentinel_fields"] = [
            {"feature": _PIMA_FEATURES[index], "count": zero_sentinel_counts[index]}
            for index in (1, 2, 3, 4, 5)
        ]
        metadata["zero_value_policy"] = (
            "keep raw zero values without missing-value imputation; zeros in glucose, blood pressure, "
            "skin thickness, serum insulin, and BMI may be missingness sentinels, while zero pregnancies "
            "is valid. Apply the recorded training-only input scaling afterward."
        )
    if source is not None:
        metadata["raw_source"] = {
            "name": source.name,
            "url": source.url,
            "sha256": source.sha256,
            "citation": source.citation,
            "provenance_note": source.provenance_note,
        }
    else:
        metadata["raw_source"] = {"name": "generated XOR truth table", "url": None, "sha256": None}
    return dataset, metadata


def make_architecture_dataset(task: str, data_seed: int) -> tuple[Dataset, dict[str, Any]]:
    """Return a paired 75/25 dataset and auditable source/preprocessing metadata."""
    if task not in TASK_SPECS:
        raise ValueError(f"unsupported architecture-stability task: {task}")
    if not isinstance(data_seed, int) or data_seed < 0:
        raise ValueError("data_seed must be a nonnegative integer")

    if task in {"f1", "f4"}:
        dataset = make_dataset(task, data_seed)
        input_dim, _small, _oversized, output_dim = TASK_SPECS[task]
        return dataset, {
            "task": task,
            "task_type": "regression",
            "data_seed": data_seed,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "row_count": int(len(dataset.train_x) + len(dataset.test_x)),
            "train_count": int(len(dataset.train_x)),
            "test_count": int(len(dataset.test_x)),
            "split": "existing punn_landscape.make_dataset seeded 75/25 reconstruction",
            "normalization": {"method": "none; generated values already in [-1, 1]"},
            "dataset_digest": dataset.digest,
            "raw_source": {"name": "generated PUNN f1/f4 regression samples", "url": None, "sha256": None},
            "paper_preprocessing": "synthetic inputs generated locally as in the existing f1/f4 reconstruction",
        }

    if task == "xor":
        x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=torch.float32)
        # XOR: equal signs map to class zero; opposite signs map to class one.
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
        return _assemble_dataset(
            task, data_seed, x, y, classification=True, source=None,
            classes=("0", "1"), label_encoding="scalar binary target 0/1",
            normalize=False,
        )

    source = RAW_SOURCES[task]
    raw = _verified_raw(source)
    x, label_text = _parse_raw(task, source, raw)
    class_to_index = {label: index for index, label in enumerate(source.classes)}
    labels = torch.tensor([class_to_index[label] for label in label_text], dtype=torch.long)
    if task in {"iris", "wine"}:
        y = torch.nn.functional.one_hot(labels, num_classes=len(source.classes)).to(torch.float32)
        label_encoding = "one-hot in recorded class order"
    else:
        y = labels.to(torch.float32).unsqueeze(1)
        label_encoding = "scalar binary target 0/1"
    raw_zero_counts = (x == 0).sum(dim=0).tolist() if task == "diabetes" else None
    return _assemble_dataset(
        task, data_seed, x, y, classification=True, source=source,
        classes=source.classes, label_encoding=label_encoding, normalize=True,
        zero_sentinel_counts=raw_zero_counts,
    )

