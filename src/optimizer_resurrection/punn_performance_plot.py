"""Final-performance plots for the four-method PUNN comparison."""

from __future__ import annotations

import hashlib
import math
from typing import Any


METHODS = ("sgd", "adamw", "muon_moonlight", "manifold_muon_da10")
METHOD_LABELS = {
    "sgd": "SGD",
    "adamw": "AdamW",
    "muon_moonlight": "Muon",
    "manifold_muon_da10": "MM-DA10",
}
METHOD_COLORS = {
    "sgd": "#0072b2",
    "adamw": "#e69f00",
    "muon_moonlight": "#009e73",
    "manifold_muon_da10": "#cc79a7",
}
ARCHITECTURES = ("small", "oversized", "regularized")
ARCHITECTURE_LABELS = {
    "small": "Small",
    "oversized": "Oversized",
    "regularized": "L2-regularized",
}
TASKS = ("f1", "f4", "xor", "iris", "wine", "diabetes")
DISPLAY_FLOOR = 1e-12
SEEDS_PER_CELL = 30


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _jitter(seed: Any, run_id: Any) -> float:
    """Return stable horizontal jitter in [-0.12, 0.12] for a completed run."""
    key = f"{seed!r}\0{run_id!r}".encode("utf-8")
    raw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    unit = raw / (2**64 - 1)
    return (unit - 0.5) * 0.24


def plot_final_performance(
    report: dict[str, Any], config: dict[str, Any], artifact: Any, metric: str,
) -> None:
    """Write the six-panel final train or held-out MSE comparison to a W&B artifact."""
    if metric not in {"train_mse", "test_mse"}:
        raise ValueError("metric must be 'train_mse' or 'test_mse'")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tasks = tuple(dict.fromkeys(condition["task"] for condition in config["conditions"]))
    if tasks != TASKS:
        raise ValueError(f"expected six tasks in report order {TASKS}, got {tasks}")

    rows_by_cell = {
        (row["task"], row["architecture"]): row
        for row in report["rows"]
    }
    figure, axes = plt.subplots(
        2, 3, figsize=(18, 10.6), squeeze=False,
    )
    figure.subplots_adjust(
        left=0.055, right=0.99, top=0.88, bottom=0.24,
        hspace=0.45, wspace=0.22,
    )

    for axis, task in zip(axes.flat, tasks):
        axis.set_yscale("log")
        architectures = [
            architecture for architecture in ARCHITECTURES
            if (task, architecture) in rows_by_cell
        ]
        positions: list[float] = []
        tick_labels: list[str] = []
        task_display_values: list[float] = []
        for arch_index, architecture in enumerate(architectures):
            row = rows_by_cell[(task, architecture)]
            for method_index, method in enumerate(METHODS):
                position = arch_index * (len(METHODS) + 1) + method_index
                positions.append(position)
                tick_labels.append(METHOD_LABELS[method])

                method_row = row[method]
                performance = method_row["final_performance"]
                attempted = method_row["attempted"]
                if isinstance(attempted, bool) or attempted != SEEDS_PER_CELL:
                    raise ValueError(
                        f"expected {SEEDS_PER_CELL} attempts for {task}/{architecture}/{method}, "
                        f"got {attempted!r}"
                    )
                completed = sorted(
                    performance["completed"],
                    key=lambda item: (item.get("seed", 0), item.get("run_id", "")),
                )
                completed_count = method_row["finite_completions"]
                if (isinstance(completed_count, bool)
                        or not isinstance(completed_count, int)
                        or len(completed) != completed_count):
                    raise ValueError(
                        f"completed records/count disagree for {task}/{architecture}/{method}: "
                        f"{len(completed)} records, count {completed_count!r}"
                    )

                points = []
                for sample in completed:
                    value = _as_finite_float(sample.get(metric))
                    if value is None or value < 0:
                        raise ValueError(
                            f"invalid {metric} in completed run {sample.get('run_id')!r}: "
                            f"{sample.get(metric)!r}"
                        )
                    shown_value = max(value, DISPLAY_FLOOR)
                    task_display_values.append(shown_value)
                    points.append((sample, shown_value))

                summary = performance[metric]
                if summary.get("count") != completed_count:
                    raise ValueError(
                        f"{metric} summary/count disagree for {task}/{architecture}/{method}: "
                        f"{summary.get('count')!r} versus {completed_count}"
                    )
                summary_values = {}
                if completed_count:
                    for name in ("median", "q25", "q75"):
                        value = _as_finite_float(summary.get(name))
                        if value is None or value < 0:
                            raise ValueError(
                                f"invalid {metric} {name} for {task}/{architecture}/{method}: "
                                f"{summary.get(name)!r}"
                            )
                        summary_values[name] = max(value, DISPLAY_FLOOR)
                        task_display_values.append(summary_values[name])

                axis.text(
                    position, 0.985, f"{completed_count}/{SEEDS_PER_CELL}",
                    transform=axis.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.5, color="#333333", clip_on=False,
                )

                color = METHOD_COLORS[method]
                if completed_count == 0:
                    axis.scatter(
                        [position], [0.06], transform=axis.get_xaxis_transform(),
                        marker="x", color="#888888", s=25, linewidths=1.2, zorder=4,
                    )
                    axis.annotate(
                        "no data", (position, 0.06), xytext=(0, 5),
                        xycoords=axis.get_xaxis_transform(), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, color="#777777",
                    )
                    continue

                axis.scatter(
                    [position + _jitter(sample.get("seed"), sample.get("run_id"))
                     for sample, _value in points],
                    [value for _sample, value in points],
                    s=15, color=color, alpha=0.7, edgecolors="none", zorder=2,
                )

                median = summary_values["median"]
                q25 = summary_values["q25"]
                q75 = summary_values["q75"]
                axis.vlines(position, q25, q75, color=color, linewidth=2.2, zorder=3)
                axis.hlines(
                    median, position - 0.18, position + 0.18,
                    color=color, linewidth=2.2, zorder=3,
                )

            first = arch_index * (len(METHODS) + 1)
            last = first + len(METHODS) - 1
            center = (first + last) / 2
            axis.plot(
                [first - 0.35, last + 0.35], [-0.20, -0.20],
                transform=axis.get_xaxis_transform(), color="#777777",
                linewidth=0.7, clip_on=False,
            )
            label = ARCHITECTURE_LABELS[architecture]
            if task == "f1" and architecture == "small":
                label += "†"
            axis.text(
                center, -0.23, label, transform=axis.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, clip_on=False,
            )

        axis.set_xticks(positions, tick_labels)
        axis.tick_params(axis="x", labelsize=7, pad=2)
        axis.set_title(task)
        axis.set_ylabel("Final MSE (log scale)")
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=0.55)
        axis.set_axisbelow(True)
        if task_display_values:
            log_min = math.log10(min(task_display_values))
            log_max = math.log10(max(task_display_values))
            log_span = max(log_max - log_min, 1.0)
            bottom = max(DISPLAY_FLOOR, 10 ** (log_min - 0.08 * log_span))
            top = 10 ** (log_max + 0.18 * log_span)
        else:
            bottom = DISPLAY_FLOOR
            top = DISPLAY_FLOOR * 100
        axis.set_ylim(bottom=bottom, top=top)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=METHOD_COLORS[method],
               markersize=5, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.08),
        ncol=4, fontsize=8,
    )
    metric_title = (
        "Final unregularized training MSE" if metric == "train_mse"
        else "Final held-out test MSE"
    )
    figure.suptitle(
        f"{metric_title} after 500 epochs | lower is better\n"
        "FP32, frozen recipes",
        fontsize=13,
    )
    figure.text(
        0.5, 0.008,
        "Dots: completed seeds; colored marks: median and IQR; n/30: completed seeds out of 30.\n"
        "Failed scientific runs omitted; log display floor 1e-12. "
        "Stiefel projection changes model capacity; † f1/small is below-capacity.",
        ha="center", va="bottom", fontsize=7.5,
    )

    filename = "final_train_performance.png" if metric == "train_mse" else "final_test_performance.png"
    with artifact.new_file(filename, mode="wb") as handle:
        figure.savefig(handle, format="png", dpi=160)
    plt.close(figure)
