"""Plots for the recorded DA-10 loss-landscape audit report.

This module consumes an already validated report.  It performs no data
retrieval and writes images only through the supplied W&B artifact handle.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable


TASKS = ("f1", "f4", "xor", "iris", "wine", "diabetes")
CELL_ARCHITECTURES = {
    "f1": ("small", "oversized"),
    "f4": ("small", "oversized"),
    "xor": ("small", "oversized", "regularized"),
    "iris": ("small", "oversized", "regularized"),
    "wine": ("small", "oversized", "regularized"),
    "diabetes": ("small", "oversized", "regularized"),
}
ARCH_LABELS = {
    "small": "Small",
    "oversized": "Oversized",
    "regularized": "Regularized",
}
ARCH_COLORS = {
    "small": "#0072b2",
    "oversized": "#d55e00",
    "regularized": "#009e73",
}
VIEW_COLORS = {"ambient": "#0072b2", "manifold": "#d55e00"}
METHODS = ("sgd", "adamw", "muon_moonlight", "manifold_muon_da10")
METHOD_LABELS = {
    "sgd": "SGD",
    "adamw": "AdamW",
    "muon_moonlight": "Moonlight Muon",
    "muon": "Moonlight Muon",
    "manifold_muon_da10": "Manifold Muon DA-10",
}
METHOD_COLORS = {
    "sgd": "#0072b2",
    "adamw": "#e69f00",
    "muon_moonlight": "#009e73",
    "muon": "#009e73",
    "manifold_muon_da10": "#cc79a7",
}
EXPECTED_SEEDS = 30
EXPECTED_BASELINE_SEEDS = 10
POSITIVE_FLOOR = 1e-15
ABS_METRIC = "absolute_p95_mse_delta"
SIGNED_METRIC = "signed_p95_mse_delta"
SENSITIVITY_LABEL = "95th percentile |MSE - center MSE|"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _display_positive(value: float) -> float:
    """Keep true zero sensitivities visible on the requested log axes."""
    return max(float(value), POSITIVE_FLOOR)


def _set_log_limits(axis: Any, values: Iterable[float], *, has_zero: bool = False) -> None:
    """Tighten a positive log axis, reserving the fixed floor only for zeros."""
    values = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0]
    if not values:
        axis.set_ylim(POSITIVE_FLOOR / 2, 1.0)
        return
    low = POSITIVE_FLOOR / 2 if has_zero else min(values) / 1.8
    high = max(values) * 1.8
    if high <= low:
        high = low * 2
    axis.set_ylim(bottom=low, top=high)


def _quantile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _summary(values: Iterable[float]) -> tuple[float, float, float] | None:
    values = list(values)
    if not values:
        return None
    return (
        float(_quantile(values, 0.25)),
        float(_quantile(values, 0.50)),
        float(_quantile(values, 0.75)),
    )


def _jitter(seed: Any) -> float:
    digest = hashlib.blake2b(str(seed).encode("utf-8"), digest_size=4).digest()
    unit = int.from_bytes(digest, "big") / (2**32 - 1)
    return (unit - 0.5) * 0.13


def _condition(row: dict[str, Any]) -> dict[str, Any]:
    condition = row.get("condition")
    if not isinstance(condition, dict):
        raise ValueError("each landscape report row must have a condition mapping")
    missing = {"task", "architecture", "method", "seed"} - set(condition)
    if missing:
        raise ValueError(f"landscape report condition is missing fields: {sorted(missing)}")
    return condition


def _indexed_runs(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        condition = _condition(row)
        if condition.get("method") != "manifold_muon_da10":
            continue
        key = (str(condition["task"]), str(condition["architecture"]))
        logical = (*key, str(condition["method"]), str(condition["seed"]))
        if logical in seen:
            raise ValueError(f"duplicate Manifold Muon DA-10 seed in plot report: {logical}")
        seen.add(logical)
        by_cell[key].append(row)
    return by_cell


def _slice(row: dict[str, Any], view: str, phase: str, epoch: int | None = None) -> dict[str, Any] | None:
    matches = [
        item for item in row.get("slices", [])
        if item.get("view") == view
        and item.get("phase") == phase
        and (epoch is None or item.get("epoch") == epoch)
    ]
    if len(matches) > 1:
        raise ValueError(
            "duplicate landscape slice for "
            f"{row.get('condition', {})}, {view}, {phase}, {epoch}"
        )
    return matches[0] if matches else None


def _device_note(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        condition = row.get("condition", {})
        device = condition.get("training_device", row.get("training_device", "unknown"))
        counts[str(device)] += 1
    if not counts:
        return "Training device not reported"
    return "Training device records: " + ", ".join(
        f"{count} {device}" for device, count in sorted(counts.items())
    )


def _finish_figure(figure: Any, artifact: Any, filename: str, *, dpi: int = 160) -> None:
    with artifact.new_file(filename, mode="wb") as handle:
        figure.savefig(handle, format="png", dpi=dpi, facecolor="white")


def _terminal_distribution_figure(report: dict[str, Any], rows: list[dict[str, Any]], artifact: Any) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_cell = _indexed_runs(rows)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.20, hspace=0.33, wspace=0.24)

    for axis, task in zip(axes.flat, TASKS):
        architectures = CELL_ARCHITECTURES[task]
        ticks: list[float] = []
        labels: list[str] = []
        ymax_values: list[float] = []
        any_zero = False
        for arch_index, architecture in enumerate(architectures):
            cell_rows = by_cell.get((task, architecture), [])
            center = arch_index * 3.0
            ambient_x = center - 0.38
            manifold_x = center + 0.38
            ticks.append(center)
            labels.append(ARCH_LABELS[architecture])
            values_by_view: dict[str, dict[str, float]] = {"ambient": {}, "manifold": {}}
            nonfinite_by_view = {"ambient": 0, "manifold": 0}
            zero_by_view = {"ambient": 0, "manifold": 0}

            for row in cell_rows:
                seed = _condition(row)["seed"]
                for view in ("ambient", "manifold"):
                    record = _slice(row, view, "terminal")
                    if record and int(record.get("mse_nonfinite_count") or 0) > 0:
                        nonfinite_by_view[view] += 1
                    value = _finite_number(record.get(ABS_METRIC)) if record else None
                    if value is not None and value >= 0:
                        if value == 0:
                            zero_by_view[view] += 1
                            any_zero = True
                        values_by_view[view][str(seed)] = _display_positive(value)

            seeds = sorted(set(values_by_view["ambient"]) | set(values_by_view["manifold"]))
            for seed in seeds:
                offset = _jitter(seed)
                left = values_by_view["ambient"].get(seed)
                right = values_by_view["manifold"].get(seed)
                if left is not None and right is not None:
                    axis.plot(
                        [ambient_x + offset, manifold_x + offset], [left, right],
                        color="#888888", alpha=0.20, linewidth=0.55, zorder=1,
                    )
                if left is not None:
                    axis.scatter(ambient_x + offset, left, s=12, color=VIEW_COLORS["ambient"], alpha=0.58, zorder=2)
                    ymax_values.append(left)
                if right is not None:
                    axis.scatter(manifold_x + offset, right, s=12, color=VIEW_COLORS["manifold"], alpha=0.58, zorder=2)
                    ymax_values.append(right)

            for view, x_position in (("ambient", ambient_x), ("manifold", manifold_x)):
                summary = _summary(values_by_view[view].values())
                if summary:
                    q25, median, q75 = (_display_positive(value) for value in summary)
                    axis.vlines(x_position, q25, q75, color=VIEW_COLORS[view], linewidth=2.0, zorder=4)
                    axis.hlines(median, x_position - 0.16, x_position + 0.16,
                                color=VIEW_COLORS[view], linewidth=2.2, zorder=5)

            missing_ambient = EXPECTED_SEEDS - len(values_by_view["ambient"])
            missing_manifold = EXPECTED_SEEDS - len(values_by_view["manifold"])
            annotation = (
                f"A {len(values_by_view['ambient'])}/{EXPECTED_SEEDS}; "
                f"M {len(values_by_view['manifold'])}/{EXPECTED_SEEDS}\n"
                f"missing {missing_ambient}/{missing_manifold}; "
                f"NF {nonfinite_by_view['ambient']}/{nonfinite_by_view['manifold']}\n"
                f"zero→floor {zero_by_view['ambient']}/{zero_by_view['manifold']}"
            )
            axis.text(center, 0.98, annotation, transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=6.3, color="#333333")

        axis.set_title(task.upper())
        axis.set_xticks(ticks, labels)
        axis.set_xlim(-1.15, max(ticks, default=0) + 1.15)
        axis.set_yscale("log")
        axis.set_ylabel(SENSITIVITY_LABEL)
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=0.55)
        _set_log_limits(axis, ymax_values, has_zero=any_zero)

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color=VIEW_COLORS["ambient"], label="Ambient; dots are seeds"),
        Line2D([0], [0], marker="o", linestyle="none", color=VIEW_COLORS["manifold"], label="Manifold; dots are seeds"),
        Line2D([0], [0], color="#333333", linewidth=2.2, label="Median; line spans IQR"),
        Line2D([0], [0], color="#888888", linewidth=0.8, alpha=0.5, label="Paired seed"),
    ]
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=4, frameon=False, fontsize=8)
    figure.suptitle("Sampled 2D slice sensitivity — terminal snapshots", fontsize=15)
    figure.text(
        0.5, 0.025,
        f"{_device_note(rows)}. DA-10 projects updates onto its constrained product-unit manifold; "
        "f1-small is below capacity. These sampled 2D slices do not estimate full-space flatness. "
        "Unavailable counts include missing or nonfinite terminal values; NF counts are runs with nonfinite grid points. "
        "Exact zero sensitivities are clamped to 1e-15 on log axes and counted in each cell.",
        ha="center", va="bottom", fontsize=8, wrap=True,
    )
    filename = "da10_terminal_landscape_sensitivity.png"
    _finish_figure(figure, artifact, filename)
    plt.close(figure)
    return filename


def _temporal_figure(rows: list[dict[str, Any]], artifact: Any) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_cell = _indexed_runs(rows)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.19, hspace=0.34, wspace=0.24)
    epochs = (0, 1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)

    for axis, task in zip(axes.flat, TASKS):
        terminal_notes: list[str] = []
        positive_values: list[float] = []
        plotted_bounds: list[float] = []
        snapshot_counts: list[int] = []
        any_zero = False
        for architecture in CELL_ARCHITECTURES[task]:
            cell_rows = by_cell.get((task, architecture), [])
            total = EXPECTED_SEEDS
            for view in ("ambient", "manifold"):
                grouped: dict[int, list[float]] = defaultdict(list)
                terminal_values: list[float] = []
                nonfinite_epoch500 = 0
                nonfinite_terminal = 0
                zero_epoch500 = 0
                zero_terminal = 0
                for row in cell_rows:
                    for epoch in epochs:
                        phase = "initialization" if epoch == 0 else "epoch"
                        record = _slice(row, view, phase, epoch)
                        value = _finite_number(record.get(ABS_METRIC)) if record else None
                        if value is not None and value >= 0:
                            if epoch == 500 and value == 0:
                                zero_epoch500 += 1
                                any_zero = True
                            grouped[epoch].append(_display_positive(value))
                            positive_values.append(_display_positive(value))
                        if (epoch == 500 and record
                                and int(record.get("mse_nonfinite_count") or 0) > 0):
                            nonfinite_epoch500 += 1
                    terminal = _slice(row, view, "terminal")
                    value = _finite_number(terminal.get(ABS_METRIC)) if terminal else None
                    if value is not None and value >= 0:
                        if value == 0:
                            zero_terminal += 1
                            any_zero = True
                        terminal_values.append(_display_positive(value))
                        positive_values.append(_display_positive(value))
                    if terminal and int(terminal.get("mse_nonfinite_count") or 0) > 0:
                        nonfinite_terminal += 1

                available_epochs = sorted(grouped)
                snapshot_counts.extend(len(grouped.get(epoch, [])) for epoch in epochs)
                snapshot_counts.append(len(terminal_values))
                if available_epochs:
                    medians = [_summary(grouped[epoch])[1] for epoch in available_epochs]
                    q25 = [_summary(grouped[epoch])[0] for epoch in available_epochs]
                    q75 = [_summary(grouped[epoch])[2] for epoch in available_epochs]
                    plotted_bounds.extend(medians + q25 + q75)
                    color = ARCH_COLORS[architecture]
                    linestyle = "-" if view == "ambient" else "--"
                    axis.plot(available_epochs, medians, color=color, linestyle=linestyle,
                              marker="o", markersize=2.7, linewidth=1.35, alpha=0.95, zorder=3)
                    axis.fill_between(available_epochs, q25, q75, color=color, alpha=0.10, linewidth=0)
                summary = _summary(terminal_values)
                if summary:
                    terminal_median = _display_positive(summary[1])
                    plotted_bounds.append(terminal_median)
                    axis.scatter(
                        [500], [terminal_median], marker="s", s=42, facecolors="none",
                        edgecolors=ARCH_COLORS[architecture], linewidths=1.4, zorder=6,
                    )
                terminal_notes.append(
                    f"{ARCH_LABELS[architecture]} {view}: "
                    f"epoch500 {len(grouped.get(500, []))}/{total}, "
                    f"terminal {len(terminal_values)}/{total}; "
                    f"NF {nonfinite_epoch500}/{nonfinite_terminal}; "
                    f"zero→floor {zero_epoch500}/{zero_terminal}"
                )

        axis.set_title(task.upper())
        axis.set_xlabel("Training epoch")
        axis.set_ylabel(SENSITIVITY_LABEL)
        axis.set_xlim(-12, 512)
        axis.set_xticks((0, 100, 200, 300, 400, 500))
        axis.set_yscale("log")
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=0.55)
        _set_log_limits(axis, plotted_bounds, has_zero=any_zero)
        task_rows = [r for r in rows if _condition(r)["task"] == task]
        nf_slices = sum(int(s.get("mse_nonfinite_count", 0)) > 0 for r in task_rows for s in r["slices"])
        if snapshot_counts and all(n == EXPECTED_SEEDS for n in snapshot_counts) and not any_zero and not nf_slices:
            note = "Every architecture/view/checkpoint: 30/30\nNonfinite grids: 0; zero sensitivities: 0"
        else:
            note = "Terminal counts (epoch500 / terminal; NF / zeros):\n" + "\n".join(terminal_notes)
        axis.text(0.01, 0.015, note, transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=5.1, color="#333333", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.5})

    legend: list[Any] = []
    for architecture in ("small", "oversized", "regularized"):
        legend.append(Line2D([0], [0], color=ARCH_COLORS[architecture], linewidth=1.7,
                             label=ARCH_LABELS[architecture]))
    legend.extend([
        Line2D([0], [0], color="#444444", linestyle="-", label="Ambient median; solid"),
        Line2D([0], [0], color="#444444", linestyle="--", label="Manifold median; dashed"),
        Line2D([0], [0], marker="s", markerfacecolor="none", color="#444444", linestyle="none",
               label="Terminal snapshot at epoch 500"),
    ])
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=3, frameon=False, fontsize=8)
    figure.suptitle("Sampled 2D slice sensitivity over training", fontsize=15)
    figure.text(
        0.5, 0.025,
        "Lines show medians and shaded bands show IQR across finite recorded slices at initialization, "
        "epoch 1 and every 50 epochs. The outlined square is the separately recorded terminal snapshot; "
        "its count and the epoch-500 count are printed per architecture/view. Runs with missing or "
        "nonfinite sensitivity values are excluded from that point's summary and remain visible in n/30 counts; "
        "runs with nonfinite grid candidates and exact zeros clamped to 1e-15 are counted separately. "
        "Axes cover the plotted median/IQR; individual terminal tails are shown in the companion distribution graph. "
        "This is a sampled 2D view, not a full-space flatness estimate.",
        ha="center", va="bottom", fontsize=8, wrap=True,
    )
    filename = "da10_landscape_sensitivity_over_epochs.png"
    _finish_figure(figure, artifact, filename)
    plt.close(figure)
    return filename


def _method_for_baseline(row: dict[str, Any]) -> str | None:
    condition = row.get("condition", {})
    method = condition.get("method", row.get("method"))
    aliases = {"muon": "muon_moonlight", "moonlight_muon": "muon_moonlight"}
    method = aliases.get(str(method), str(method)) if method is not None else None
    return method if method in METHODS else None


def _optimizer_rows(report: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in runs:
        condition = _condition(row)
        if condition.get("task") in {"f1", "f4"} and condition.get("architecture") == "small":
            selected.append(row)
    for row in report.get("baselines", []) or []:
        condition = row.get("condition", {})
        if not isinstance(condition, dict):
            continue
        if condition.get("task") not in {"f1", "f4"} or condition.get("architecture") != "small":
            continue
        if _method_for_baseline(row) is not None:
            selected.append(row)
    return selected


def _baseline_comparison_figure(report: dict[str, Any], runs: list[dict[str, Any]], artifact: Any) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods_rows = _optimizer_rows(report, runs)
    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    seen: set[tuple[str, str, str, str]] = set()
    failures: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    zero_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in methods_rows:
        condition = _condition(row)
        task = str(condition["task"])
        seed = str(condition["seed"])
        if seed not in {str(i) for i in range(EXPECTED_BASELINE_SEEDS)}:
            continue
        method = "manifold_muon_da10" if condition.get("method") == "manifold_muon_da10" else _method_for_baseline(row)
        if method is None:
            continue
        logical = (task, "small", method, seed)
        if logical in seen:
            raise ValueError(f"duplicate optimizer baseline seed in plot report: {logical}")
        seen.add(logical)
        failure_status = row.get("terminal_outcome") or condition.get("terminal_outcome")
        if failure_status == "numerical_failure":
            failures[(task, method)]["numerical failure"] += 1
        record = _slice(row, "ambient", "terminal")
        for metric in (ABS_METRIC, SIGNED_METRIC):
            value = _finite_number(record.get(metric)) if record else None
            if value is not None and (metric != ABS_METRIC or value >= 0):
                grouped[(task, metric)][method][seed][metric] = value
                if metric == ABS_METRIC and value == 0:
                    zero_counts[(task, metric, method)] += 1

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.19, hspace=0.38, wspace=0.24)
    for row_index, task in enumerate(("f1", "f4")):
        for column_index, metric in enumerate((ABS_METRIC, SIGNED_METRIC)):
            axis = axes[row_index][column_index]
            method_values = grouped.get((task, metric), {})
            method_position = {method: index for index, method in enumerate(METHODS)}
            common_seeds = set.intersection(*(set(method_values.get(method, {})) for method in METHODS))
            by_seed: dict[str, list[tuple[float, float]]] = defaultdict(list)
            positive_values: list[float] = []
            any_zero = False
            for method in METHODS:
                seed_values = method_values.get(method, {})
                for seed, metrics in seed_values.items():
                    value = metrics.get(metric)
                    if value is None:
                        continue
                    shown = _display_positive(value) if metric == ABS_METRIC else value
                    if metric == ABS_METRIC:
                        positive_values.append(shown)
                        any_zero = any_zero or value == 0
                    x = method_position[method] + _jitter(seed) * 0.45
                    by_seed[seed].append((x, shown))
            for seed, points in by_seed.items():
                points.sort(key=lambda item: item[0])
                if len(points) > 1:
                    axis.plot([point[0] for point in points], [point[1] for point in points],
                              color="#888888", alpha=0.23, linewidth=0.65, zorder=1)
                for method, method_index in method_position.items():
                    for x, y in points:
                        if abs(x - method_index) <= 0.10:
                            axis.scatter(x, y, s=18, color=METHOD_COLORS[method], alpha=0.62, zorder=2)
            count_notes: list[str] = []
            for method in METHODS:
                values = [
                    _display_positive(metrics[metric]) if metric == ABS_METRIC else metrics[metric]
                    for metrics in method_values.get(method, {}).values()
                    if metric in metrics
                ]
                summary = _summary(values)
                if summary:
                    q25, median, q75 = summary
                    x = method_position[method]
                    axis.vlines(x, q25, q75, color=METHOD_COLORS[method], linewidth=2.4, zorder=4)
                    axis.hlines(median, x - 0.14, x + 0.14, color=METHOD_COLORS[method], linewidth=2.8, zorder=5)
                if common_seeds:
                    common_values = [method_values[method][seed][metric] for seed in common_seeds]
                    common_median = _summary(common_values)[1]
                    if metric == ABS_METRIC:
                        common_median = _display_positive(common_median)
                    axis.scatter([method_position[method]], [common_median], marker="D", s=48,
                                 facecolors="white", edgecolors="black", linewidths=1.1, zorder=7)
                recorded = sum(1 for row in methods_rows
                               if str(row.get("condition", {}).get("task")) == task
                               and str(row.get("condition", {}).get("seed")) in {str(i) for i in range(10)}
                               and ((row.get("condition", {}).get("method") == "manifold_muon_da10")
                                    if method == "manifold_muon_da10"
                                    else _method_for_baseline(row) == method))
                available = len(method_values.get(method, {}))
                numerical = failures[(task, method)]["numerical failure"]
                count_notes.append(
                    f"{METHOD_LABELS[method]} {available}/10 finite; "
                    f"{max(0, EXPECTED_BASELINE_SEEDS - recorded)} absent, "
                    f"{numerical} numerical failures"
                    + (f", {zero_counts[(task, metric, method)]} zero→floor"
                       if metric == ABS_METRIC else "")
                )

            axis.set_xticks(range(len(METHODS)), [METHOD_LABELS[item] for item in METHODS], rotation=12, ha="right")
            axis.set_title(f"{task.upper()} — {'absolute' if metric == ABS_METRIC else 'signed'} terminal delta")
            if metric == ABS_METRIC:
                axis.set_yscale("log")
                _set_log_limits(axis, positive_values, has_zero=any_zero)
                axis.set_ylabel(SENSITIVITY_LABEL)
            else:
                axis.axhline(0, color="#555555", linewidth=0.8)
                axis.set_ylabel("95th percentile signed (MSE - center MSE)")
            axis.grid(axis="y", which="both", color="#dddddd", linewidth=0.55)
            axis.text(0.01, 0.98, "\n".join(count_notes), transform=axis.transAxes,
                      ha="left", va="top", fontsize=6.2,
                      bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.4})
            axis.text(.99, .04, f"Diamond: same {len(common_seeds)} completed seeds",
                      transform=axis.transAxes, ha="right", va="bottom", fontsize=7,
                      bbox={"facecolor":"white", "alpha":.82, "edgecolor":"none"})

    figure.suptitle("Sampled 2D slice sensitivity — matched small-architecture seeds 0–9", fontsize=14)
    figure.text(
        0.5, 0.025,
        "Dots are seed-level values; faint lines connect available values for the same seed; colored bars show "
        "median and IQR of each method's available endpoints. Black-outlined diamonds show medians on the "
        "same seeds completed by all four methods; this conditions on SGD survival and does not score its failures. "
        "Earlier SGD, AdamW and Moonlight Muon records use different recipes and CPU reference "
        "landscapes; DA-10 uses a constrained projection. This comparison covers sampled 2D slices only, not "
        "full-space flatness. Missing seeds and numerical failures are counted in each panel.",
        ha="center", va="bottom", fontsize=8, wrap=True,
    )
    filename = "optimizer_baseline_landscape_sensitivity.png"
    _finish_figure(figure, artifact, filename)
    plt.close(figure)
    return filename


def _performance_figure(rows: list[dict[str, Any]], artifact: Any) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_cell = _indexed_runs(rows)
    metrics = (("train_mse", "Train MSE", "#0072b2"), ("test_mse", "Held-out MSE", "#d55e00"))
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.20, hspace=0.34, wspace=0.24)

    for axis, task in zip(axes.flat, TASKS):
        ticks: list[float] = []
        labels: list[str] = []
        ymax_values: list[float] = []
        axis_has_zero = False
        for arch_index, architecture in enumerate(CELL_ARCHITECTURES[task]):
            cell_rows = by_cell.get((task, architecture), [])
            center = arch_index * 3.0
            ticks.append(center)
            labels.append(ARCH_LABELS[architecture])
            values_by_metric: dict[str, dict[str, float]] = {metric: {} for metric, _, _ in metrics}
            zero_counts = {metric: 0 for metric, _, _ in metrics}
            for row in cell_rows:
                seed = str(_condition(row)["seed"])
                for metric, _, _ in metrics:
                    value = _finite_number(row.get(metric))
                    if value is not None and value >= 0:
                        if value == 0:
                            zero_counts[metric] += 1
                            axis_has_zero = True
                        values_by_metric[metric][seed] = _display_positive(value)
            for seed in sorted(set(values_by_metric["train_mse"]) | set(values_by_metric["test_mse"])):
                offset = _jitter(seed)
                train = values_by_metric["train_mse"].get(seed)
                test = values_by_metric["test_mse"].get(seed)
                positions = (center - 0.38 + offset, center + 0.38 + offset)
                point_values = (train, test)
                finite_points = [(positions[index], value) for index, value in enumerate(point_values) if value is not None]
                if len(finite_points) == 2:
                    axis.plot([item[0] for item in finite_points], [item[1] for item in finite_points],
                              color="#888888", alpha=0.17, linewidth=0.55, zorder=1)
                for index, (metric, _, color) in enumerate(metrics):
                    value = point_values[index]
                    if value is not None:
                        axis.scatter(positions[index], value, s=12, color=color, alpha=0.55, zorder=2)
                        ymax_values.append(value)
            for index, (metric, _, color) in enumerate(metrics):
                values = values_by_metric[metric]
                summary = _summary(values.values())
                x = center + (-0.38 if index == 0 else 0.38)
                if summary:
                    q25, median, q75 = (_display_positive(value) for value in summary)
                    axis.vlines(x, q25, q75, color=color, linewidth=2.0, zorder=4)
                    axis.hlines(median, x - 0.15, x + 0.15, color=color, linewidth=2.2, zorder=5)
            train_valid = len(values_by_metric["train_mse"])
            test_valid = len(values_by_metric["test_mse"])
            axis.text(center, 0.98,
                      f"Train {train_valid}/{EXPECTED_SEEDS}; test {test_valid}/{EXPECTED_SEEDS}\n"
                      f"unavailable {EXPECTED_SEEDS-train_valid}/{EXPECTED_SEEDS-test_valid}\n"
                      f"zero→floor {zero_counts['train_mse']}/{zero_counts['test_mse']}",
                      transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=6.6)

        axis.set_title(task.upper())
        axis.set_xticks(ticks, labels)
        axis.set_xlim(-1.15, max(ticks, default=0) + 1.15)
        axis.set_yscale("log")
        axis.set_ylabel("Prediction MSE")
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=0.55)
        _set_log_limits(axis, ymax_values, has_zero=axis_has_zero)

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=f"{label}; dots are seeds")
        for _, label, color in metrics
    ] + [Line2D([0], [0], color="#333333", linewidth=2.2, label="Median; line spans IQR")]
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=3, frameon=False, fontsize=8)
    figure.suptitle("Manifold Muon DA-10 terminal prediction MSE", fontsize=15)
    figure.text(
        0.5, 0.025,
        f"{_device_note(rows)}. Metrics are taken from the recorded terminal training and held-out evaluations. "
        "Unavailable values remain counted in the annotations. Exact zeros are clamped to 1e-15 on log axes "
        "and their counts are printed per task/architecture.",
        ha="center", va="bottom", fontsize=8, wrap=True,
    )
    filename = "da10_terminal_prediction_mse.png"
    _finish_figure(figure, artifact, filename)
    plt.close(figure)
    return filename


def _terminal_surface_figure(report: dict[str, Any], artifact: Any) -> str:
    """Show actual retained grids using the lowest shared completed seed."""
    from pathlib import Path
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cache = Path(__file__).resolve().parents[2] / "wandb/artifacts/punn-manifold-final-readback"
    rows = _optimizer_rows(report, report["runs"])
    figure, axes = plt.subplots(2, 5, figsize=(17, 8), squeeze=False)
    figure.subplots_adjust(left=.055, right=.93, top=.88, bottom=.19, wspace=.20, hspace=.35)
    selected_seeds = []
    for row_index, task in enumerate(("f1", "f4")):
        matched = {}
        for method in METHODS:
            matched[method] = {int(r["condition"]["seed"]): r for r in rows
                               if r["condition"]["task"] == task and r["condition"]["method"] == method
                               and int(r["condition"]["seed"]) < 10 and r["terminal_outcome"] == "completed"}
        shared = set.intersection(*(set(v) for v in matched.values()))
        if not shared:
            raise ValueError(f"no completed seed shared by the four surface methods for {task}")
        seed = min(shared)
        selected_seeds.append(f"{task.upper()}: seed {seed}")
        grids = []
        for method in METHODS:
            record = matched[method][seed]
            is_da10 = method == "manifold_muon_da10"
            folder = cache / ("runs" if is_da10 else "baselines") / record["run_id"]
            filename = "recorded_landscapes.npz" if is_da10 else "training_landscapes.npz"
            with np.load(folder / filename, allow_pickle=False) as payload:
                if is_da10:
                    for view in ("ambient", "manifold"):
                        slice_row = _slice(record, view, "terminal")
                        index = next(i for i, item in enumerate(record["slices"]) if item is slice_row)
                        grids.append(payload[f"slices/{index:04d}/mse"].copy())
                else:
                    metadata = json.loads(str(payload["metadata_json"]))
                    index = next(key.split("/")[1] for key, epoch in metadata.items()
                                 if key.startswith("slices/") and key.endswith("/epoch") and epoch == 500)
                    grids.append(payload[f"slices/{index}/sample/cpu_reference_mse"].copy())
        positives = np.concatenate([g[np.isfinite(g) & (g > 0)] for g in grids])
        norm = LogNorm(vmin=float(positives.min()) / 1.05, vmax=float(positives.max()) * 1.05)
        titles = ("SGD", "AdamW", "Moonlight Muon", "DA-10 ambient", "DA-10 feasible")
        for axis, grid, title in zip(axes[row_index], grids, titles):
            displayed = np.ma.masked_where(~np.isfinite(grid), np.maximum(grid, norm.vmin))
            image = axis.imshow(displayed.T, extent=(-1, 1, -1, 1), origin="lower", norm=norm,
                                cmap="viridis", interpolation="nearest", aspect="equal")
            axis.scatter([0], [0], marker="o", s=27, facecolors="none", edgecolors="white")
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Direction coordinate 1")
            axis.set_ylabel(f"{task.upper()} / direction 2")
            axis.text(.02, .02, f"Center MSE {grid[15,15]:.3g}\nNonfinite {int((~np.isfinite(grid)).sum())}/961",
                      transform=axis.transAxes, color="white", fontsize=7,
                      bbox={"facecolor":"black", "alpha":.5, "edgecolor":"none"})
        color_axis = figure.add_axes([.945, .56 if row_index == 0 else .235, .012, .27])
        figure.colorbar(image, cax=color_axis, label="Training MSE (log scale)")
    figure.suptitle("Recorded terminal loss surfaces — small models, shared seeds", fontsize=15)
    figure.text(.5, .07, "Lowest completed seed shared by all methods, chosen before inspecting surface values: "
                + "; ".join(selected_seeds) + ".\nEach task uses one common color scale. White ring: retained center. "
                "Nonfinite samples are masked; exact zeros use the positive color-scale floor.\n"
                "Ambient directions/data/preprojection initialization/order are paired. Feasible axes change after tangent projection/polar retraction.\n"
                "Recipes and capacity constraints differ; DA-10 uses CUDA and baseline grids are CPU references. "
                "These examples are 2D slices, not a full-space flatness estimate.",
                ha="center", va="bottom", fontsize=8)
    filename = "da10_four_method_terminal_surfaces.png"
    _finish_figure(figure, artifact, filename)
    plt.close(figure)
    return filename


def plot_report(report: dict[str, Any], artifact: Any) -> list[str]:
    """Render four comparison figures into a W&B artifact and return their names.

    The report is expected to contain ``runs`` with the raw, already audited
    DA-10 run rows described by the landscape audit.  Optional optimizer
    baselines are read from ``baselines``.  This function never writes to the
    local filesystem or communicates with W&B directly. For the completed
    scientific report it reads the already audited W&B-managed raw grid cache.
    """
    if not isinstance(report, dict) or not isinstance(report.get("runs"), list):
        raise ValueError("report must contain a runs list")
    runs = report["runs"]
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("every landscape run report row must be a mapping")

    filenames = [
        _terminal_distribution_figure(report, runs, artifact),
        _temporal_figure(runs, artifact),
        _baseline_comparison_figure(report, runs, artifact),
        _performance_figure(runs, artifact),
    ]
    if report.get("verified_recorded_outcomes") == 480:
        filenames.append(_terminal_surface_figure(report, artifact))
    return filenames

