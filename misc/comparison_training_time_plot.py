"""Publication-quality comparison figures for training-time analysis.

The module automatically reads every experiment in a results directory and
creates three standalone Matplotlib figures:

1. tradeoff_bubble.png
2. accuracy_vs_parameters.png
3. training_time_vs_parameters.png

Only PNG output is produced. The code is modular and reusable so the same
loading, color assignment, label placement, and legend rendering logic is
shared across all figures.
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any, cast

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np

from misc.comparison_metrics_common import format_model_display_name

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _first_numeric(entry: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(entry.get(key))
        if value is not None:
            return value
    return None


def _normalize_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


def _normalize_model_name(raw_name: str) -> str:
    """Return a readable display name without hardcoded architecture names."""
    if not raw_name:
        return "Unknown"
    return format_model_display_name(raw_name)


def _load_metrics_entries(results_dir: Path) -> list[dict[str, Any]]:
    """Load per-model evaluation entries from any available summary file."""
    candidates = [
        results_dir / "overall_report.json",
        results_dir / "metrics_summary.json",
        results_dir / "overall_report_summary.json",
    ]

    for path in candidates:
        data = _load_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            return data["models"]

    return []


def _load_training_history(model_dir: Path) -> dict[str, Any] | None:
    for candidate in (model_dir / "history.json", model_dir / "training_history.json"):
        data = _load_json(candidate)
        if isinstance(data, dict) and data:
            return data
    return None


def _load_training_summary(model_dir: Path) -> dict[str, Any] | None:
    data = _load_json(model_dir / "training_summary.json")
    return data if isinstance(data, dict) and data else None


def _infer_model_key(entry: dict[str, Any], fallback: str) -> str:
    model = entry.get("model") or entry.get("model_key")
    if isinstance(model, str) and model:
        return model

    checkpoint_file = entry.get("checkpoint_file") or entry.get("file") or ""
    if checkpoint_file:
        return Path(str(checkpoint_file)).parent.name or fallback

    return fallback


def load_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load and merge all experiment artifacts under the given results directory."""
    results_path = Path(results_dir)
    metrics_entries = _load_metrics_entries(results_path)
    metrics_map: dict[str, dict[str, Any]] = {}
    for entry in metrics_entries:
        if not isinstance(entry, dict):
            continue
        key = _infer_model_key(entry, fallback="")
        if key and key not in metrics_map:
            metrics_map[key] = entry

    model_dirs = [directory for directory in sorted(results_path.iterdir()) if directory.is_dir() and ((directory / "training_summary.json").exists() or (directory / "history.json").exists() or (directory / "training_history.json").exists())]

    records: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        summary = _load_training_summary(model_dir) or {}
        history = _load_training_history(model_dir)
        model_key = summary.get("model") or model_dir.name
        metrics = metrics_map.get(model_key, {})

        record = {
            "model": model_key,
            "display_name": _normalize_model_name(str(model_key)),
            "params_m": _first_numeric(summary, "num_parameters_millions") or _first_numeric(metrics, "num_parameters_millions"),
            "accuracy": _normalize_percent(_first_numeric(metrics, "test_accuracy", "accuracy") or _first_numeric(summary, "test_accuracy", "accuracy")),
            "precision": _normalize_percent(_first_numeric(metrics, "test_macro_precision", "macro_precision") or _first_numeric(summary, "test_macro_precision", "macro_precision")),
            "recall": _normalize_percent(_first_numeric(metrics, "test_macro_recall", "macro_recall") or _first_numeric(summary, "test_macro_recall", "macro_recall")),
            "macro_f1": _normalize_percent(_first_numeric(metrics, "test_macro_f1", "macro_f1") or _first_numeric(summary, "test_macro_f1", "macro_f1")),
            "training_time_min": _first_numeric(summary, "total_training_time_min") or _first_numeric(metrics, "total_training_time_min"),
            "history": history,
        }

        if record["params_m"] is None or record["accuracy"] is None or record["training_time_min"] is None:
            continue

        records.append(record)

    if not records and metrics_entries:
        # Fallback if the directory contains only summary files and no per-model
        # subdirectories.
        for index, entry in enumerate(metrics_entries):
            if not isinstance(entry, dict):
                continue
            model_key = _infer_model_key(entry, fallback=f"model_{index + 1}")
            record = {
                "model": model_key,
                "display_name": _normalize_model_name(str(model_key)),
                "params_m": _first_numeric(entry, "num_parameters_millions"),
                "accuracy": _normalize_percent(_first_numeric(entry, "test_accuracy", "accuracy")),
                "precision": _normalize_percent(_first_numeric(entry, "test_macro_precision", "macro_precision")),
                "recall": _normalize_percent(_first_numeric(entry, "test_macro_recall", "macro_recall")),
                "macro_f1": _normalize_percent(_first_numeric(entry, "test_macro_f1", "macro_f1")),
                "training_time_min": _first_numeric(entry, "total_training_time_min"),
                "history": entry.get("training_history") if isinstance(entry.get("training_history"), dict) else None,
            }
            if record["params_m"] is not None and record["accuracy"] is not None and record["training_time_min"] is not None:
                records.append(record)

    return sorted(records, key=lambda item: (item["params_m"], item["model"]))


def build_model_dataframe(results_dir: str | Path) -> list[dict[str, Any]]:
    """Public helper that returns the normalized record list.

    A list of dictionaries is used instead of pandas to keep the script light
    and dependency-free.
    """
    return load_results(results_dir)


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------


def assign_model_colors(records: list[dict[str, Any]]) -> dict[str, str]:
    """Generate a stable unique color for each model based on sorted order."""
    unique_models: list[str] = []
    for record in records:
        name = record["model"]
        if name not in unique_models:
            unique_models.append(name)

    n = len(unique_models)
    if n <= 0:
        return {}
    if n <= 10:
        cmap = plt.get_cmap("tab10", n)
    elif n <= 20:
        cmap = plt.get_cmap("tab20", n)
    else:
        cmap = plt.get_cmap("hsv", n)

    return {name: mcolors.to_hex(cmap(index)) for index, name in enumerate(unique_models)}


def _style_axis(ax: Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#e2e2e2", linewidth=0.75)
    ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", labelsize=9.5, length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_color("#b7b7b7")
        spine.set_linewidth(0.8)


def _set_log_x(ax: Axes, x_values: np.ndarray | None = None, *, lower_pad: float = 0.84, upper_pad: float = 1.22, snap_lower_to_decade: bool = False) -> None:
    ax.set_xscale("log")

    if x_values is None or len(x_values) == 0:
        return

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    lower = x_min * lower_pad
    upper = x_max * upper_pad

    if snap_lower_to_decade and lower > 0.0:
        lower = 10.0 ** np.floor(np.log10(lower))

    lower_decade = int(np.floor(np.log10(lower)))
    upper_decade = int(np.ceil(np.log10(upper)))
    major_ticks = np.power(10.0, np.arange(lower_decade, upper_decade + 1, dtype=float))
    major_ticks = major_ticks[(major_ticks >= lower) & (major_ticks <= upper)]
    ax.set_xticks(major_ticks)
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lower, upper)


def _save_figure(fig: Figure, output_path: str | Path, *, dpi: int = 220) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)


def _choose_legend_columns(model_count: int) -> int:
    if model_count <= 3:
        return 1
    if model_count <= 6:
        return 2
    return 3


def _darken_color(color: str, factor: float = 0.78) -> str:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    darkened = np.clip(rgb * factor, 0.0, 1.0)
    red = float(darkened[0])
    green = float(darkened[1])
    blue = float(darkened[2])
    return mcolors.to_hex((red, green, blue))


def _bubble_size_scale(values: np.ndarray, min_size: float = 140.0, max_size: float = 2400.0) -> np.ndarray:
    if values.size == 0:
        return values
    scaled = np.power(values, 0.65)
    lo = float(scaled.min())
    hi = float(scaled.max())
    if np.isclose(lo, hi):
        return np.full_like(values, (min_size + max_size) * 0.5)
    return np.interp(scaled, (lo, hi), (min_size, max_size))


def _single_size(value: float, reference_values: np.ndarray, min_size: float = 140.0, max_size: float = 2400.0) -> float:
    ref_scaled = np.power(reference_values, 0.65)
    scaled = float(np.power(value, 0.65))
    return float(np.interp(scaled, (float(ref_scaled.min()), float(ref_scaled.max())), (min_size, max_size)))


def _fit_regression_line(x_values: np.ndarray, y_values: np.ndarray, degree: int = 2) -> tuple[np.ndarray | None, np.ndarray | None]:
    if x_values.size < 3:
        return None, None
    degree = min(degree, x_values.size - 1)
    coeffs = np.polyfit(np.log10(x_values), y_values, degree)
    x_fit = np.logspace(np.log10(x_values.min()), np.log10(x_values.max()), 240)
    y_fit = np.polyval(coeffs, np.log10(x_fit))
    return x_fit, y_fit


def _fit_regression_curve(x_values: np.ndarray, y_values: np.ndarray, grid_points: int = 240) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit a smooth regression curve in log-log space.

    Training time is strictly positive in the collected results, so a power-law
    fit gives a stable least-squares trend line without the wavy behavior of a
    higher-order polynomial or a local smoother.
    """
    if x_values.size < 2:
        return None, None

    positive_mask = (x_values > 0) & (y_values > 0)
    if positive_mask.sum() < 2:
        return None, None

    x_log = np.log10(x_values[positive_mask])
    y_log = np.log10(y_values[positive_mask])
    slope, intercept = np.polyfit(x_log, y_log, 1)

    x_fit = np.logspace(np.log10(x_values[positive_mask].min()), np.log10(x_values[positive_mask].max()), grid_points)
    y_fit = np.power(10.0, intercept) * np.power(x_fit, slope)
    return x_fit, y_fit


def _compute_pareto_frontier(x_values: np.ndarray, y_values: np.ndarray, *, maximize_y: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    if x_values.size < 2:
        return None, None

    points = sorted(
        ((float(x_value), float(y_value)) for x_value, y_value in zip(x_values, y_values)),
        key=lambda item: (item[0], -item[1] if maximize_y else item[1]),
    )

    frontier: list[tuple[float, float]] = []
    best_y = -np.inf if maximize_y else np.inf
    for x_value, y_value in points:
        if maximize_y:
            if y_value > best_y:
                frontier.append((x_value, y_value))
                best_y = y_value
        else:
            if y_value < best_y:
                frontier.append((x_value, y_value))
                best_y = y_value

    if len(frontier) < 2:
        return None, None

    x_frontier = np.array([point[0] for point in frontier], dtype=float)
    y_frontier = np.array([point[1] for point in frontier], dtype=float)
    return x_frontier, y_frontier


def _add_trend_legend(ax: Axes) -> None:
    handles = [
        Line2D([0], [0], color="#4d4d4d", linestyle="--", linewidth=1.4, label="Regression"),
        Line2D([0], [0], color="#245b28", linestyle="-", linewidth=1.8, label="Pareto-optimal models"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d0d0d0",
        fontsize=8.0,
        handlelength=2.5,
        borderpad=0.6,
        labelspacing=0.4,
    )
    legend.set_zorder(20)


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------


def _adjust_labels(
    ax: Axes,
    texts: list[Text],
    point_positions: list[tuple[float, float]],
    min_display_offset: float = 8.0,
    start_positions: np.ndarray | None = None,
    keep_above_points: bool = True,
) -> None:
    """Equivalent to adjustText for this small-number-of-labels use case."""
    if not texts:
        return

    fig = ax.figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(cast(Figure, fig))
    canvas.draw()
    renderer = canvas.get_renderer()
    point_disp = np.array([ax.transData.transform(position) for position in point_positions], dtype=float)
    if start_positions is None:
        positions = point_disp + np.array([0.0, min_display_offset], dtype=float)
    else:
        positions = np.asarray(start_positions, dtype=float).copy()
    axes_bbox = ax.get_window_extent(renderer)

    for _ in range(160):
        for text, position in zip(texts, positions):
            new_xy = ax.transData.inverted().transform(position)
            text.set_position((float(new_xy[0]), float(new_xy[1])))

        canvas.draw()
        renderer = canvas.get_renderer()
        boxes = [text.get_window_extent(renderer).expanded(1.03, 1.08) for text in texts]
        moves = np.zeros_like(positions)

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue

                center_i = np.array([boxes[i].x0 + boxes[i].width / 2.0, boxes[i].y0 + boxes[i].height / 2.0])
                center_j = np.array([boxes[j].x0 + boxes[j].width / 2.0, boxes[j].y0 + boxes[j].height / 2.0])
                direction = center_i - center_j
                distance = float(np.linalg.norm(direction))
                if np.isclose(distance, 0.0):
                    direction = np.array([1.0, 1.0])
                    distance = float(np.linalg.norm(direction))
                step = direction / distance * 2.2
                moves[i] += step
                moves[j] -= step

        for index, bbox in enumerate(boxes):
            point = point_disp[index]
            if bbox.contains(*point) or bbox.y0 < point[1] + min_display_offset * 0.5:
                moves[index][1] += 1.5

        positions += moves
        positions[:, 0] = np.clip(positions[:, 0], axes_bbox.x0 + 8.0, axes_bbox.x1 - 8.0)
        positions[:, 1] = np.clip(positions[:, 1], axes_bbox.y0 + 8.0, axes_bbox.y1 - 8.0)
        if keep_above_points:
            positions[:, 1] = np.maximum(positions[:, 1], point_disp[:, 1] + min_display_offset)

        if np.max(np.linalg.norm(moves, axis=1)) < 0.35:
            break

    for text, position in zip(texts, positions):
        new_xy = ax.transData.inverted().transform(position)
        text.set_position((float(new_xy[0]), float(new_xy[1])))


def _place_model_labels(ax: Axes, records: list[dict[str, Any]], x_key: str, y_key: str, *, fontsize: float = 8.0) -> None:
    texts: list[Text] = []
    points: list[tuple[float, float]] = []

    for record in records:
        x_value = float(record[x_key])
        y_value = float(record[y_key])
        points.append((x_value, y_value))
        text = ax.text(
            x_value,
            y_value,
            record["display_name"],
            fontsize=fontsize,
            ha="center",
            va="bottom",
            color="#111111",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )
        texts.append(text)

    _adjust_labels(ax, texts, points)


def _place_tradeoff_labels(ax: Axes, records: list[dict[str, Any]], color_map: dict[str, str], *, fontsize: float = 8.0) -> None:
    texts: list[Text] = []
    points: list[tuple[float, float]] = []
    side_offsets = [
        (24.0, 12.0),  # mv
        (18.0, 8.0),
        (18.0, -10.0),
        (24.0, -12.0),  # cait
        (-70.0, 16.0),  # deit
        (18.0, 4.0),
        (36.0, 0.0),
        (22.0, 10.0),
        (28.0, -16.0),
    ]

    for index, record in enumerate(records):
        x_value = float(record["params_m"])
        y_value = float(record["accuracy"])
        points.append((x_value, y_value))
        label_name = str(record["display_name"]).lower()
        if "deit small" in label_name:
            dx, dy = (-26.0, 12.0)
        elif "vit small" in label_name:
            dx, dy = (26.0, -12.0)
        else:
            dx, dy = side_offsets[index % len(side_offsets)]
        label_color = _darken_color(color_map[record["model"]])
        text = ax.text(
            x_value,
            y_value,
            record["display_name"],
            fontsize=fontsize,
            ha="left",
            va="center",
            color=label_color,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )
        texts.append(text)

    fig = ax.figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(cast(Figure, fig))
    canvas.draw()
    start_offsets = []
    for index, record in enumerate(records):
        label_name = str(record["display_name"]).lower()
        if "deit small" in label_name:
            start_offsets.append((-26.0, 12.0))
        elif "vit small" in label_name:
            start_offsets.append((26.0, -12.0))
        else:
            start_offsets.append(side_offsets[index % len(side_offsets)])
    start_positions = np.array([ax.transData.transform(point) + np.array(offset) for point, offset in zip(points, start_offsets)], dtype=float)
    _adjust_labels(ax, texts, points, min_display_offset=6.0, start_positions=start_positions, keep_above_points=False)


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------


def draw_bubble_legend(ax: Axes, reference_values: np.ndarray) -> None:
    """Bubble-size reference legend used inside Figure 1."""
    handles = []
    labels = ["50 min", "150 min", "300+ min"]
    values = [50.0, 150.0, 300.0]

    for value in values:
        size = _single_size(value, reference_values)
        marker_size = max(8.0, min(20.0, np.sqrt(size) * 0.35))
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="#444444",
                markeredgewidth=0.8,
                markersize=marker_size,
            )
        )

    legend = ax.legend(
        handles,
        labels,
        title="     Bubble Size\nTraining Time (min)",
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=1,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#d0d0d0",
        fontsize=8.0,
        title_fontsize=8.5,
        borderpad=1.55,
        labelspacing=1.25,
        handletextpad=0.6,
        columnspacing=2.15,
    )
    legend.get_title().set_fontweight("semibold")


def draw_model_legend(ax: Axes, records: list[dict[str, Any]], color_map: dict[str, str]) -> None:
    """Compact legend-style information panel with no borders or cells."""
    ax.set_axis_off()
    sorted_records = sorted(records, key=lambda record: (-float(record["accuracy"]), float(record["params_m"]), str(record["model"])))
    ax.text(
        0.0,
        1.0,
        "Model Information",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#111111",
    )

    n_models = len(sorted_records)
    columns = _choose_legend_columns(n_models)
    rows = ceil(n_models / columns)

    x_origin = 0.02
    available_width = 0.96
    column_width = available_width / columns
    y_top = 0.86
    y_step = 0.1 if rows <= 3 else 0.05

    for index, record in enumerate(sorted_records):
        column = index // rows
        row = index % rows
        x = x_origin + column * column_width
        y = y_top - row * y_step

        ax.scatter(
            [x],
            [y + 0.01],
            s=60,
            color=color_map[record["model"]],
            edgecolors="#333333",
            linewidths=0.6,
            transform=ax.transAxes,
            clip_on=False,
            zorder=4,
        )
        ax.text(
            x + 0.028,
            y + 0.014,
            record["display_name"],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="semibold",
            color="#1b1b1b",
        )

        ax.text(
            x + 0.115,
            y + 0.014,
            (f"{record['accuracy']:.2f}%  " f"{record['params_m']:.1f}M  " f"{record['training_time_min']:.0f} min"),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.7,
            color="#3a3a3a",
        )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_common_scatter(ax: Axes, records: list[dict[str, Any]], color_map: dict[str, str], y_key: str, y_label: str, *, title: str, y_margin: float = 0.08) -> None:
    x_values = np.array([record["params_m"] for record in records], dtype=float)
    y_values = np.array([record[y_key] for record in records], dtype=float)

    for record in records:
        ax.scatter(
            record["params_m"],
            record[y_key],
            s=80,
            color=color_map[record["model"]],
            edgecolors="#222222",
            linewidths=0.6,
            alpha=0.92,
            zorder=4,
        )

    _set_log_x(ax, x_values, snap_lower_to_decade=True)
    _style_axis(ax)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Parameter Count (Millions)", fontsize=10.4, labelpad=5)
    ax.set_ylabel(y_label, fontsize=10.4, labelpad=5)

    y_margin_abs = max(0.8, (y_values.max() - y_values.min()) * y_margin)
    ax.set_ylim(y_values.min() - y_margin_abs, y_values.max() + y_margin_abs)

    _place_model_labels(ax, records, "params_m", y_key, fontsize=7.8)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def plot_tradeoff(records: list[dict[str, Any]], color_map: dict[str, str], output_path: str | Path) -> None:
    """Figure 1: bubble trade-off plot."""
    fig = plt.figure(figsize=(11.6, 11.9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4.9, 1.25], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])

    x_values = np.array([record["params_m"] for record in records], dtype=float)
    y_values = np.array([record["accuracy"] for record in records], dtype=float)
    time_values = np.array([record["training_time_min"] for record in records], dtype=float)
    bubble_sizes = _bubble_size_scale(time_values)

    _set_log_x(ax, x_values, snap_lower_to_decade=True)
    _style_axis(ax)
    ax.set_title("Model Size, Accuracy, and Training Cost", fontsize=11.8, fontweight="bold", pad=9)
    ax.set_xlabel("Parameter Count (Millions)", fontsize=10.6, labelpad=5)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10.6, labelpad=5)
    y_margin = max(3.0, (y_values.max() - y_values.min()) * 0.3)
    ax.set_ylim(y_values.min() - y_margin, y_values.max() + y_margin)

    bubble_colors = [color_map[record["model"]] for record in records]
    ax.scatter(
        x_values,
        y_values,
        s=bubble_sizes,
        c=bubble_colors,
        alpha=0.80,
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
    )

    best_index = int(np.argmax(y_values))
    best_record = records[best_index]
    ax.axhline(y_values[best_index], color="#c62828", linestyle="--", linewidth=1.1, alpha=0.9, zorder=2)
    ax.annotate(
        "Best Overall Trade-off",
        xy=(best_record["params_m"], best_record["accuracy"]),
        xytext=(52, 40),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#111111",
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.0,
            shrinkA=0,
            shrinkB=14,
            connectionstyle="arc3,rad=0.28",
        ),
        zorder=6,
    )
    ax.text(
        x_values.min() * 0.78,
        y_values[best_index] - 0.25,
        f"Best Accuracy : {best_record['accuracy']:.2f}%",
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color="#c62828",
        zorder=6,
    )

    _place_tradeoff_labels(ax, records, color_map, fontsize=8.0)
    draw_bubble_legend(ax, time_values)

    legend_ax.set_axis_off()
    draw_model_legend(legend_ax, records, color_map)

    _save_figure(fig, output_path, dpi=240)
    plt.close(fig)


def plot_accuracy_vs_parameters(records: list[dict[str, Any]], color_map: dict[str, str], output_path: str | Path) -> None:
    """Figure 2: accuracy vs. parameter count."""
    fig = plt.figure(figsize=(10.0, 8.5), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4.7, 1.65], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    # legend_ax = fig.add_subplot(gs[1])

    _plot_common_scatter(
        ax,
        records,
        color_map,
        y_key="accuracy",
        y_label="Accuracy (%)",
        title="Accuracy vs. Parameter Count",
        y_margin=0.09,
    )

    x_values = np.array([record["params_m"] for record in records], dtype=float)
    y_values = np.array([record["accuracy"] for record in records], dtype=float)
    x_fit, y_fit = _fit_regression_line(x_values, y_values, degree=1)
    if x_fit is not None and y_fit is not None:
        ax.plot(x_fit, y_fit, linestyle="--", color="#4d4d4d", linewidth=1.25, zorder=2)

    x_frontier, y_frontier = _compute_pareto_frontier(x_values, y_values, maximize_y=True)
    if x_frontier is not None and y_frontier is not None:
        ax.plot(x_frontier, y_frontier, linestyle="-", color="#245b28", linewidth=1.5, zorder=2.5)

    _add_trend_legend(ax)

    # legend_ax.set_axis_off()
    # draw_model_legend(legend_ax, records, color_map)

    _save_figure(fig, output_path, dpi=220)
    plt.close(fig)


def plot_training_time_vs_parameters(records: list[dict[str, Any]], color_map: dict[str, str], output_path: str | Path) -> None:
    """Figure 3: training time vs. parameter count."""
    fig = plt.figure(figsize=(10.0, 8.5), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4.7, 1.65], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    # legend_ax = fig.add_subplot(gs[1])

    _plot_common_scatter(
        ax,
        records,
        color_map,
        y_key="training_time_min",
        y_label="Training Time (minutes)",
        title="Training Time vs. Parameter Count",
        y_margin=0.11,
    )

    x_values = np.array([record["params_m"] for record in records], dtype=float)
    y_values = np.array([record["training_time_min"] for record in records], dtype=float)
    x_fit, y_fit = _fit_regression_curve(x_values, y_values)
    if x_fit is not None and y_fit is not None:
        ax.plot(x_fit, y_fit, linestyle="--", color="#4d4d4d", linewidth=1.25, zorder=2)

    # legend_ax.set_axis_off()
    # draw_model_legend(legend_ax, records, color_map)

    _save_figure(fig, output_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public orchestration helpers
# ---------------------------------------------------------------------------


def save_figure(fig: Figure, output_path: str | Path) -> None:
    _save_figure(fig, output_path)


def generate_publication_figures(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load results and generate all three publication figures."""
    results_path = Path(results_dir)
    records = build_model_dataframe(results_path)
    if not records:
        return []

    color_map = assign_model_colors(records)
    plot_tradeoff(records, color_map, results_path / "tradeoff_bubble.png")
    plot_accuracy_vs_parameters(records, color_map, results_path / "accuracy_vs_parameters.png")
    plot_training_time_vs_parameters(records, color_map, results_path / "training_time_vs_parameters.png")
    return records


def plot_training_time_comparison(models_data, out_path, title: str | None = None):
    """Backward-compatible wrapper.

    If the first argument is a results directory, the three publication figures
    are generated automatically. If a list of model dictionaries is supplied,
    the function renders the trade-off figure to the requested path.
    """
    if isinstance(models_data, (str, Path)) and out_path is None:
        generate_publication_figures(models_data)
        return

    if isinstance(models_data, list):
        records = []
        for index, entry in enumerate(models_data):
            if not isinstance(entry, dict):
                continue
            model_key = str(entry.get("model") or entry.get("model_key") or f"model_{index + 1}")
            params = _first_numeric(entry, "num_parameters_millions")
            accuracy = _normalize_percent(_first_numeric(entry, "test_accuracy", "accuracy"))
            training_time = _first_numeric(entry, "total_training_time_min", "training_time_min")
            if params is None or accuracy is None or training_time is None:
                continue
            records.append(
                {
                    "model": model_key,
                    "display_name": _normalize_model_name(model_key),
                    "params_m": params,
                    "accuracy": accuracy,
                    "training_time_min": training_time,
                }
            )

        if records and out_path is not None:
            color_map = assign_model_colors(records)
            plot_tradeoff(records, color_map, out_path)


def main(results_dir: str | Path = "results") -> None:
    generate_publication_figures(results_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "assign_model_colors",
    "build_model_dataframe",
    "draw_bubble_legend",
    "draw_model_legend",
    "generate_publication_figures",
    "load_results",
    "main",
    "plot_accuracy_vs_parameters",
    "plot_tradeoff",
    "plot_training_time_comparison",
    "plot_training_time_vs_parameters",
    "save_figure",
]
