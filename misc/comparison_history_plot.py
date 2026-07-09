"""Combined loss/accuracy history plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from misc.comparison_metrics_common import format_model_display_name, trim_series


def plot_metric_overview_on_axis(ax, models_data, metric_name, ylabel, title):
    color_handles = []
    for item in models_data:
        history = item["history"]
        color = item["color"]
        model_name = item.get("display_label") or format_model_display_name(item["label"])
        train_values = history.get(f"train_{metric_name}", [])
        val_values = history.get(f"val_{metric_name}", [])
        series = trim_series(train_values, val_values)
        if not series:
            continue
        train_values = list(series[0] or [])
        val_values = list(series[1] or [])
        if not train_values or not val_values:
            continue

        epochs = np.arange(1, len(train_values) + 1)
        ax.plot(epochs, train_values, color=color, linestyle="-", linewidth=2, alpha=0.9)
        ax.plot(epochs, val_values, color=color, linestyle="--", linewidth=2, alpha=0.9)
        color_handles.append(Line2D([0], [0], color=color, lw=3, label=model_name))

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    style_handle_train = Line2D([0], [0], color="black", lw=2, linestyle="-", label="train")
    style_handle_val = Line2D([0], [0], color="black", lw=2, linestyle="--", label="val")
    return color_handles + [style_handle_train, style_handle_val]


def plot_history_loss_accuracy_comparison(models_data, out_path):
    if not models_data:
        return

    fig_width = max(16, len(models_data) * 1.25)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 7))

    legend_handles = plot_metric_overview_on_axis(
        axes[0],
        models_data,
        metric_name="loss",
        ylabel="Loss",
        title="Training and validation loss by model",
    )
    plot_metric_overview_on_axis(
        axes[1],
        models_data,
        metric_name="acc",
        ylabel="Accuracy",
        title="Training and validation accuracy by model",
    )

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_history_loss_accuracy_comparison"]
