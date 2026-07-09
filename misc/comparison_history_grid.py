"""Per-model history grid."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from misc.comparison_metrics_common import adjust_color, format_model_display_name, trim_series


def _panel_prefix(index):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if index < len(alphabet):
        return f"({alphabet[index]})"
    return f"({index + 1})"


def _resolve_grid_shape(count, grid_cols=3, grid_rows=None):
    cols = max(1, int(grid_cols) if grid_cols is not None else 1)
    rows = max(1, int(grid_rows) if grid_rows is not None else math.ceil(count / cols))
    if rows * cols < count:
        rows = math.ceil(count / cols)
    return rows, cols


def get_best_history_point(history, metric_name):
    values = trim_series(history.get(f"val_{metric_name}", []))
    if not values:
        return None, None
    values = values[0]
    if not values:
        return None, None
    best_idx = int(np.argmin(values) if metric_name == "loss" else np.argmax(values))
    return best_idx, values[best_idx]


def plot_model_grid(models_data, out_path, best_metric_name="loss", grid_cols=3, grid_rows=None):
    count = len(models_data)
    if count == 0:
        return

    rows, cols = _resolve_grid_shape(count, grid_cols=grid_cols, grid_rows=grid_rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.2))
    axes = np.array(axes).reshape(-1)

    for idx, item in enumerate(models_data):
        ax = axes[idx]
        history = item["history"]
        color = item["color"]
        display_label = item.get("display_label") or format_model_display_name(item["label"])
        loss_color = color
        acc_color = adjust_color(color, 0.45)

        loss_series = trim_series(history.get("train_loss", []), history.get("val_loss", []))
        acc_series = trim_series(history.get("train_acc", []), history.get("val_acc", []))
        if not loss_series or not acc_series:
            ax.set_axis_off()
            continue

        train_loss = list(loss_series[0] or [])
        val_loss = list(loss_series[1] or [])
        train_acc = list(acc_series[0] or [])
        val_acc = list(acc_series[1] or [])

        epochs_loss = np.arange(1, len(train_loss) + 1)
        epochs_acc = np.arange(1, len(train_acc) + 1)

        ax2 = ax.twinx()
        ax.plot(epochs_loss, train_loss, color=loss_color, linestyle="-", linewidth=2, alpha=0.9, label="train loss")
        ax.plot(epochs_loss, val_loss, color=loss_color, linestyle="--", linewidth=2, alpha=0.9, label="val loss")
        ax2.plot(epochs_acc, train_acc, color=acc_color, linestyle=":", linewidth=2, alpha=0.9, label="train acc")
        ax2.plot(epochs_acc, val_acc, color=acc_color, linestyle="-.", linewidth=2, alpha=0.9, label="val acc")

        best_history_idx, best_history_value = get_best_history_point(history, best_metric_name)
        if best_history_idx is not None:
            best_epoch = best_history_idx + 1
            best_color = acc_color if best_metric_name == "acc" else loss_color
            ax.scatter([best_epoch], [best_history_value], s=120, color=best_color, edgecolors="white", linewidths=1.5, zorder=6)
            ax.annotate(
                f"best val {best_metric_name}={best_history_value:.3f}\nepoch={best_epoch}",
                xy=(best_epoch, best_history_value),
                xytext=(8, -18),
                textcoords="offset points",
                fontsize=8,
                color=best_color,
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=best_color, alpha=0.85),
            )

        ax.set_title(f"({_panel_prefix(idx)[1:-1]}) {display_label}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color=loss_color)
        ax2.set_ylabel("Acc", color=acc_color)
        ax.tick_params(axis="y", labelcolor=loss_color)
        ax2.tick_params(axis="y", labelcolor=acc_color)
        ax.grid(True, alpha=0.2)

        # best_metric_caption = f"best val {best_metric_name}={best_history_value:.3f}" if best_history_idx is not None else None
        # caption_parts = []
        # if item.get("epochs_trained") is not None:
        #     caption_parts.append(f"ep={item['epochs_trained']}")
        # if best_metric_caption is not None:
        #     caption_parts.append(best_metric_caption)
        # elif item.get("best_val_loss") is not None:
        #     caption_parts.append(f"best val loss={item['best_val_loss']:.3f}")
        # if caption_parts:
        #     ax.text(0.02, 0.02, " | ".join(caption_parts), transform=ax.transAxes, fontsize=8, va="bottom")

    for idx in range(count, len(axes)):
        axes[idx].set_axis_off()

    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="train loss"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="val loss"),
        Line2D([0], [0], color="black", lw=2, linestyle=":", label="train acc"),
        Line2D([0], [0], color="black", lw=2, linestyle="-.", label="val acc"),
    ]
    fig.legend(handles=style_handles, loc="upper center", ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_model_grid"]
