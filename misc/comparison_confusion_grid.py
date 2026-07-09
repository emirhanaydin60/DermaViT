"""Per-model confusion matrix grid."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from misc.comparison_metrics_common import format_model_display_name


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


def plot_confusion_matrix_grid(models_data, out_path, class_names=None, grid_cols=3, grid_rows=None):
    count = len(models_data)
    if count == 0:
        return

    rows, cols = _resolve_grid_shape(count, grid_cols=grid_cols, grid_rows=grid_rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.8))
    axes = np.array(axes).reshape(-1)

    matrices = [np.asarray(item.get("confusion_matrix")) for item in models_data if item.get("confusion_matrix") is not None]
    matrices = [cm for cm in matrices if cm.size > 0]
    if not matrices:
        plt.close(fig)
        return

    vmax = max(float(np.max(cm)) for cm in matrices)
    if vmax <= 0:
        vmax = 1.0

    cmap = plt.get_cmap("Blues")
    for idx, item in enumerate(models_data):
        ax = axes[idx]
        cm = item.get("confusion_matrix")
        if cm is None:
            ax.set_axis_off()
            continue

        cm = np.asarray(cm)
        if cm.size == 0:
            ax.set_axis_off()
            continue

        item_class_names = item.get("class_names") or class_names
        if not item_class_names or len(item_class_names) != cm.shape[0]:
            item_class_names = [str(i) for i in range(cm.shape[0])]

        ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                value = cm[row_idx, col_idx]
                text_color = "white" if value > (vmax * 0.55) else "black"
                ax.text(col_idx, row_idx, f"{int(value)}", ha="center", va="center", fontsize=6.5, color=text_color)

        display_label = item.get("display_label") or format_model_display_name(item["label"])
        ax.set_title(f"({_panel_prefix(idx)[1:-1]}) {display_label}", fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(len(item_class_names)))
        ax.set_yticks(np.arange(len(item_class_names)))
        ax.set_xticklabels(item_class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(item_class_names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)

        caption_parts = []
        if item.get("accuracy") is not None:
            caption_parts.append(f"acc={item['accuracy']:.3f}")
        if item.get("macro_f1") is not None:
            caption_parts.append(f"mf1={item['macro_f1']:.3f}")
        if caption_parts:
            ax.text(0.02, 0.02, " | ".join(caption_parts), transform=ax.transAxes, fontsize=8, va="bottom")

    for idx in range(count, len(axes)):
        axes[idx].set_axis_off()

    fig.suptitle("Test confusion matrices by model", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_confusion_matrix_grid"]
