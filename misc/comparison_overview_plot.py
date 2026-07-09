"""Overview comparison plot."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from misc.comparison_metrics_common import format_model_display_name


def plot_metrics_comparison(processed, out_path, results_root=Path("results")):
    comparison_items = sorted(processed, key=lambda item: item["macro_f1"], reverse=True)
    labels = [r["label"] for r in comparison_items]
    accuracies = [r["accuracy"] for r in comparison_items]
    f1s = [r["macro_f1"] for r in comparison_items]

    base_colors = ["green", "red", "blue", "orange", "purple", "brown", "cyan", "magenta", "olive", "gray"]
    if len(labels) > len(base_colors):
        bar_colors = list(itertools.islice(itertools.cycle(base_colors), len(labels)))
    else:
        bar_colors = base_colors[: len(labels)]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 1.0), 8), sharex=True)

    def annotate_bars(ax, bars, fmt="{:.3f}", fontsize=9, highlights=None):
        if highlights is None:
            highlights = set()
        ymax = ax.get_ylim()[1]
        for i, bar in enumerate(bars):
            height = bar.get_height()
            label = fmt.format(height)
            if i in highlights:
                label += " *"
            x_pos = bar.get_x() + bar.get_width() / 2
            if height >= 0.95 * ymax:
                y = height - 0.01 * ymax
                va = "top"
                color = "white"
            else:
                y = height + 0.01 * ymax
                va = "bottom"
                color = "black"
            ax.text(x_pos, y, label, ha="center", va=va, fontsize=fontsize, color=color, fontweight="bold" if i in highlights else "normal")

    idx_best_acc = int(np.argmax(accuracies))
    idx_best_f1 = int(np.argmax(f1s))
    bar_width = 0.8

    bars_f1 = axes[0].bar(x, f1s, color=bar_colors, width=bar_width)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Macro-F1 per model")
    axes[0].tick_params(axis="x", labelbottom=False)
    annotate_bars(axes[0], bars_f1, fmt="{:.3f}", highlights={idx_best_f1})

    bars_acc = axes[1].bar(x, accuracies, color=bar_colors, width=bar_width)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(x)
    axes[1].set_title("Accuracy per model")
    annotate_bars(axes[1], bars_acc, fmt="{:.3f}", highlights={idx_best_acc})

    patches = [mpatches.Patch(color=bar_colors[i], label=f"#{i + 1} {comparison_items[i].get('display_label', format_model_display_name(labels[i]))}") for i in range(len(labels))]
    axes[0].legend(handles=patches, title="Model order\n(Macro F1 desc)", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout(rect=(0, 0.10, 0.76, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_metrics_comparison"]
