"""Plot comparison metrics from `results/metrics_summary.json`.

Generates `results/comparison_metrics.png` with two stacked bar plots:
- Top: accuracy for all models
- Bottom: macro F1 for all models

Each model gets a unique color (1st green, 2nd red, ...). A legend
maps model index -> model name.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools


def main():
    metrics_path = Path("results") / "metrics_summary.json"
    if not metrics_path.exists():
        print(f"metrics file not found at {metrics_path}. Run evaluations first.")
        return

    with open(metrics_path, "r") as f:
        results = json.load(f)

    if not results:
        print("No results found in metrics_summary.json")
        return

    labels = [r.get("model_key") or Path(r.get("file", "")).parent.name or Path(r.get("file", "")).stem for r in results]
    accuracies = [r.get("accuracy", 0.0) for r in results]
    f1s = [r.get("macro_f1", 0.0) for r in results]

    base_colors = ["green", "red", "blue", "orange", "purple", "brown", "cyan", "magenta", "olive", "gray"]
    if len(labels) > len(base_colors):
        colors = list(itertools.islice(itertools.cycle(base_colors), len(labels)))
    else:
        colors = base_colors[: len(labels)]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 1.2), 8), sharex=True)

    def annotate_bars(ax, bars, fmt="{:.3f}", fontsize=9, highlights=None):
        if highlights is None:
            highlights = set()
        ymax = ax.get_ylim()[1]
        for i, bar in enumerate(bars):
            height = bar.get_height()
            label = fmt.format(height)
            fontweight = "bold" if i in highlights else "normal"
            if i in highlights:
                label = label + " *"
            x_pos = bar.get_x() + bar.get_width() / 2
            # place label inside bar if tall, otherwise above
            if height >= 0.95 * ymax:
                y = height - 0.01 * ymax
                va = "top"
                color = "white"
            else:
                y = height + 0.01 * ymax
                va = "bottom"
                color = "black"
            ax.text(x_pos, y, label, ha="center", va=va, fontsize=fontsize, color=color, fontweight=fontweight)

    idx_best_acc = int(np.argmax(accuracies))
    idx_best_f1 = int(np.argmax(f1s))

    bars_acc = axes[0].bar(x, accuracies, color=colors)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy per model")
    annotate_bars(axes[0], bars_acc, fmt="{:.3f}", highlights={idx_best_acc})

    bars_f1 = axes[1].bar(x, f1s, color=colors)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xticks(x)
    axes[1].set_title("Macro-F1 per model")
    # hide x-axis labels (we use legend for mapping)
    axes[1].tick_params(axis="x", labelbottom=False)
    annotate_bars(axes[1], bars_f1, fmt="{:.3f}", highlights={idx_best_f1})

    # legend mapping index -> model name (colored)
    patches = [mpatches.Patch(color=colors[i], label=f"{i+1}. {labels[i]}") for i in range(len(labels))]
    axes[0].legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    # embed best values in a small box on the FIGURE (bottom-right)
    best_acc = accuracies[idx_best_acc]
    best_f1 = f1s[idx_best_f1]

    # inset axes placed relative to the whole figure (bottom-right)
    box_w = 0.26
    box_h = 0.10
    box_x = 0.98 - box_w - 0.01
    box_y = 0.01
    box_ax = fig.add_axes([box_x, box_y, box_w, box_h], frameon=False)
    box_ax.set_xticks([])
    box_ax.set_yticks([])

    # positions inside inset axes (in axes coords)
    y1 = 0.67
    y2 = 0.30
    txt1 = f"Best Acc: {labels[idx_best_acc]} = {best_acc:.3f}"
    txt2 = f"Best F1:  {labels[idx_best_f1]} = {best_f1:.3f}"
    box_ax.text(0.02, y1, txt1, ha='left', va='center', fontsize=9)
    box_ax.text(0.02, y2, txt2, ha='left', va='center', fontsize=9)

    # draw colored squares at the end of each line inside the box
    sq_w = 0.07
    sq_h = 0.25
    # right-aligned squares
    rect1 = mpatches.Rectangle((0.90, y1 - sq_h / 2), sq_w, sq_h, transform=box_ax.transAxes, facecolor=colors[idx_best_acc], edgecolor='none')
    rect2 = mpatches.Rectangle((0.90, y2 - sq_h / 2), sq_w, sq_h, transform=box_ax.transAxes, facecolor=colors[idx_best_f1], edgecolor='none')
    box_ax.add_patch(rect1)
    box_ax.add_patch(rect2)

    plt.tight_layout()
    out_path = metrics_path.parent / "comparison_metrics.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
