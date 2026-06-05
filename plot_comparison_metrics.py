"""Plot comparison metrics and training histories from the results folder.

Outputs:
- `results/comparison_metrics.png`: accuracy and macro F1 bar charts.
- `results/comparison_history_loss.png`: all models' train/val loss curves on one plot.
- `results/comparison_history_accuracy.png`: all models' train/val accuracy curves on one plot.
- `results/comparison_history_grid.png`: one image with per-model history panels.

The script prefers `results/overall_report.json` because it can carry both
test metrics and full training history. It falls back to older summary files
or per-model `history.json` files when needed.
"""

import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import itertools


def load_json_if_exists(path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None


def load_model_history(model_name, results_root=Path("results")):
    candidates = [
        results_root / model_name / "history.json",
        results_root / model_name / "training_history.json",
    ]
    for candidate in candidates:
        data = load_json_if_exists(candidate)
        if data:
            return data
    return None


def infer_model_name(entry):
    model_name = entry.get("model") or entry.get("model_key")
    if model_name:
        return model_name

    file_field = entry.get("file") or entry.get("checkpoint_file") or ""
    if file_field:
        checkpoint_path = Path(file_field)
        if checkpoint_path.parent.name:
            return checkpoint_path.parent.name
        if checkpoint_path.stem:
            return checkpoint_path.stem

    return "unknown"


def collect_entries(results_root=Path("results")):
    overall_report_path = results_root / "overall_report.json"
    metrics_summary_path = results_root / "metrics_summary.json"

    results = load_json_if_exists(overall_report_path)
    if results:
        return results

    results = load_json_if_exists(metrics_summary_path)
    if results:
        return results

    fallback_results = []
    if results_root.exists():
        for history_path in sorted(results_root.glob("*/history.json")):
            model_name = history_path.parent.name
            history = load_json_if_exists(history_path)
            if not history:
                continue
            summary = load_json_if_exists(history_path.parent / "training_summary.json") or {}
            fallback_results.append(
                {
                    "model": model_name,
                    "model_key": model_name,
                    "training_history": history,
                    "requested_epochs": summary.get("requested_epochs"),
                    "epochs_trained": summary.get("epochs_trained"),
                    "total_training_time_min": summary.get("total_training_time_min"),
                    "best_val_loss": summary.get("best_val_loss"),
                    "best_val_f1": summary.get("best_val_f1"),
                }
            )
    return fallback_results


def extract_history(entry):
    history = entry.get("training_history") or entry.get("history")
    if history:
        return history

    model_name = infer_model_name(entry)
    return load_model_history(model_name)


def get_colors(count):
    if count <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i % 10) for i in range(count)]

    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(count)]


def trim_series(*series):
    lengths = [len(s) for s in series if s is not None]
    if not lengths:
        return []
    n = min(lengths)
    return [list(s[:n]) if s is not None else None for s in series]


def shorten_label(label, max_len=15):
    if len(label) <= max_len:
        return label
    return label[:max_len] + "..."


def adjust_color(color, factor=0.78):
    r, g, b = mcolors.to_rgb(color)
    return (
        max(0.0, min(1.0, r * factor)),
        max(0.0, min(1.0, g * factor)),
        max(0.0, min(1.0, b * factor)),
    )


def plot_metric_overview(models_data, metric_name, out_path, ylabel, title):
    fig, ax = plt.subplots(figsize=(max(12, len(models_data) * 1.3), 7))

    color_handles = []
    for item in models_data:
        history = item["history"]
        color = item["color"]
        model_name = item["label"]
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"
        train_values = history.get(train_key, [])
        val_values = history.get(val_key, [])
        series = trim_series(train_values, val_values)
        if not series:
            continue
        train_values, val_values = series
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
    handles = color_handles + [style_handle_train, style_handle_val]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_grid(models_data, out_path):
    count = len(models_data)
    if count == 0:
        return

    cols = 3
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.2))
    axes = np.array(axes).reshape(-1)

    for idx, item in enumerate(models_data):
        ax = axes[idx]
        history = item["history"]
        color = item["color"]
        label = item["label"]
        loss_color = color
        acc_color = adjust_color(color, 0.45)

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_acc = history.get("train_acc", [])
        val_acc = history.get("val_acc", [])

        loss_series = trim_series(train_loss, val_loss)
        acc_series = trim_series(train_acc, val_acc)
        if not loss_series or not acc_series:
            ax.set_axis_off()
            continue

        train_loss, val_loss = loss_series
        train_acc, val_acc = acc_series

        epochs_loss = np.arange(1, len(train_loss) + 1)
        epochs_acc = np.arange(1, len(train_acc) + 1)

        ax2 = ax.twinx()
        ax.plot(epochs_loss, train_loss, color=loss_color, linestyle="-", linewidth=2, alpha=0.9, label="train loss")
        ax.plot(epochs_loss, val_loss, color=loss_color, linestyle="--", linewidth=2, alpha=0.9, label="val loss")
        ax2.plot(epochs_acc, train_acc, color=acc_color, linestyle=":", linewidth=2, alpha=0.9, label="train acc")
        ax2.plot(epochs_acc, val_acc, color=acc_color, linestyle="-.", linewidth=2, alpha=0.9, label="val acc")

        best_loss_idx = int(np.argmin(val_loss))
        best_epoch = best_loss_idx + 1
        best_loss_value = val_loss[best_loss_idx]
        ax.scatter([best_epoch], [best_loss_value], s=120, color=loss_color, edgecolors="white", linewidths=1.5, zorder=6)
        ax.annotate(
            f"best loss={best_loss_value:.3f}\nepoch={best_epoch}",
            xy=(best_epoch, best_loss_value),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=8,
            color=loss_color,
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=loss_color, alpha=0.85),
        )

        ax.set_title(shorten_label(label, max_len=18), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color=loss_color)
        ax2.set_ylabel("Acc", color=acc_color)
        ax.tick_params(axis="y", labelcolor=loss_color)
        ax2.tick_params(axis="y", labelcolor=acc_color)
        ax.grid(True, alpha=0.2)

        best_loss = item.get("best_val_loss")
        epochs_trained = item.get("epochs_trained")
        caption_parts = []
        if epochs_trained is not None:
            caption_parts.append(f"ep={epochs_trained}")
        if best_loss is not None:
            caption_parts.append(f"best val loss={best_loss:.3f}")
        if caption_parts:
            ax.text(0.02, 0.02, " | ".join(caption_parts), transform=ax.transAxes, fontsize=8, va="bottom")

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


def main():
    results_root = Path("results")
    results = collect_entries(results_root)

    if not results:
        print("No results found in results/overall_report.json, results/metrics_summary.json, or per-model history files.")
        return

    processed = []
    for result in results:
        history = extract_history(result)
        if not history:
            continue

        processed.append(
            {
                "label": infer_model_name(result),
                "color": None,
                "history": history,
                "accuracy": result.get("test_accuracy", result.get("accuracy", 0.0)),
                "macro_f1": result.get("test_macro_f1", result.get("macro_f1", 0.0)),
                "best_val_loss": result.get("best_val_loss"),
                "epochs_trained": result.get("epochs_trained"),
            }
        )

    if not processed:
        print("No usable training history found in the available results.")
        return

    colors = get_colors(len(processed))
    for idx, item in enumerate(processed):
        item["color"] = colors[idx]

    labels = [r["label"] for r in processed]
    accuracies = [r["accuracy"] for r in processed]
    f1s = [r["macro_f1"] for r in processed]

    base_colors = ["green", "red", "blue", "orange", "purple", "brown", "cyan", "magenta", "olive", "gray"]
    if len(labels) > len(base_colors):
        bar_colors = list(itertools.islice(itertools.cycle(base_colors), len(labels)))
    else:
        bar_colors = base_colors[: len(labels)]

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

    bars_acc = axes[0].bar(x, accuracies, color=bar_colors)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy per model")
    annotate_bars(axes[0], bars_acc, fmt="{:.3f}", highlights={idx_best_acc})

    bars_f1 = axes[1].bar(x, f1s, color=bar_colors)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xticks(x)
    axes[1].set_title("Macro-F1 per model")
    # hide x-axis labels (we use legend for mapping)
    axes[1].tick_params(axis="x", labelbottom=False)
    annotate_bars(axes[1], bars_f1, fmt="{:.3f}", highlights={idx_best_f1})

    # legend mapping index -> model name (colored)
    patches = [mpatches.Patch(color=bar_colors[i], label=f"{i+1}. {labels[i]}") for i in range(len(labels))]
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
    txt1 = f"Best Accuracy: {shorten_label(labels[idx_best_acc], max_len=10)} = {best_acc:.3f}"
    txt2 = f"Best Macro F1: {shorten_label(labels[idx_best_f1], max_len=10)} = {best_f1:.3f}"
    box_ax.text(0.02, y1, txt1, ha="left", va="center", fontsize=9)
    box_ax.text(0.02, y2, txt2, ha="left", va="center", fontsize=9)

    # draw colored squares at the end of each line inside the box
    sq_w = 0.07
    sq_h = 0.25
    # right-aligned squares
    rect1 = mpatches.Rectangle((0.90, y1 - sq_h / 2), sq_w, sq_h, transform=box_ax.transAxes, facecolor=colors[idx_best_acc], edgecolor="none")
    rect2 = mpatches.Rectangle((0.90, y2 - sq_h / 2), sq_w, sq_h, transform=box_ax.transAxes, facecolor=colors[idx_best_f1], edgecolor="none")
    box_ax.add_patch(rect1)
    box_ax.add_patch(rect2)

    plt.tight_layout()
    out_path = results_root / "comparison_metrics.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {out_path}")

    loss_out = results_root / "comparison_history_loss.png"
    acc_out = results_root / "comparison_history_accuracy.png"
    grid_out = results_root / "comparison_history_grid.png"

    plot_metric_overview(
        processed,
        metric_name="loss",
        out_path=loss_out,
        ylabel="Loss",
        title="Training and validation loss by model",
    )
    print(f"Saved loss history comparison to {loss_out}")

    plot_metric_overview(
        processed,
        metric_name="acc",
        out_path=acc_out,
        ylabel="Accuracy",
        title="Training and validation accuracy by model",
    )
    print(f"Saved accuracy history comparison to {acc_out}")

    plot_model_grid(processed, grid_out)
    if grid_out.exists():
        print(f"Saved per-model history grid to {grid_out}")


if __name__ == "__main__":
    main()
