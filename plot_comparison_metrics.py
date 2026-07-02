"""Plot comparison metrics and training histories from the results folder.

Outputs:
- `results/comparison_metrics.png`: accuracy and macro F1 bar charts.
- `results/comparison_training_time.png`: horizontal training time comparison.
- `results/comparison_history_loss.png`: all models' train/val loss curves on one plot.
- `results/comparison_history_accuracy.png`: all models' train/val accuracy curves on one plot.
- `results/comparison_history_grid.png`: one image with per-model history panels.
- `results/comparison_confusion_matrix_grid.png`: one image with per-model confusion matrices.

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


def load_overall_report_summary(results_root=Path("results")):
    summary_path = results_root / "overall_report_summary.json"
    return load_json_if_exists(summary_path)


def load_training_time_entries(results_root=Path("results")):
    summary = load_overall_report_summary(results_root)
    if isinstance(summary, dict):
        models = summary.get("models", [])
        if models:
            return models

    report_path = results_root / "overall_report.json"
    report = load_json_if_exists(report_path)
    if isinstance(report, list):
        return report

    metrics_path = results_root / "metrics_summary.json"
    metrics = load_json_if_exists(metrics_path)
    if isinstance(metrics, list):
        return metrics

    return []


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
    handles = color_handles + [style_handle_train, style_handle_val]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_time_comparison(models_data, out_path, title="Training Time Comparison"):
    items = []
    for item in models_data:
        model_name = item.get("model") or item.get("label") or item.get("model_key") or "unknown"
        training_time = item.get("total_training_time_min")
        if training_time is None:
            continue
        try:
            training_time = float(training_time)
        except Exception:
            continue
        items.append({"model": model_name, "training_time": training_time})

    if not items:
        return

    items.sort(key=lambda item: item["training_time"], reverse=True)

    labels = [shorten_label(item["model"], max_len=24) for item in items]
    values = [item["training_time"] for item in items]
    colors = get_colors(len(items))
    y_positions = np.arange(len(items))

    fig_height = max(4.5, 0.55 * len(items) + 1.5)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    bars = ax.barh(y_positions, values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Models")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    xmax = max(values)
    ax.set_xlim(0, xmax * 1.15 if xmax > 0 else 1.0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_best_history_point(history, metric_name):
    values = history.get(f"val_{metric_name}", [])
    values = trim_series(values)
    if not values:
        return None, None

    values = values[0]
    if not values:
        return None, None

    if metric_name == "loss":
        best_idx = int(np.argmin(values))
    else:
        best_idx = int(np.argmax(values))

    return best_idx, values[best_idx]


def plot_model_grid(models_data, out_path, best_metric_name="loss", grid_cols=3):
    count = len(models_data)
    if count == 0:
        return

    cols = max(1, int(grid_cols))
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

        ax.set_title(shorten_label(label, max_len=18), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color=loss_color)
        ax2.set_ylabel("Acc", color=acc_color)
        ax.tick_params(axis="y", labelcolor=loss_color)
        ax2.tick_params(axis="y", labelcolor=acc_color)
        ax.grid(True, alpha=0.2)

        best_metric_caption = None
        if best_history_idx is not None:
            best_metric_caption = f"best val {best_metric_name}={best_history_value:.3f}"

        best_loss = item.get("best_val_loss")
        epochs_trained = item.get("epochs_trained")
        caption_parts = []
        if epochs_trained is not None:
            caption_parts.append(f"ep={epochs_trained}")
        if best_metric_caption is not None:
            caption_parts.append(best_metric_caption)
        elif best_loss is not None:
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


def plot_confusion_matrix_grid(models_data, out_path, class_names=None, grid_cols=3):
    count = len(models_data)
    if count == 0:
        return

    cols = max(1, int(grid_cols))
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.8))
    axes = np.array(axes).reshape(-1)

    matrices = []
    for item in models_data:
        cm = item.get("confusion_matrix")
        if cm is None:
            continue
        cm = np.asarray(cm)
        if cm.size == 0:
            continue
        matrices.append(cm)

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

        ax.set_title(shorten_label(item["label"], max_len=18), fontsize=10)
        ax.set_xticks(np.arange(len(item_class_names)))
        ax.set_yticks(np.arange(len(item_class_names)))
        ax.set_xticklabels(item_class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(item_class_names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)

        best_f1 = item.get("macro_f1")
        best_acc = item.get("accuracy")
        caption_parts = []
        if best_acc is not None:
            caption_parts.append(f"acc={best_acc:.3f}")
        if best_f1 is not None:
            caption_parts.append(f"f1={best_f1:.3f}")
        if caption_parts:
            ax.text(0.02, 0.02, " | ".join(caption_parts), transform=ax.transAxes, fontsize=8, va="bottom")

    for idx in range(count, len(axes)):
        axes[idx].set_axis_off()

    fig.suptitle("Test confusion matrices by model", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(path="results", best_history_metric="loss", grid_cols=3):
    results_root = Path(path)
    results = collect_entries(results_root)
    training_time_entries = load_training_time_entries(results_root)

    if not results:
        print("No results found in results/overall_report.json, results/metrics_summary.json, or per-model history files.")
        return

    processed = []
    for result in results:
        history = extract_history(result)
        if not history:
            continue

        confusion_matrix = result.get("test_confusion_matrix")
        if confusion_matrix is None:
            confusion_matrix = result.get("confusion_matrix")

        class_names = result.get("classes") or result.get("class_names") or result.get("class_order")

        processed.append(
            {
                "label": infer_model_name(result),
                "color": None,
                "history": history,
                "accuracy": result.get("test_accuracy", result.get("accuracy", 0.0)),
                "macro_f1": result.get("test_macro_f1", result.get("macro_f1", 0.0)),
                "total_training_time_min": result.get("total_training_time_min"),
                "best_val_loss": result.get("best_val_loss"),
                "epochs_trained": result.get("epochs_trained"),
                "confusion_matrix": confusion_matrix,
                "class_names": class_names,
            }
        )

    if not processed:
        print("No usable training history found in the available results.")
        return

    colors = get_colors(len(processed))
    for idx, item in enumerate(processed):
        item["color"] = colors[idx]

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
            fontweight = "bold" if i in highlights else "normal"
            if i in highlights:
                label = label + " *"
            x_pos = bar.get_x() + bar.get_width() / 2
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

    bar_width = 0.48

    bars_acc = axes[0].bar(x, accuracies, color=bar_colors, width=bar_width)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy per model")
    annotate_bars(axes[0], bars_acc, fmt="{:.3f}", highlights={idx_best_acc})

    bars_f1 = axes[1].bar(x, f1s, color=bar_colors, width=bar_width)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xticks(x)
    axes[1].set_title("Macro-F1 per model")
    axes[1].tick_params(axis="x", labelbottom=False)
    annotate_bars(axes[1], bars_f1, fmt="{:.3f}", highlights={idx_best_f1})

    patches = [mpatches.Patch(color=bar_colors[i], label=f"#{i+1} {labels[i]}") for i in range(len(labels))]
    axes[0].legend(
        handles=patches,
        title="Model order\n(Macro F1 desc)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=(0, 0.10, 0.76, 0.98))
    fig.canvas.draw()

    out_path = results_root / "comparison_metrics.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")

    loss_out = results_root / "comparison_history_loss.png"
    acc_out = results_root / "comparison_history_accuracy.png"
    grid_out = results_root / "comparison_history_grid.png"
    cm_grid_out = results_root / "comparison_confusion_matrix_grid.png"
    training_time_out = results_root / "comparison_training_time.png"

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

    plot_model_grid(processed, grid_out, best_metric_name=best_history_metric, grid_cols=grid_cols)
    if grid_out.exists():
        print(f"Saved per-model history grid to {grid_out}")

    if not training_time_entries:
        training_time_entries = processed
    plot_training_time_comparison(training_time_entries, training_time_out)
    if training_time_out.exists():
        print(f"Saved training time comparison to {training_time_out}")

    cm_models = [item for item in processed if item.get("confusion_matrix") is not None]
    if cm_models:
        plot_confusion_matrix_grid(cm_models, cm_grid_out, grid_cols=grid_cols)
        if cm_grid_out.exists():
            print(f"Saved per-model confusion matrix grid to {cm_grid_out}")
    else:
        print("No confusion matrices found in the available results; skipping confusion matrix grid.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot comparison metrics from a results folder.")
    parser.add_argument("path", nargs="?", default="results_top3_val_acc", help="Path to the results folder")
    parser.add_argument("--best-history-metric", default="loss", choices=["loss", "acc"], help="Metric used to highlight the best epoch in the history grid")
    parser.add_argument("--grid-cols", type=int, default=3, help="Number of columns in the per-model history grid")
    args = parser.parse_args()

    main(args.path, best_history_metric=args.best_history_metric, grid_cols=args.grid_cols)
