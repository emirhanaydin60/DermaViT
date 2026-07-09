"""Main result-generation script with editable in-file parameters."""

from __future__ import annotations

from pathlib import Path

from misc.comparison_confusion_grid import plot_confusion_matrix_grid
from misc.comparison_history_grid import plot_model_grid
from misc.comparison_history_plot import plot_history_loss_accuracy_comparison
from misc.comparison_overview_plot import plot_metrics_comparison
from misc.comparison_metrics_common import build_processed_entries, load_training_time_entries
from misc.comparison_training_time_plot import generate_publication_figures, plot_training_time_comparison

# Edit these values directly.
RESULTS_ROOT = Path("previous_results/results_top3_val_acc")
BEST_HISTORY_METRIC = "acc"
HISTORY_GRID_COLS = 3
HISTORY_GRID_ROWS = 1
CONFUSION_GRID_COLS = 3
CONFUSION_GRID_ROWS = 1


def main():
    results_root = RESULTS_ROOT
    processed = build_processed_entries(results_root)

    if not processed:
        print("No usable training history found in the available results.")
        return

    plot_metrics_comparison(processed, results_root / "comparison_metrics.png", results_root=results_root)
    print(f"Saved comparison plot to {results_root / 'comparison_metrics.png'}")

    plot_history_loss_accuracy_comparison(processed, results_root / "comparison_history_loss_accuracy.png")
    print(f"Saved loss/accuracy history comparison to {results_root / 'comparison_history_loss_accuracy.png'}")

    plot_model_grid(
        processed,
        results_root / "comparison_history_grid.png",
        best_metric_name=BEST_HISTORY_METRIC,
        grid_cols=HISTORY_GRID_COLS,
        grid_rows=HISTORY_GRID_ROWS,
    )
    print(f"Saved per-model history grid to {results_root / 'comparison_history_grid.png'}")

    generate_publication_figures(results_root)
    print(f"Saved trade-off figure to {results_root / 'tradeoff_bubble.png'}")
    print(f"Saved accuracy figure to {results_root / 'accuracy_vs_parameters.png'}")
    print(f"Saved training-time figure to {results_root / 'training_time_vs_parameters.png'}")

    cm_models = [item for item in processed if item.get("confusion_matrix") is not None]
    if cm_models:
        plot_confusion_matrix_grid(cm_models, results_root / "comparison_confusion_matrix_grid.png", grid_cols=CONFUSION_GRID_COLS, grid_rows=CONFUSION_GRID_ROWS)
    else:
        print("No confusion matrices found in the available results; skipping confusion matrix grid.")


if __name__ == "__main__":
    main()
