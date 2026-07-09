"""Main entry point for comparison plot generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from misc.comparison_confusion_grid import plot_confusion_matrix_grid
from misc.comparison_history_grid import plot_model_grid
from misc.comparison_history_plot import plot_history_loss_accuracy_comparison
from misc.comparison_overview_plot import plot_metrics_comparison
from misc.comparison_metrics_common import build_processed_entries, load_training_time_entries, plot_training_time_comparison

DEFAULT_RESULTS_ROOT = Path("previous_results/9_vit_results")
DEFAULT_HISTORY_GRID_COLS = 2
DEFAULT_HISTORY_GRID_ROWS = 2
DEFAULT_CONFUSION_GRID_COLS = 2
DEFAULT_CONFUSION_GRID_ROWS = 2
DEFAULT_INCLUDE_CONFUSION_GRID = False


def main(
    path=DEFAULT_RESULTS_ROOT,
    best_history_metric="loss",
    history_grid_cols=DEFAULT_HISTORY_GRID_COLS,
    history_grid_rows=DEFAULT_HISTORY_GRID_ROWS,
    confusion_grid_cols=DEFAULT_CONFUSION_GRID_COLS,
    confusion_grid_rows=DEFAULT_CONFUSION_GRID_ROWS,
    include_confusion_grid=DEFAULT_INCLUDE_CONFUSION_GRID,
):
    results_root = Path(path)
    processed = build_processed_entries(results_root)
    training_time_entries = load_training_time_entries(results_root) or processed

    if not processed:
        print("No usable training history found in the available results.")
        return

    # Overview metrics
    plot_metrics_comparison(processed, results_root / "comparison_metrics.png", results_root=results_root)
    print(f"Saved comparison plot to {results_root / 'comparison_metrics.png'}")

    # Combined history curves
    plot_history_loss_accuracy_comparison(processed, results_root / "comparison_history_loss_accuracy.png")
    print(f"Saved loss/accuracy history comparison to {results_root / 'comparison_history_loss_accuracy.png'}")

    # Per-model history panels
    plot_model_grid(
        processed,
        results_root / "comparison_history_grid.png",
        best_metric_name=best_history_metric,
        grid_cols=history_grid_cols,
        grid_rows=history_grid_rows,
    )
    print(f"Saved per-model history grid to {results_root / 'comparison_history_grid.png'}")

    # Training time
    plot_training_time_comparison(training_time_entries, results_root / "comparison_training_time.png")
    print(f"Saved training time comparison to {results_root / 'comparison_training_time.png'}")

    # Confusion matrices are optional so you can comment this block out or flip the flag.
    if include_confusion_grid:
        cm_models = [item for item in processed if item.get("confusion_matrix") is not None]
        if cm_models:
            plot_confusion_matrix_grid(
                cm_models,
                results_root / "comparison_confusion_matrix_grid.png",
                grid_cols=confusion_grid_cols,
                grid_rows=confusion_grid_rows,
            )
            print(f"Saved per-model confusion matrix grid to {results_root / 'comparison_confusion_matrix_grid.png'}")
        else:
            print("No confusion matrices found in the available results; skipping confusion matrix grid.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate comparison plots from a results folder.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_RESULTS_ROOT), help="Path to the results folder")
    parser.add_argument("--best-history-metric", default="loss", choices=["loss", "acc"], help="Metric used to highlight the best epoch in the history grid")
    parser.add_argument("--history-grid-cols", type=int, default=DEFAULT_HISTORY_GRID_COLS, help="Number of columns in the per-model history grid")
    parser.add_argument("--history-grid-rows", type=int, default=DEFAULT_HISTORY_GRID_ROWS, help="Number of rows in the per-model history grid")
    parser.add_argument("--confusion-grid-cols", type=int, default=DEFAULT_CONFUSION_GRID_COLS, help="Number of columns in the confusion matrix grid")
    parser.add_argument("--confusion-grid-rows", type=int, default=DEFAULT_CONFUSION_GRID_ROWS, help="Number of rows in the confusion matrix grid")
    parser.add_argument("--confusion-grid", action="store_true", help="Also generate the confusion matrix grid plot")
    return parser.parse_args(argv)


def cli(argv=None):
    args = parse_args(argv)
    main(
        path=args.path,
        best_history_metric=args.best_history_metric,
        history_grid_cols=args.history_grid_cols,
        history_grid_rows=args.history_grid_rows,
        confusion_grid_cols=args.confusion_grid_cols,
        confusion_grid_rows=args.confusion_grid_rows,
        include_confusion_grid=args.confusion_grid,
    )


if __name__ == "__main__":
    cli()
