"""Create a compact summary from `results/overall_report.json`.

This script intentionally excludes long fields such as:
- `classes`
- `test_confusion_matrix`
- `training_history`

It writes a short JSON summary to `results/overall_report_summary.json`
and prints a compact table to the console.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

LONG_FIELDS = {"classes", "test_confusion_matrix", "training_history"}


def load_report(report_path: Path) -> list[dict]:
    if not report_path.exists():
        raise FileNotFoundError(f"overall report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("overall_report.json must contain a JSON array")

    return data


def short_model_name(entry: dict) -> str:
    return entry.get("model") or entry.get("model_key") or Path(entry.get("checkpoint_file", "")).stem or "unknown"


def safe_round(value, digits=4):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return value


def summarize_entry(entry: dict) -> dict:
    return {
        "model": short_model_name(entry),
        "checkpoint_file": entry.get("checkpoint_file") or entry.get("file"),
        "test_accuracy": safe_round(entry.get("test_accuracy", entry.get("accuracy"))),
        "test_macro_precision": safe_round(entry.get("test_macro_precision", entry.get("macro_precision"))),
        "test_macro_recall": safe_round(entry.get("test_macro_recall", entry.get("macro_recall"))),
        "test_macro_f1": safe_round(entry.get("test_macro_f1", entry.get("macro_f1"))),
        "epochs_trained": entry.get("epochs_trained"),
        "requested_epochs": entry.get("requested_epochs"),
        "early_stopped": entry.get("early_stopped"),
        "total_training_time_min": safe_round(entry.get("total_training_time_min")),
        "avg_epoch_time_min": safe_round(entry.get("avg_epoch_time_min")),
        "num_parameters_millions": safe_round(entry.get("num_parameters_millions"), 3),
        "best_val_loss": safe_round(entry.get("best_val_loss"), 6),
        "best_val_f1": safe_round(entry.get("best_val_f1"), 6),
    }


def build_summary(report: list[dict]) -> dict:
    models = [summarize_entry(entry) for entry in report]

    accuracies = [m["test_accuracy"] for m in models if isinstance(m.get("test_accuracy"), (int, float))]
    f1s = [m["test_macro_f1"] for m in models if isinstance(m.get("test_macro_f1"), (int, float))]

    best_accuracy_model = max(models, key=lambda x: x["test_accuracy"] if isinstance(x.get("test_accuracy"), (int, float)) else float("-inf"), default=None)
    best_f1_model = max(models, key=lambda x: x["test_macro_f1"] if isinstance(x.get("test_macro_f1"), (int, float)) else float("-inf"), default=None)

    return {
        "model_count": len(models),
        "mean_test_accuracy": safe_round(mean(accuracies)) if accuracies else None,
        "mean_test_macro_f1": safe_round(mean(f1s)) if f1s else None,
        "best_accuracy_model": best_accuracy_model,
        "best_macro_f1_model": best_f1_model,
        "models": models,
    }


def print_compact_summary(summary: dict) -> None:
    print(f"Models: {summary['model_count']}")
    if summary.get("mean_test_accuracy") is not None:
        print(f"Mean accuracy: {summary['mean_test_accuracy']}")
    if summary.get("mean_test_macro_f1") is not None:
        print(f"Mean macro F1: {summary['mean_test_macro_f1']}")

    best_acc = summary.get("best_accuracy_model")
    best_f1 = summary.get("best_macro_f1_model")
    if best_acc:
        print(f"Best accuracy: {best_acc['model']} -> {best_acc['test_accuracy']}")
    if best_f1:
        print(f"Best macro F1: {best_f1['model']} -> {best_f1['test_macro_f1']}")

    print("")
    print("Per-model summary:")
    for item in summary["models"]:
        print(f"- {item['model']}: acc={item['test_accuracy']}, f1={item['test_macro_f1']}, " f"epochs={item['epochs_trained']}, time_min={item['total_training_time_min']}, " f"best_val_loss={item['best_val_loss']}")


def main(path) -> None:
    parser = argparse.ArgumentParser(description="Summarize results/overall_report.json into a compact JSON file.")
    parser.add_argument("--report", default=f"{path}/overall_report.json", help="Path to overall_report.json")
    parser.add_argument(
        "--output",
        default=f"{path}/overall_report_summary.json",
        help="Path to the compact summary JSON file",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output)

    report = load_report(report_path)
    summary = build_summary(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved compact summary to {output_path}")
    print_compact_summary(summary)


if __name__ == "__main__":
    main("results_top3_val_acc")
