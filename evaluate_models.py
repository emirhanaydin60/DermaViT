"""Evaluate newly added .pth models in `results/` on the test set and
append their metrics to `results/metrics_summary.json`.

Usage: run from repo root:
        python evaluate_models.py

Behavior:
 - Reads existing `results/metrics_summary.json` if present.
 - Finds all .pth files under `results/` and `finetune/results/`.
 - Evaluates only those .pth files that are NOT already present in
     `metrics_summary.json` (by file path).
 - Appends new per-model results to `metrics_summary.json` and saves a
     per-model confusion matrix image under `results/<model>/`.

Note: comparison plot generation was moved to a separate script
`plot_comparison_metrics.py`.
"""

import os
import re
import json
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets

import timm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def get_test_loader(data_dir, image_size=224, batch_size=32, num_workers=4):
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found under {data_dir}/test")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms = T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader, test_ds.classes


def find_pth_files(root_dirs=("results", "finetune/results")):
    files = []
    for rd in root_dirs:
        p = Path(rd)
        if not p.exists():
            continue
        for fp in p.rglob("*.pth"):
            files.append(fp)
    return sorted(files)


def infer_model_name_from_filename(fname):
    # Expect pattern like <model_name>_finetuned*.pth
    # Use simple split to avoid accidental short matches
    base = Path(fname).name
    if "_finetuned" in base:
        return base.split("_finetuned", 1)[0]
    return Path(fname).stem


def resolve_timm_model_key(candidate):
    available = timm.list_models()
    if candidate in available:
        return candidate
    # try exact prefix match
    for a in available:
        if a.startswith(candidate):
            return a
    # try contains
    for a in available:
        if candidate in a:
            return a
    # difflib fallback
    try:
        import difflib

        close = difflib.get_close_matches(candidate, available, n=1)
        if close:
            return close[0]
    except Exception:
        pass
    return None


def build_model(model_key, num_classes, device, pretrained=False):
    model = timm.create_model(model_key, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)


def evaluate_checkpoint(pth_path, test_loader, device):
    ckpt = torch.load(str(pth_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        classes = ckpt.get("classes", None)
    else:
        # assume full model saved
        state = None
        classes = None

    model_name = infer_model_name_from_filename(pth_path.name)
    print(f"Evaluating file {pth_path.name} -> model candidate '{model_name}'")

    # determine classes/num_classes
    if classes is None:
        # try to infer from test loader dataset
        try:
            classes = test_loader.dataset.classes
        except Exception:
            classes = None
    num_classes = len(classes) if classes is not None else None

    # resolve timm model key
    model_key = resolve_timm_model_key(model_name)
    if model_key is None:
        print(f"  Could not resolve timm model for candidate '{model_name}', skipping.")
        return None

    if num_classes is None:
        # fallback: try default 1000
        num_classes = 1000

    model = build_model(model_key, num_classes=num_classes, device=device, pretrained=False)
    if state is not None:
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"  Warning: load_state_dict failed strict=True, trying strict=False: {e}")
            model.load_state_dict(state, strict=False)

    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, p = torch.max(outputs, 1)
            preds.extend(p.cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())

    if len(labels) == 0:
        print(f"  No test samples found, skipping {pth_path}")
        return None

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "file": str(pth_path),
        "model_key": model_key,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def save_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.title(Path(out_path).stem)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# comparison plotting is handled in `plot_comparison_metrics.py`


def main():
    data_dir = r"C:/Users/emirh/Desktop/Projects/datasets/input_sk"
    image_size = 224
    batch_size = 32
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    test_loader, test_classes = get_test_loader(data_dir, image_size=image_size, batch_size=batch_size, num_workers=num_workers)

    pth_files = find_pth_files()
    if not pth_files:
        print("No .pth files found under results/ or finetune/results/")
        return

    out_root = Path("results")
    out_root.mkdir(exist_ok=True)
    metrics_path = out_root / "metrics_summary.json"

    # load existing metrics (if any)
    existing_results = []
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read existing metrics at {metrics_path}: {e}")
            existing_results = []

    # Build set of already-tested file absolute paths for quick membership checks
    tested_set = set()
    for r in existing_results:
        file_field = r.get("file")
        if not file_field:
            continue
        try:
            tested_set.add(str(Path(file_field).resolve(strict=False)))
        except Exception:
            tested_set.add(file_field)

    # determine which .pth files are new (not present in metrics_summary.json)
    new_pth_files = []
    for p in pth_files:
        try:
            resolved = str(p.resolve(strict=False))
        except Exception:
            resolved = str(p)
        if resolved not in tested_set:
            new_pth_files.append(p)

    if not new_pth_files:
        print("No new models to evaluate. All found .pth files are already in metrics_summary.json")
        return

    new_results = []
    for p in new_pth_files:
        res = evaluate_checkpoint(p, test_loader, device)
        if res is None:
            continue
        # if classes missing, fill with test_classes
        if res["classes"] is None:
            res["classes"] = test_classes
        model_name = infer_model_name_from_filename(p.name)
        model_dir = out_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        cm_path = model_dir / "confusion_matrix_eval.png"
        save_confusion_matrix(res["confusion_matrix"], res["classes"], cm_path)
        new_results.append(res)

    if new_results:
        combined = existing_results + new_results
        try:
            with open(metrics_path, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"Appended {len(new_results)} new result(s) to {metrics_path}")
        except Exception as e:
            print(f"Error writing metrics_summary.json: {e}")
    else:
        print("No new evaluation results produced.")

    print("Evaluation finished. Per-model confusion matrices saved under results/<model>/")


if __name__ == "__main__":
    main()
