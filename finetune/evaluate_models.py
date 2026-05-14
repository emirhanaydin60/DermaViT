"""Evaluate all saved .pth models in `results/` on the test set and
produce comparison metrics and plots.

Usage: run from repo root:
    python finetune/evaluate_models.py

This script:
 - Finds all .pth files under `results/` and `finetune/results/`.
 - Loads each checkpoint (expects dict with 'model_state_dict' and optionally 'classes').
 - Reconstructs model via timm, loads state_dict (strict=False), runs inference on test set.
 - Computes accuracy, macro F1, confusion matrix.
 - Saves per-model confusion matrix images and a single comparison plot.
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
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found under {data_dir}/test")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transforms = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader, test_ds.classes


def find_pth_files(root_dirs=('results', 'finetune/results')):
    files = []
    for rd in root_dirs:
        p = Path(rd)
        if not p.exists():
            continue
        for fp in p.rglob('*.pth'):
            files.append(fp)
    return sorted(files)


def infer_model_name_from_filename(fname):
    # Expect pattern like <model_name>_finetuned*.pth
    # Use simple split to avoid accidental short matches
    base = Path(fname).name
    if '_finetuned' in base:
        return base.split('_finetuned', 1)[0]
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
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        classes = ckpt.get('classes', None)
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
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        'file': str(pth_path),
        'model_key': model_key,
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'classes': classes,
    }


def save_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title(Path(out_path).stem)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_comparison_plot(results, out_path):
    models = [Path(r['file']).stem for r in results]
    accs = [r['accuracy'] for r in results]
    f1s = [r['macro_f1'] for r in results]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(max(8, len(models)*1.2), 6))
    plt.bar(x - width/2, accs, width, label='Accuracy')
    plt.bar(x + width/2, f1s, width, label='Macro F1')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Model comparison on test set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    data_dir = r'C:/Users/emirh/Desktop/Projects/datasets/input_sk'
    image_size = 224
    batch_size = 32
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device:', device)

    test_loader, test_classes = get_test_loader(data_dir, image_size=image_size, batch_size=batch_size, num_workers=num_workers)

    pth_files = find_pth_files()
    if not pth_files:
        print('No .pth files found under results/ or finetune/results/')
        return

    results = []
    out_root = Path('results')
    out_root.mkdir(exist_ok=True)

    for p in pth_files:
        res = evaluate_checkpoint(p, test_loader, device)
        if res is None:
            continue
        # if classes missing, fill with test_classes
        if res['classes'] is None:
            res['classes'] = test_classes
        model_name = infer_model_name_from_filename(p.name)
        model_dir = out_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        cm_path = model_dir / 'confusion_matrix_eval.png'
        save_confusion_matrix(res['confusion_matrix'], res['classes'], cm_path)
        results.append(res)

    # save metrics summary
    with open(out_root / 'metrics_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # comparison plot
    if results:
        make_comparison_plot(results, out_root / 'comparison_metrics.png')
        print('Saved comparison_metrics.png and per-model confusion matrices under results/<model>/')
    else:
        print('No results to plot')


if __name__ == '__main__':
    main()
