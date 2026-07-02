"""Plot parameter counts for the main ViT-family model variants.

The script builds one grouped chart for these families:
- beit
- cait
- convit
- deit
- maxvit
- mobilevit
- pvt_v2
- swin
- vit

It prefers the model registry in `timm` and uses the current project's
saved summaries as a fallback cache for parameter counts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import timm

FAMILY_VARIANTS = {
    "vit": [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "vit_large_patch14_224",
        "vit_huge_patch14_224",
    ],
    "swin": [
        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        "swin_large_patch4_window7_224",
    ],
    "pvt_v2": [
        "pvt_v2_b0",
        "pvt_v2_b1",
        "pvt_v2_b2",
        "pvt_v2_b2_li",
        "pvt_v2_b3",
        "pvt_v2_b4",
        "pvt_v2_b5",
    ],
    "mobilevit": [
        "mobilevit_xxs",
        "mobilevit_xs",
        "mobilevit_s",
    ],
    "maxvit": [
        "maxvit_tiny_rw_224",
        "maxvit_small_rw_224",
        "maxvit_base_rw_224",
        "maxvit_large_rw_224",
    ],
    "deit": [
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
    ],
    "convit": [
        "convit_tiny",
        "convit_small",
        "convit_base",
    ],
    "cait": [
        "cait_xxs24_224",
        "cait_xxs36_224",
        "cait_xs24_384",
        "cait_s24_224",
        "cait_s36_384",
        "cait_m36_384",
        "cait_m48_448",
    ],
    "beit": [
        "beit_base_patch16_224",
        "beit_large_patch16_224",
    ],
}

FAMILY_COLORS = {
    "vit": "#264653",
    "swin": "#e76f51",
    "pvt_v2": "#2a9d8f",
    "mobilevit": "#f39c12",
    "maxvit": "#8e5ea2",
    "deit": "#1f77b4",
    "convit": "#7f7f7f",
    "cait": "#d62728",
    "beit": "#6a994e",
}

FAMILY_DISPLAY = {
    "vit": "ViT",
    "swin": "Swin",
    "pvt_v2": "PVT v2",
    "mobilevit": "MobileViT",
    "maxvit": "MaxViT",
    "deit": "DeiT",
    "convit": "ConViT",
    "cait": "CaiT",
    "beit": "BEiT",
}

HIGHLIGHTED_VARIANTS = {
    "beit": {"beit_base_patch16_224"},
    "cait": {"cait_xxs36_224"},
    "convit": {"convit_tiny"},
    "deit": {"deit_small_patch16_224"},
    "maxvit": {"maxvit_tiny_rw_224"},
    "mobilevit": {"mobilevit_xs"},
    "pvt_v2": {"pvt_v2_b0"},
    "swin": {"swin_small_patch4_window7_224"},
    "vit": {"vit_small_patch16_224"},
}

MODEL_ALIASES = {
    "maxvit_tiny_rw_224": ["maxvit_tiny_tf_224"],
    "maxvit_small_rw_224": ["maxvit_small_tf_224"],
    "maxvit_base_rw_224": ["maxvit_base_tf_224"],
    "maxvit_large_rw_224": ["maxvit_large_tf_224"],
}


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None


def collect_parameter_cache(workspace_root: Path) -> dict[str, float]:
    cache: dict[str, float] = {}

    for path in sorted(workspace_root.rglob("training_summary.json")):
        data = load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        model_name = data.get("model") or path.parent.name
        params = data.get("num_parameters_millions")
        if model_name and isinstance(params, (int, float)):
            cache.setdefault(str(model_name), float(params))

    for path in sorted(workspace_root.rglob("overall_report_summary.json")):
        data = load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        models = data.get("models")
        if not isinstance(models, list):
            continue
        for item in models:
            if not isinstance(item, dict):
                continue
            model_name = item.get("model") or item.get("model_key")
            params = item.get("num_parameters_millions")
            if model_name and isinstance(params, (int, float)):
                cache.setdefault(str(model_name), float(params))

    for path in sorted(workspace_root.rglob("overall_report.json")):
        data = load_json_if_exists(path)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            model_name = item.get("model") or item.get("model_key")
            params = item.get("num_parameters_millions")
            if model_name and isinstance(params, (int, float)):
                cache.setdefault(str(model_name), float(params))

    return cache


def resolve_parameter_count(model_name: str, num_classes: int, cache: dict[str, float]) -> float:
    cached = cache.get(model_name)
    if cached is not None:
        return float(cached)

    candidate_names = [model_name, *MODEL_ALIASES.get(model_name, [])]
    last_error: Exception | None = None

    for candidate_name in candidate_names:
        try:
            model = timm.create_model(candidate_name, pretrained=False, num_classes=num_classes)
        except Exception as exc:
            last_error = exc
            continue

        try:
            return round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
        finally:
            del model

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to resolve model name: {model_name}")


def short_variant_label(family: str, model_name: str) -> str:
    if model_name.startswith(family + "_"):
        label = model_name[len(family) + 1 :]
    else:
        label = model_name
    label = re.sub(r"_(224|256|384|448|512)$", "", label)
    label = label.replace("_rw", "").replace("_tf", "")
    return label


def lighten_color(color: str, whiteness: float) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    return tuple((base * (1.0 - whiteness) + white * whiteness).clip(0.0, 1.0))


def build_series(workspace_root: Path, num_classes: int) -> list[dict]:
    cache = collect_parameter_cache(workspace_root)
    series: list[dict] = []

    for family, model_names in FAMILY_VARIANTS.items():
        items = []
        for model_name in model_names:
            try:
                params_m = resolve_parameter_count(model_name, num_classes=num_classes, cache=cache)
            except Exception as exc:
                print(f"Warning: skipping {model_name}: {exc}")
                continue
            items.append(
                {
                    "family": family,
                    "model_name": model_name,
                    "variant_label": short_variant_label(family, model_name),
                    "params_m": float(params_m),
                }
            )

        items.sort(key=lambda item: item["params_m"])
        series.extend(items)

    return series


def plot_parameter_counts(series: list[dict], output_path: Path) -> None:
    if not series:
        print("No model variants found to plot.")
        return

    fig_width = max(20.0, len(series) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, 10))

    family_groups: list[tuple[str, float, float]] = []
    x_positions: list[float] = []
    x_labels: list[str] = []
    bar_colors: list[tuple[float, float, float]] = []
    values: list[float] = []

    x = 0.0
    gap = 1.0
    family_to_items: dict[str, list[dict]] = {}
    for item in series:
        family_to_items.setdefault(item["family"], []).append(item)

    for family, items in family_to_items.items():
        start = x
        n = len(items)
        shade_values = np.linspace(0.55, 0.12, num=max(n, 1))
        for idx, item in enumerate(items):
            x_positions.append(x)
            x_labels.append(item["variant_label"])
            bar_colors.append(lighten_color(FAMILY_COLORS[family], float(shade_values[idx])))
            values.append(item["params_m"])
            x += 1.0
        end = x - 1.0
        family_groups.append((family, start, end))
        x += gap

    max_value = max(values)
    bottom = max(0.5, min(values) * 0.75)

    bars = ax.bar(x_positions, values, width=0.78, color=bar_colors, edgecolor="white", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("Parameters (millions, log scale)")
    ax.set_title("Model parameter counts by family and variant")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    for bar, value, item in zip(bars, values, series):
        family = item["family"]
        is_highlighted = item["model_name"] in HIGHLIGHTED_VARIANTS.get(family, set())
        if is_highlighted:
            bar.set_edgecolor(FAMILY_COLORS[family])
            bar.set_linewidth(5.0)
            ax.annotate(
                "\n\n*",
                xy=(bar.get_x() + bar.get_width() / 2, value / 1.08),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="#d62728",
            )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.06,
            f"{value:.1f}M",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#202020",
        )

    for family, start, end in family_groups:
        center = (start + end) / 2.0
        ax.text(
            center,
            -0.18,
            FAMILY_DISPLAY[family],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=FAMILY_COLORS[family],
            clip_on=False,
        )

    boundaries = [group[2] + 0.5 for group in family_groups[:-1]]
    for boundary in boundaries:
        ax.axvline(boundary, color="#dddddd", linewidth=1.0, zorder=0)

    ax.set_ylim(bottom, max_value * 1.35)
    ax.set_xlim(min(x_positions) - 0.8, max(x_positions) + 0.8)

    fig.suptitle("", y=0.985, fontsize=14)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parameter count chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot parameter counts for the main ViT-family variants.")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used to scan cached summaries and write the plot",
    )
    parser.add_argument(
        "--output",
        default="parameter_count_comparison.png",
        help="Output image path",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=8,
        help="Number of output classes used for parameter counting",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root)
    output_path = Path(args.output)

    series = build_series(workspace_root, num_classes=args.num_classes)
    plot_parameter_counts(series, output_path)


if __name__ == "__main__":
    main()
