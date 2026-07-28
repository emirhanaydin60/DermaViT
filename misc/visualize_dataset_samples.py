"""Create a publication-style figure with class samples and class counts.

The script expects a folder layout like:

    data_dir/
        train/
            MEL/
            NV/
            BCC/
            ...

It renders one example per class together with a horizontal class-count chart
in a single PNG figure. The sample panel is labeled a) and the count chart is
labeled b).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec

DEFAULT_DATA_DIR = r"C:/Users/emirh/Desktop/Projects/datasets/input_sk"

CLASS_NAME_MAP = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
}

GRID_ORDER = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
PRIMARY_COLOR = "#0b2d5b"
CLASS_SAMPLE_PATHS = {
    "MEL": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\MEL\\ISIC_0000463_downsampled.jpg",
    "NV": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\NV\\ISIC_0000128_downsampled.jpg",
    "BCC": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\BCC\\ISIC_0024432.jpg",
    "AK": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\AK\\ISIC_0027668.jpg",
    "BKL": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\BKL\\ISIC_0014586_downsampled.jpg",
    "DF": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\DF\\ISIC_0027118.jpg",
    "VASC": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\VASC\\ISIC_0061611.jpg",
    "SCC": "C:\\Users\\emirh\\Desktop\\Projects\\datasets\\input_sk\\train\\SCC\\ISIC_0027343.jpg",
}
FIXED_CLASS_COUNTS = {
    "NV": 12875,
    "MEL": 4522,
    "BCC": 3323,
    "BKL": 2624,
    "AK": 867,
    "SCC": 628,
    "VASC": 253,
    "DF": 239,
}


def resolve_split_dir(data_dir: Path, split: str) -> Path:
    split_dir = data_dir / split
    if split_dir.is_dir():
        return split_dir
    if split == "train" and data_dir.is_dir():
        return data_dir
    raise FileNotFoundError(f"Split folder not found: {split_dir}")


def get_class_counts(dataset) -> dict[str, int]:
    class_counts = {class_name: 0 for class_name in dataset.classes}
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [label for _, label in dataset.samples]

    for label in targets:
        class_name = dataset.classes[int(label)]
        class_counts[class_name] += 1

    return class_counts


def load_image(image_path: Path):
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB")


def fit_image_to_square(image, size: int = 512):
    from PIL import Image, ImageOps

    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def resolve_class_sample_path(class_code: str) -> Path | None:
    sample_path = CLASS_SAMPLE_PATHS.get(class_code, "").strip()
    if not sample_path or sample_path == "random/path":
        return None

    path = Path(sample_path)
    return path if path.is_file() else None


def render_class_grid(dataset, out_path: Path) -> None:
    class_names = list(dataset.classes)
    if not class_names:
        raise ValueError("No classes found in the dataset")

    missing = [class_name for class_name in GRID_ORDER if class_name not in class_names]
    if missing:
        raise ValueError(f"Dataset is missing expected classes: {missing}")

    class_counts = {class_code: FIXED_CLASS_COUNTS.get(class_code, 0) for class_code in GRID_ORDER}
    ordered_classes = sorted(
        GRID_ORDER,
        key=lambda class_code: class_counts.get(class_code, 0),
        reverse=True,
    )

    fig = plt.figure(figsize=(20, 10), constrained_layout=False)
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.45, 0.75], wspace=0.10, figure=fig)
    sample_grid = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0], wspace=0.10, hspace=0.30)
    sample_axes = [fig.add_subplot(sample_grid[i, j]) for i in range(2) for j in range(4)]
    count_ax = fig.add_subplot(outer[1])

    fig.patch.set_facecolor("white")
    fig.suptitle("ISIC2019 class samples and class counts", fontsize=20, fontweight="bold", y=0.98, color=PRIMARY_COLOR)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.92, bottom=0.12)

    for idx, class_code in enumerate(ordered_classes):
        ax = sample_axes[idx]
        ax.set_axis_off()
        ax.set_facecolor("#f7f7f7")
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)

        full_name = CLASS_NAME_MAP.get(class_code, class_code)
        sample_path = resolve_class_sample_path(class_code)

        if sample_path is None:
            ax.text(
                0.5,
                0.5,
                class_code,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=PRIMARY_COLOR,
                transform=ax.transAxes,
            )
            ax.text(0.5, 0.06, full_name, ha="center", va="bottom", fontsize=11, transform=ax.transAxes, wrap=True)
            continue

        image = fit_image_to_square(load_image(sample_path), size=512)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)

        for spine in ax.spines.values():
            spine.set_edgecolor("#d0d0d0")
            spine.set_linewidth(1.0)

        ax.text(
            0.5,
            0.98,
            class_code,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="white",
            transform=ax.transAxes,
            bbox={"facecolor": PRIMARY_COLOR, "alpha": 0.75, "pad": 3, "edgecolor": "none"},
        )
        ax.text(
            0.5,
            -0.08,
            full_name,
            ha="center",
            va="top",
            fontsize=11,
            transform=ax.transAxes,
            wrap=True,
        )

    for ax in sample_axes[len(ordered_classes) :]:
        ax.set_axis_off()

    counts = [class_counts[class_code] for class_code in ordered_classes]
    y_positions = list(range(len(ordered_classes)))

    count_ax.barh(y_positions, counts, color=PRIMARY_COLOR, edgecolor=PRIMARY_COLOR, height=0.55)
    count_ax.set_yticks(y_positions)
    count_ax.set_yticklabels(ordered_classes, fontsize=11, fontweight="bold")
    count_ax.invert_yaxis()
    count_ax.set_xlabel("Number of images", fontsize=11)
    count_ax.grid(axis="x", linestyle="--", alpha=0.25)
    count_ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        count_ax.spines[spine].set_visible(False)
    count_ax.spines["left"].set_color("#b0b0b0")
    count_ax.spines["bottom"].set_color("#b0b0b0")

    max_count = max(counts) if counts else 0
    count_ax.set_xlim(0, max_count * 1.15 if max_count else 1)
    for y, count in zip(y_positions, counts):
        count_ax.text(count + max(max_count * 0.015, 1), y, str(count), va="center", ha="left", fontsize=10, color=PRIMARY_COLOR)

    count_ax.set_xlabel("Number of images", fontsize=11)

    sample_panel_left = min(ax.get_position().x0 for ax in sample_axes)
    sample_panel_right = max(ax.get_position().x1 for ax in sample_axes)
    sample_panel_bottom = min(ax.get_position().y0 for ax in sample_axes)
    sample_panel_center = (sample_panel_left + sample_panel_right) / 2
    count_panel = count_ax.get_position()

    fig.text(
        sample_panel_center,
        sample_panel_bottom - 0.03,
        "a)",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=PRIMARY_COLOR,
    )
    fig.text(
        count_panel.x0,
        count_panel.y0 - 0.03,
        "b)",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=PRIMARY_COLOR,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a publication-style dermatology figure with samples and class counts.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Dataset root directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--output", default="dataset_class_samples.png", help="Output PNG path")
    args = parser.parse_args()

    try:
        from torchvision import datasets
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torchvision is required to load the dataset. Install it in the active environment before running this script.") from exc

    data_dir = Path(args.data_dir)
    split_dir = resolve_split_dir(data_dir, args.split)

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    dataset = datasets.ImageFolder(split_dir)
    out_path = Path(args.output)
    render_class_grid(dataset, out_path=out_path)
    print(f"Saved visualization to {out_path.resolve()}")


if __name__ == "__main__":
    main()
