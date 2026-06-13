from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


CLASS_ID_TO_NAME = {
    0: "background",
    1: "leaf",
    2: "rust",
    3: "alternaria_leaf_spot",
    4: "gray_spot",
    5: "brown_spot",
}

FOLDER_TO_DISEASE_ID = {
    "Healthy leaf": 0,
    "Rust": 2,
    "Alternaria leaf spot": 3,
    "Gray spot": 4,
    "Brown spot": 5,
}

DISEASE_IDS = {2, 3, 4, 5}

PALETTE_COLORS = {
    0: (0, 0, 0),
    1: (0, 160, 80),
    2: (245, 130, 48),
    3: (230, 25, 75),
    4: (145, 30, 180),
    5: (160, 82, 45),
}


@dataclass(frozen=True)
class SourceSample:
    category: str
    image_path: Path
    label_path: Path


def build_palette() -> list[int]:
    palette = [0] * (256 * 3)
    for class_id, color in PALETTE_COLORS.items():
        start = class_id * 3
        palette[start : start + 3] = list(color)
    return palette


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "sample"


def find_image_by_stem(image_dir: Path, stem: str) -> Path | None:
    matches = [
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.stem.lower() == stem.lower()
        and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Multiple images match label stem {stem!r}: {matches}")
    return matches[0]


def collect_samples(source_root: Path) -> list[SourceSample]:
    samples: list[SourceSample] = []
    missing_images: list[Path] = []

    for category_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        image_dir = category_dir / "image"
        label_dir = category_dir / "label"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue

        for label_path in sorted(label_dir.glob("*.png")):
            image_path = find_image_by_stem(image_dir, label_path.stem)
            if image_path is None:
                missing_images.append(label_path)
                continue
            samples.append(
                SourceSample(
                    category=category_dir.name,
                    image_path=image_path,
                    label_path=label_path,
                )
            )

    if missing_images:
        preview = "\n".join(str(path) for path in missing_images[:20])
        raise RuntimeError(
            f"Found {len(missing_images)} labels without matching JPEG images:\n{preview}"
        )

    if not samples:
        raise RuntimeError(f"No image/label samples found under {source_root}")

    return samples


def validate_mask(mask: Image.Image, sample: SourceSample) -> tuple[dict[int, int], int, int]:
    histogram = mask.histogram()
    counts = {
        class_id: histogram[class_id]
        for class_id in range(min(len(histogram), 256))
        if histogram[class_id] > 0
    }

    unknown_ids = sorted(set(counts) - set(CLASS_ID_TO_NAME))
    if unknown_ids:
        raise RuntimeError(
            f"Unexpected class ids {unknown_ids} in {sample.label_path}; "
            f"known ids are {sorted(CLASS_ID_TO_NAME)}"
        )

    expected_disease_id = FOLDER_TO_DISEASE_ID.get(sample.category)
    allowed_ids = {0, 1}
    if expected_disease_id is None:
        raise RuntimeError(
            f"Unknown category folder {sample.category!r}; add it to FOLDER_TO_DISEASE_ID"
        )
    if expected_disease_id:
        allowed_ids.add(expected_disease_id)

    unexpected_for_folder = sorted(set(counts) - allowed_ids)
    if unexpected_for_folder:
        raise RuntimeError(
            f"Mask {sample.label_path} in category {sample.category!r} contains "
            f"class ids {unexpected_for_folder}, expected only {sorted(allowed_ids)}"
        )

    leaf_pixels = sum(count for class_id, count in counts.items() if class_id > 0)
    lesion_pixels = sum(count for class_id, count in counts.items() if class_id in DISEASE_IDS)
    return counts, leaf_pixels, lesion_pixels


def severity_from_ratio(ratio: float) -> tuple[int, str]:
    if ratio <= 0:
        return 0, "healthy"
    if ratio <= 0.05:
        return 1, "slight"
    if ratio <= 0.15:
        return 2, "moderate"
    if ratio <= 0.30:
        return 3, "severe"
    return 4, "very_severe"


def stratified_split(
    image_ids_by_category: dict[str, list[str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for category, image_ids in sorted(image_ids_by_category.items()):
        shuffled = list(image_ids)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = round(n_total * train_ratio)
        n_val = round(n_total * val_ratio)
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    for split_ids in splits.values():
        split_ids.sort()
    return splits


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_voc(
    source_root: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    overwrite: bool,
) -> None:
    voc_root = output_root / "VOCdevkit" / "VOC2012"
    if voc_root.exists() and any(voc_root.rglob("*")):
        if not overwrite:
            raise RuntimeError(
                f"Output already exists: {voc_root}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(voc_root)

    jpeg_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    split_dir = voc_root / "ImageSets" / "Segmentation"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(source_root)
    palette = build_palette()
    used_ids: set[str] = set()
    rows: list[dict[str, str | int | float]] = []
    image_ids_by_category: dict[str, list[str]] = defaultdict(list)
    class_pixel_totals = {class_id: 0 for class_id in CLASS_ID_TO_NAME}

    for sample in samples:
        category_slug = slugify(sample.category)
        stem_slug = slugify(sample.image_path.stem)
        image_id = f"{category_slug}_{stem_slug}"
        suffix = 2
        while image_id in used_ids:
            image_id = f"{category_slug}_{stem_slug}_{suffix}"
            suffix += 1
        used_ids.add(image_id)

        with Image.open(sample.image_path) as image, Image.open(sample.label_path) as mask:
            if image.size != mask.size:
                raise RuntimeError(
                    f"Image/mask size mismatch for {sample.image_path}: "
                    f"{image.size} vs {mask.size}"
                )

            mask = mask.convert("P")
            counts, leaf_pixels, lesion_pixels = validate_mask(mask, sample)
            for class_id, count in counts.items():
                class_pixel_totals[class_id] += count

            shutil.copy2(sample.image_path, jpeg_dir / f"{image_id}.jpg")
            mask.putpalette(palette)
            mask.save(mask_dir / f"{image_id}.png", optimize=True)

        lesion_ratio = lesion_pixels / leaf_pixels if leaf_pixels else 0.0
        severity_id, severity_name = severity_from_ratio(lesion_ratio)
        disease_id = FOLDER_TO_DISEASE_ID[sample.category]
        disease_pixels = counts.get(disease_id, 0) if disease_id else 0
        disease_ratio = disease_pixels / leaf_pixels if leaf_pixels else 0.0

        image_ids_by_category[sample.category].append(image_id)
        rows.append(
            {
                "image_id": image_id,
                "category": sample.category,
                "class_id": disease_id,
                "class_name": CLASS_ID_TO_NAME[disease_id],
                "source_image": str(sample.image_path),
                "source_label": str(sample.label_path),
                "voc_image": str(jpeg_dir / f"{image_id}.jpg"),
                "voc_mask": str(mask_dir / f"{image_id}.png"),
                "width": image.size[0],
                "height": image.size[1],
                "leaf_pixels": leaf_pixels,
                "lesion_pixels": lesion_pixels,
                "lesion_ratio": round(lesion_ratio, 8),
                "disease_pixels": disease_pixels,
                "disease_ratio": round(disease_ratio, 8),
                "severity_id": severity_id,
                "severity_name": severity_name,
            }
        )

    splits = stratified_split(image_ids_by_category, train_ratio, val_ratio, seed)
    write_lines(split_dir / "train.txt", splits["train"])
    write_lines(split_dir / "val.txt", splits["val"])
    write_lines(split_dir / "test.txt", splits["test"])
    write_lines(split_dir / "trainval.txt", sorted(splits["train"] + splits["val"]))

    write_lines(
        voc_root / "classes.txt",
        [f"{class_id} {name}" for class_id, name in sorted(CLASS_ID_TO_NAME.items())],
    )

    severity_csv = voc_root / "severity.csv"
    with severity_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_root": str(source_root),
        "voc_root": str(voc_root),
        "sample_count": len(rows),
        "classes": CLASS_ID_TO_NAME,
        "splits": {name: len(ids) for name, ids in splits.items()},
        "trainval_count": len(splits["train"]) + len(splits["val"]),
        "categories": {
            category: len(ids) for category, ids in sorted(image_ids_by_category.items())
        },
        "class_pixel_totals": class_pixel_totals,
        "severity_bins": [
            {"severity_id": 0, "severity_name": "healthy", "lesion_ratio": "0"},
            {"severity_id": 1, "severity_name": "slight", "lesion_ratio": "(0, 0.05]"},
            {"severity_id": 2, "severity_name": "moderate", "lesion_ratio": "(0.05, 0.15]"},
            {"severity_id": 3, "severity_name": "severe", "lesion_ratio": "(0.15, 0.30]"},
            {"severity_id": 4, "severity_name": "very_severe", "lesion_ratio": "(0.30, 1]"},
        ],
    }
    (voc_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""# ATLDSD VOC2012

This directory was generated from `{source_root}`.

## Layout

- `JPEGImages/*.jpg`: RGB leaf images.
- `SegmentationClass/*.png`: paletted single-channel semantic masks.
- `ImageSets/Segmentation/train.txt`: training image ids.
- `ImageSets/Segmentation/val.txt`: validation image ids.
- `ImageSets/Segmentation/test.txt`: test image ids.
- `ImageSets/Segmentation/trainval.txt`: train + validation image ids.
- `classes.txt`: semantic class id mapping.
- `severity.csv`: per-image lesion ratio and severity labels.
- `dataset_summary.json`: counts and split summary.

## Class IDs

{chr(10).join(f"- `{class_id}`: `{name}`" for class_id, name in sorted(CLASS_ID_TO_NAME.items()))}

## Severity

`severity.csv` computes `lesion_ratio = lesion_pixels / leaf_pixels`, where
`leaf_pixels` are mask pixels with class id greater than 0 and `lesion_pixels`
are pixels with disease class ids 2, 3, 4, or 5.

The generated severity bins are:

- `0 healthy`: ratio = 0
- `1 slight`: 0 < ratio <= 0.05
- `2 moderate`: 0.05 < ratio <= 0.15
- `3 severe`: 0.15 < ratio <= 0.30
- `4 very_severe`: ratio > 0.30
"""
    (voc_root / "README_ATLDSD_VOC.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ATLDSD to VOC2012 segmentation format.")
    parser.add_argument("--source", type=Path, default=Path(r"D:\dataset\ATLDSD"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\dataset\ATLDSD"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.train_ratio <= 0 or args.val_ratio < 0:
        parser.error("--train-ratio must be > 0 and --val-ratio must be >= 0")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("--train-ratio + --val-ratio must be < 1 so test split is non-empty")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    prepare_voc(
        source_root=args.source,
        output_root=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
