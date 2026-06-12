"""Build a tiny VOCdevkit clone for fast ATLDSD module screening.

The script keeps the original VOCdevkit untouched. It samples IDs from the
existing train/val splits, preserves disease-class coverage when possible, and
copies only the selected images/masks into an ignored output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = [
    "background",
    "leaf",
    "rust",
    "alternaria_leaf_spot",
    "gray_spot",
    "brown_spot",
]
DISEASE_CLASS_IDS = [2, 3, 4, 5]
PREFIX_GROUPS = ["healthy", "rust", "alternaria", "gray", "brown"]


def read_ids(split_file: Path) -> list[str]:
    return [line.strip().split()[0] for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def mask_classes(mask_path: Path, num_classes: int) -> set[int]:
    mask = np.array(Image.open(mask_path), dtype=np.uint8)
    return {int(v) for v in np.unique(mask) if 0 <= int(v) < num_classes}


def collect_metadata(voc2007: Path, ids: list[str], num_classes: int) -> dict[str, set[int]]:
    mask_dir = voc2007 / "SegmentationClass"
    metadata: dict[str, set[int]] = {}
    for image_id in ids:
        mask_path = mask_dir / f"{image_id}.png"
        if mask_path.exists():
            metadata[image_id] = mask_classes(mask_path, num_classes)
    return metadata


def disease_hint_from_name(image_id: str) -> int | None:
    lowered = image_id.lower()
    if lowered.startswith("rust"):
        return 2
    if lowered.startswith("alternaria"):
        return 3
    if lowered.startswith("gray"):
        return 4
    if lowered.startswith("brown"):
        return 5
    return None


def prefix_group(image_id: str) -> str | None:
    lowered = image_id.lower()
    for group in PREFIX_GROUPS:
        if lowered.startswith(group):
            return group
    return None


def parse_prefix_weights(raw: str) -> dict[str, float]:
    weights = {group: 1.0 for group in PREFIX_GROUPS}
    if not raw:
        return weights
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid prefix weight '{item}'. Expected group=value.")
        group, value = item.split("=", 1)
        group = group.strip().lower()
        if group not in weights:
            raise ValueError(f"Unknown prefix group '{group}'. Expected one of {PREFIX_GROUPS}.")
        weights[group] = max(float(value), 0.0)
    return weights


def weighted_prefix_quotas(count: int, weights: dict[str, float]) -> dict[str, int]:
    total = sum(weights.get(group, 1.0) for group in PREFIX_GROUPS)
    if total <= 0:
        return {group: 0 for group in PREFIX_GROUPS}
    raw = {group: count * weights.get(group, 1.0) / total for group in PREFIX_GROUPS}
    quotas = {group: int(raw[group]) for group in PREFIX_GROUPS}
    remainder = count - sum(quotas.values())
    order = sorted(PREFIX_GROUPS, key=lambda group: raw[group] - quotas[group], reverse=True)
    for group in order[:remainder]:
        quotas[group] += 1
    return quotas


def choose_prefix_balanced_ids(
    ids: list[str],
    metadata: dict[str, set[int]],
    count: int,
    seed: int,
    prefix_weights: dict[str, float],
) -> list[str]:
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {group: [] for group in PREFIX_GROUPS}
    leftovers: list[str] = []
    for image_id in ids:
        if image_id not in metadata:
            continue
        group = prefix_group(image_id)
        if group is None:
            leftovers.append(image_id)
        else:
            buckets[group].append(image_id)

    for values in buckets.values():
        rng.shuffle(values)
    rng.shuffle(leftovers)

    selected: list[str] = []
    selected_set: set[str] = set()
    quotas = weighted_prefix_quotas(count, prefix_weights)
    for group in PREFIX_GROUPS:
        for image_id in buckets[group][: quotas[group]]:
            selected.append(image_id)
            selected_set.add(image_id)

    remaining = [
        image_id
        for group in PREFIX_GROUPS
        for image_id in buckets[group][quotas[group] :]
        if image_id not in selected_set
    ] + [image_id for image_id in leftovers if image_id not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(count - len(selected), 0)])
    return selected[:count]


def choose_ids(
    ids: list[str],
    metadata: dict[str, set[int]],
    count: int,
    seed: int,
    required_classes: list[int],
    balance_prefixes: bool,
    prefix_weights: dict[str, float],
) -> list[str]:
    if balance_prefixes:
        selected = choose_prefix_balanced_ids(ids, metadata, count, seed, prefix_weights)
        covered = {class_id for image_id in selected for class_id in metadata.get(image_id, set())}
        if all(class_id in covered for class_id in required_classes):
            return selected

    rng = random.Random(seed)
    available = [image_id for image_id in ids if image_id in metadata]
    selected: list[str] = []
    selected_set: set[str] = set()

    for class_id in required_classes:
        candidates = [image_id for image_id in available if class_id in metadata[image_id] and image_id not in selected_set]
        if not candidates:
            candidates = [
                image_id
                for image_id in available
                if disease_hint_from_name(image_id) == class_id and image_id not in selected_set
            ]
        if candidates:
            pick = rng.choice(candidates)
            selected.append(pick)
            selected_set.add(pick)

    remaining = [image_id for image_id in available if image_id not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(count - len(selected), 0)])
    return selected[:count]


def copy_selected_files(source_voc2007: Path, target_voc2007: Path, split_ids: dict[str, list[str]]) -> None:
    image_source = source_voc2007 / "JPEGImages"
    mask_source = source_voc2007 / "SegmentationClass"
    image_target = target_voc2007 / "JPEGImages"
    mask_target = target_voc2007 / "SegmentationClass"
    image_set_target = target_voc2007 / "ImageSets" / "Segmentation"
    image_target.mkdir(parents=True, exist_ok=True)
    mask_target.mkdir(parents=True, exist_ok=True)
    image_set_target.mkdir(parents=True, exist_ok=True)

    all_ids = sorted({image_id for ids in split_ids.values() for image_id in ids})
    for image_id in all_ids:
        shutil.copy2(image_source / f"{image_id}.jpg", image_target / f"{image_id}.jpg")
        shutil.copy2(mask_source / f"{image_id}.png", mask_target / f"{image_id}.png")

    for split, ids in split_ids.items():
        (image_set_target / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


def summarize(split_ids: dict[str, list[str]], metadata: dict[str, set[int]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split, ids in split_ids.items():
        class_counter: Counter[int] = Counter()
        disease_image_counter: Counter[int] = Counter()
        for image_id in ids:
            classes = metadata.get(image_id, set())
            class_counter.update(classes)
            for class_id in DISEASE_CLASS_IDS:
                if class_id in classes:
                    disease_image_counter[class_id] += 1
        summary[split] = {
            "num_images": len(ids),
            "class_presence_images": {CLASS_NAMES[k]: int(v) for k, v in sorted(class_counter.items())},
            "disease_presence_images": {CLASS_NAMES[k]: int(disease_image_counter[k]) for k in DISEASE_CLASS_IDS},
            "ids": ids,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fast ATLDSD VOC subset for module screening.")
    parser.add_argument("--source-vocdevkit", type=Path, default=Path(r"D:\dataset\ATLDSD\VOCdevkit"))
    parser.add_argument("--output-vocdevkit", type=Path, default=Path("outputs/fast_voc/VOCdevkit_fast_64_32"))
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--val-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--no-prefix-balance", action="store_true", help="Disable healthy/disease-prefix balanced sampling.")
    parser.add_argument(
        "--prefix-weights",
        default="",
        help="Comma-separated prefix sampling weights, e.g. gray=2.5,brown=1.5.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing generated subset directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_voc2007 = args.source_vocdevkit / "VOC2007"
    target_vocdevkit = args.output_vocdevkit.resolve()
    target_voc2007 = target_vocdevkit / "VOC2007"

    split_dir = source_voc2007 / "ImageSets" / "Segmentation"
    train_ids = read_ids(split_dir / "train.txt")
    val_ids = read_ids(split_dir / "val.txt")
    metadata = collect_metadata(source_voc2007, train_ids + val_ids, args.num_classes)
    prefix_weights = parse_prefix_weights(args.prefix_weights)

    if target_vocdevkit.exists():
        if not args.force:
            raise FileExistsError(f"{target_vocdevkit} exists. Pass --force to replace it.")
        repo_root = Path.cwd().resolve()
        allowed_root = (repo_root / "outputs").resolve()
        if not target_vocdevkit.is_relative_to(allowed_root):
            raise ValueError(f"Refusing to delete outside outputs/: {target_vocdevkit}")
        shutil.rmtree(target_vocdevkit)

    selected_train = choose_ids(
        train_ids,
        metadata,
        args.train_count,
        args.seed,
        DISEASE_CLASS_IDS,
        balance_prefixes=not args.no_prefix_balance,
        prefix_weights=prefix_weights,
    )
    selected_val = choose_ids(
        val_ids,
        metadata,
        args.val_count,
        args.seed + 1000,
        DISEASE_CLASS_IDS,
        balance_prefixes=not args.no_prefix_balance,
        prefix_weights=prefix_weights,
    )
    split_ids = {
        "train": selected_train,
        "val": selected_val,
        "trainval": selected_train + selected_val,
        "test": selected_val,
    }
    copy_selected_files(source_voc2007, target_voc2007, split_ids)

    for extra_name in ["classes.txt", "README_ATLDSD_VOC.md"]:
        source = source_voc2007 / extra_name
        if source.exists():
            shutil.copy2(source, target_voc2007 / extra_name)

    manifest = {
        "source_vocdevkit": str(args.source_vocdevkit.resolve()),
        "output_vocdevkit": str(target_vocdevkit),
        "seed": args.seed,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "num_classes": args.num_classes,
        "prefix_balance": not args.no_prefix_balance,
        "prefix_weights": prefix_weights,
        "summary": summarize(split_ids, metadata),
    }
    target_voc2007.mkdir(parents=True, exist_ok=True)
    (target_voc2007 / "fast_subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
