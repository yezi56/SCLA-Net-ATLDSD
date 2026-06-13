import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Compute class weights for VOC-style segmentation masks.")
    parser.add_argument("--vocdevkit-path", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--min-weight", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=4.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    split_file = args.vocdevkit_path / "VOC2007" / "ImageSets" / "Segmentation" / f"{args.split}.txt"
    mask_dir = args.vocdevkit_path / "VOC2007" / "SegmentationClass"
    image_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = np.zeros(args.num_classes, dtype=np.float64)

    for image_id in image_ids:
        mask = np.array(Image.open(mask_dir / f"{image_id}.png"))
        valid = (mask >= 0) & (mask < args.num_classes) & (mask != args.ignore_index)
        hist = np.bincount(mask[valid].reshape(-1), minlength=args.num_classes)
        counts += hist[: args.num_classes]

    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(freq + 1e-6)
    weights = weights / weights.mean()
    weights = np.clip(weights, args.min_weight, args.max_weight)

    result = {
        "split": args.split,
        "num_classes": args.num_classes,
        "counts": counts.astype(int).tolist(),
        "frequency": freq.tolist(),
        "weights": weights.tolist(),
        "cls_weights_arg": " ".join(f"{value:.6f}" for value in weights),
    }

    print("cls_weights:", result["cls_weights_arg"])
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
