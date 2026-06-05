import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_CLASSES = ["background", "Bacterialblight", "Blast", "Brownspot", "Tungro"]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_attention_type(value):
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"", "none", "identity"}:
        return ""
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RiceSeg segmentation predictions, metrics, confusion matrix, and complexity files."
    )
    parser.add_argument(
        "--model-adapter",
        type=str,
        default="deeplabv3plus",
        choices=["deeplabv3plus", "unet"],
        help="Model family adapter. More adapters can be added for U-Net, PSPNet, HRNet and SegNeXt.",
    )
    parser.add_argument("--model-root", type=Path, default=Path(r"D:\Code\all\src\models\deeplabv3plus"))
    parser.add_argument("--vocdevkit-path", type=Path, default=Path(r"D:\dataset\RiceSeg\RiceSegdevkit"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            r"D:\Code\all\outputs\riceseg\deeplabv3plus_baseline\weights\best_epoch_weights.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\Code\all\outputs\riceseg\deeplabv3plus_baseline\report_source"
        ),
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--backbone", type=str, default="mobilenet")
    parser.add_argument("--attention-type", type=str, default="none")
    parser.add_argument("--attention-low-type", type=str, default=None)
    parser.add_argument("--attention-high-type", type=str, default=None)
    parser.add_argument("--attention-aspp-type", type=str, default=None)
    parser.add_argument("--attention-decoder-type", type=str, default=None)
    parser.add_argument("--decoder-conv-type", type=str, default="standard", choices=["standard", "pconv", "repconv"])
    parser.add_argument("--use-ppm", type=str2bool, default=False)
    parser.add_argument("--component-aux", type=str2bool, default=False)
    parser.add_argument("--lesion-boundary-sharpen", type=str2bool, default=False)
    parser.add_argument("--lesion-boundary-sharpen-alpha", type=float, default=0.25)
    parser.add_argument("--downsample-factor", type=int, default=16)
    parser.add_argument("--input-shape", nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--fps-interval", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=None, help="Optional quick-check limit.")
    parser.add_argument("--leaf-class-id", type=int, default=1)
    parser.add_argument(
        "--lesion-class-ids",
        nargs="+",
        type=int,
        default=None,
        help="Class ids counted as disease lesion pixels. Defaults to classes 2..num_classes-1.",
    )
    parser.add_argument(
        "--severity-thresholds",
        nargs=2,
        type=float,
        default=[0.05, 0.20],
        metavar=("LOW_MEDIUM", "MEDIUM_HIGH"),
    )
    return parser.parse_args()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_image_ids(vocdevkit_path, split):
    split_file = vocdevkit_path / "VOC2007" / "ImageSets" / "Segmentation" / f"{split}.txt"
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def fast_hist(label, pred, num_classes):
    valid = (label >= 0) & (label < num_classes)
    return np.bincount(
        num_classes * label[valid].astype(int) + pred[valid].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def safe_div(num, den):
    return float(num / den) if den != 0 else 0.0


def compute_metrics_from_hist(hist, class_names):
    tp = np.diag(hist).astype(np.float64)
    gt = hist.sum(axis=1).astype(np.float64)
    pred = hist.sum(axis=0).astype(np.float64)
    fp = pred - tp
    fn = gt - tp

    iou = np.divide(tp, gt + pred - tp, out=np.zeros_like(tp), where=(gt + pred - tp) != 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred != 0)
    recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    rows = []
    for idx, name in enumerate(class_names):
        rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "iou": float(iou[idx]),
                "dice": float(dice[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "gt_pixels": int(gt[idx]),
                "pred_pixels": int(pred[idx]),
                "true_positive_pixels": int(tp[idx]),
            }
        )

    fg_slice = slice(1, len(class_names))
    summary = {
        "pixel_accuracy": safe_div(float(tp.sum()), float(hist.sum())),
        "miou_all": float(np.mean(iou)),
        "mdice_all": float(np.mean(dice)),
        "mprecision_all": float(np.mean(precision)),
        "mrecall_all": float(np.mean(recall)),
        "mf1_all": float(np.mean(f1)),
        "miou_foreground": float(np.mean(iou[fg_slice])),
        "mdice_foreground": float(np.mean(dice[fg_slice])),
        "mprecision_foreground": float(np.mean(precision[fg_slice])),
        "mrecall_foreground": float(np.mean(recall[fg_slice])),
        "mf1_foreground": float(np.mean(f1[fg_slice])),
    }
    return summary, rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path, hist, class_names):
    rows = []
    for idx, name in enumerate(class_names):
        row = {"class_name": name}
        for j, pred_name in enumerate(class_names):
            row[f"pred_{pred_name}"] = int(hist[idx, j])
        rows.append(row)
    write_csv(path, rows, ["class_name"] + [f"pred_{name}" for name in class_names])


def mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def root_mean_square(values):
    return float(math.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def pearson_corr(x, y):
    if len(x) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_corr(x, y):
    if len(x) < 2:
        return 0.0
    return pearson_corr(rankdata(x), rankdata(y))


def severity_grade(value, thresholds):
    low_medium, medium_high = thresholds
    if value < low_medium:
        return "low"
    if value < medium_high:
        return "medium"
    return "high"


def compute_severity_metrics(gt_dir, pred_dir, image_ids, leaf_class_id, lesion_class_ids, thresholds):
    grade_names = ["low", "medium", "high"]
    confusion = {gt_name: {pred_name: 0 for pred_name in grade_names} for gt_name in grade_names}
    rows = []
    gt_values = []
    pred_values = []
    abs_errors = []
    sq_errors = []
    correct_grades = 0
    valid_count = 0

    for image_id in tqdm(image_ids, desc="Compute severity metrics"):
        gt = np.array(Image.open(gt_dir / f"{image_id}.png"))
        pred = np.array(Image.open(pred_dir / f"{image_id}.png"))
        gt_leaf = (gt == leaf_class_id) | np.isin(gt, lesion_class_ids)
        pred_leaf = (pred == leaf_class_id) | np.isin(pred, lesion_class_ids)
        gt_lesion = np.isin(gt, lesion_class_ids)
        pred_lesion = np.isin(pred, lesion_class_ids)

        gt_leaf_pixels = int(gt_leaf.sum())
        pred_leaf_pixels = int(pred_leaf.sum())
        gt_lesion_pixels = int(gt_lesion.sum())
        pred_lesion_pixels = int(pred_lesion.sum())
        if gt_leaf_pixels == 0:
            continue

        gt_severity = safe_div(gt_lesion_pixels, gt_leaf_pixels)
        pred_severity = safe_div(pred_lesion_pixels, pred_leaf_pixels)
        abs_error = abs(pred_severity - gt_severity)
        sq_error = (pred_severity - gt_severity) ** 2
        gt_grade = severity_grade(gt_severity, thresholds)
        pred_grade = severity_grade(pred_severity, thresholds)
        correct = int(gt_grade == pred_grade)

        gt_values.append(gt_severity)
        pred_values.append(pred_severity)
        abs_errors.append(abs_error)
        sq_errors.append(sq_error)
        correct_grades += correct
        valid_count += 1
        confusion[gt_grade][pred_grade] += 1
        rows.append(
            {
                "image_id": image_id,
                "gt_leaf_pixels": gt_leaf_pixels,
                "pred_leaf_pixels": pred_leaf_pixels,
                "gt_lesion_pixels": gt_lesion_pixels,
                "pred_lesion_pixels": pred_lesion_pixels,
                "gt_severity": gt_severity,
                "pred_severity": pred_severity,
                "abs_error": abs_error,
                "squared_error": sq_error,
                "gt_grade": gt_grade,
                "pred_grade": pred_grade,
                "grade_correct": correct,
            }
        )

    summary = {
        "severity_mae": mean(abs_errors),
        "severity_rmse": float(math.sqrt(np.mean(sq_errors))) if len(sq_errors) else 0.0,
        "severity_pearson": pearson_corr(gt_values, pred_values),
        "severity_spearman": spearman_corr(gt_values, pred_values),
        "severity_grade_accuracy": safe_div(correct_grades, valid_count),
        "severity_thresholds": {
            "low_medium": thresholds[0],
            "medium_high": thresholds[1],
        },
        "valid_images": valid_count,
        "lesion_class_ids": list(lesion_class_ids),
        "leaf_class_id": leaf_class_id,
    }
    confusion_rows = []
    for gt_name in grade_names:
        row = {"gt_grade": gt_name}
        for pred_name in grade_names:
            row[f"pred_{pred_name}"] = confusion[gt_name][pred_name]
        confusion_rows.append(row)
    return summary, rows, confusion_rows


def load_model_api(args):
    model_root = args.model_root.resolve()
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    if args.model_adapter == "deeplabv3plus":
        from deeplab import DeeplabV3
        from nets.deeplabv3_plus import DeepLab

        return DeeplabV3, DeepLab
    if args.model_adapter == "unet":
        from unet import Unet as UnetPredictor
        from nets.unet import Unet as UnetModel

        return UnetPredictor, UnetModel
    raise ValueError(f"Unsupported model adapter: {args.model_adapter}")


def build_predictor(args):
    Predictor, _ = load_model_api(args)
    attention_type = "" if args.attention_type.lower() in {"none", "identity"} else args.attention_type
    attention_low_type = normalize_attention_type(args.attention_low_type)
    attention_high_type = normalize_attention_type(args.attention_high_type)
    attention_aspp_type = normalize_attention_type(args.attention_aspp_type)
    attention_decoder_type = normalize_attention_type(args.attention_decoder_type)
    kwargs = {
        "model_path": str(args.model_path),
        "num_classes": args.num_classes,
        "backbone": args.backbone,
        "attention_type": attention_type,
        "input_shape": args.input_shape,
        "cuda": args.cuda and torch.cuda.is_available(),
    }
    if args.model_adapter == "deeplabv3plus":
        kwargs.update(
            {
                "use_ppm": args.use_ppm,
                "downsample_factor": args.downsample_factor,
                "attention_low_type": attention_low_type,
                "attention_high_type": attention_high_type,
                "attention_aspp_type": attention_aspp_type,
                "attention_decoder_type": attention_decoder_type,
                "decoder_conv_type": args.decoder_conv_type,
                "component_aux": args.component_aux,
                "lesion_boundary_sharpen": args.lesion_boundary_sharpen,
                "lesion_boundary_sharpen_alpha": args.lesion_boundary_sharpen_alpha,
            }
        )
    return Predictor(**kwargs)


def export_predictions(args, image_ids, pred_dir):
    pred_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor(args)

    jpeg_dir = args.vocdevkit_path / "VOC2007" / "JPEGImages"
    for image_id in tqdm(image_ids, desc="Export predictions"):
        image_path = jpeg_dir / f"{image_id}.jpg"
        image = Image.open(image_path)
        pred = predictor.get_miou_png(image)
        pred.save(pred_dir / f"{image_id}.png")
    return predictor


def compute_complexity(args, sample_image_path):
    _, Model = load_model_api(args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    attention_type = "" if args.attention_type.lower() in {"none", "identity"} else args.attention_type
    attention_low_type = normalize_attention_type(args.attention_low_type)
    attention_high_type = normalize_attention_type(args.attention_high_type)
    attention_aspp_type = normalize_attention_type(args.attention_aspp_type)
    attention_decoder_type = normalize_attention_type(args.attention_decoder_type)
    if args.model_adapter == "deeplabv3plus":
        model = Model(
            num_classes=args.num_classes,
            backbone=args.backbone,
            downsample_factor=args.downsample_factor,
            pretrained=False,
            attention_type=attention_type,
            attention_low_type=attention_low_type,
            attention_high_type=attention_high_type,
            attention_aspp_type=attention_aspp_type,
            attention_decoder_type=attention_decoder_type,
            decoder_conv_type=args.decoder_conv_type,
            use_ppm=args.use_ppm,
            use_component_aux=args.component_aux,
            use_lbsb=args.lesion_boundary_sharpen,
            lbsb_alpha=args.lesion_boundary_sharpen_alpha,
        ).to(device)
    else:
        model = Model(
            num_classes=args.num_classes,
            backbone=args.backbone,
            pretrained=False,
            attention_type=attention_type,
        ).to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(1, 3, args.input_shape[0], args.input_shape[1]).to(device)
    macs = None
    flops_estimate = None
    thop_error = None
    try:
        from thop import profile

        macs, _ = profile(model, (dummy,), verbose=False)
        flops_estimate = int(macs * 2)
        macs = int(macs)
    except Exception as exc:  # noqa: BLE001
        thop_error = str(exc)

    predictor = build_predictor(args)
    sample_image = Image.open(sample_image_path)
    start = time.perf_counter()
    time_per_image = predictor.get_FPS(sample_image, args.fps_interval)
    elapsed = time.perf_counter() - start

    return {
        "params": int(params),
        "trainable_params": int(trainable_params),
        "macs": macs,
        "flops_estimate_2x_macs": flops_estimate,
        "thop_error": thop_error,
        "fps_interval": args.fps_interval,
        "time_per_image_seconds": float(time_per_image),
        "fps": float(1.0 / time_per_image) if time_per_image > 0 else 0.0,
        "fps_benchmark_elapsed_seconds": float(elapsed),
        "device": str(device),
    }


def main():
    args = parse_args()
    if len(args.class_names) != args.num_classes:
        raise ValueError("--class-names length must match --num-classes")
    if args.lesion_class_ids is None:
        args.lesion_class_ids = list(range(2, args.num_classes))
    invalid_lesion_ids = [idx for idx in args.lesion_class_ids if idx < 0 or idx >= args.num_classes]
    if invalid_lesion_ids:
        raise ValueError(f"--lesion-class-ids contains invalid ids: {invalid_lesion_ids}")
    if args.leaf_class_id < 0 or args.leaf_class_id >= args.num_classes:
        raise ValueError("--leaf-class-id must be in [0, num_classes)")
    if args.severity_thresholds[0] >= args.severity_thresholds[1]:
        raise ValueError("--severity-thresholds must be increasing")

    output_dir = args.output_dir
    pred_dir = output_dir / "pred_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = read_image_ids(args.vocdevkit_path, args.split)
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    run_config = {
        "model_root": str(args.model_root),
        "vocdevkit_path": str(args.vocdevkit_path),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "num_classes": args.num_classes,
        "class_names": args.class_names,
        "backbone": args.backbone,
        "attention_type": args.attention_type,
        "attention_low_type": args.attention_low_type,
        "attention_high_type": args.attention_high_type,
        "attention_aspp_type": args.attention_aspp_type,
        "attention_decoder_type": args.attention_decoder_type,
        "decoder_conv_type": args.decoder_conv_type,
        "use_ppm": args.use_ppm,
        "component_aux": args.component_aux,
        "lesion_boundary_sharpen": args.lesion_boundary_sharpen,
        "lesion_boundary_sharpen_alpha": args.lesion_boundary_sharpen_alpha,
        "downsample_factor": args.downsample_factor,
        "input_shape": args.input_shape,
        "cuda": args.cuda,
        "image_count": len(image_ids),
        "leaf_class_id": args.leaf_class_id,
        "lesion_class_ids": args.lesion_class_ids,
        "severity_thresholds": args.severity_thresholds,
    }
    write_json(output_dir / "run_config.json", run_config)

    if not args.skip_predict:
        export_predictions(args, image_ids, pred_dir)

    gt_dir = args.vocdevkit_path / "VOC2007" / "SegmentationClass"
    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

    for image_id in tqdm(image_ids, desc="Compute segmentation metrics"):
        gt = np.array(Image.open(gt_dir / f"{image_id}.png"))
        pred = np.array(Image.open(pred_dir / f"{image_id}.png"))
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {image_id}: gt={gt.shape}, pred={pred.shape}")
        hist += fast_hist(gt.flatten(), pred.flatten(), args.num_classes)

    summary, per_class_rows = compute_metrics_from_hist(hist, args.class_names)
    summary.update(
        {
            "split": args.split,
            "num_images": len(image_ids),
            "model_path": str(args.model_path),
        }
    )
    write_json(output_dir / "metrics_summary.json", summary)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows, list(per_class_rows[0].keys()))
    write_confusion_matrix(output_dir / "confusion_matrix.csv", hist, args.class_names)

    severity_summary, severity_rows, severity_confusion_rows = compute_severity_metrics(
        gt_dir,
        pred_dir,
        image_ids,
        args.leaf_class_id,
        args.lesion_class_ids,
        args.severity_thresholds,
    )
    write_json(output_dir / "severity_metrics.json", severity_summary)
    write_csv(output_dir / "severity_per_image.csv", severity_rows, list(severity_rows[0].keys()) if severity_rows else ["image_id"])
    write_csv(
        output_dir / "severity_confusion_matrix.csv",
        severity_confusion_rows,
        list(severity_confusion_rows[0].keys()),
    )

    sample_image = args.vocdevkit_path / "VOC2007" / "JPEGImages" / f"{image_ids[0]}.jpg"
    complexity = compute_complexity(args, sample_image)
    write_json(output_dir / "complexity.json", complexity)

    print(f"Saved report to: {output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(severity_summary, ensure_ascii=False, indent=2))
    print(json.dumps(complexity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
