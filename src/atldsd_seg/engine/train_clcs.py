"""Train CLCS-Net on ATLDSD VOC-style segmentation data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from atldsd_seg.constants import CLASS_NAMES, NUM_CLASSES
from atldsd_seg.losses.compositional import CompositionalSegmentationLoss
from atldsd_seg.metrics.segmentation import compute_confusion_matrix, summarize_metrics
from atldsd_seg.models.clcs_deeplabv3plus import CLCSDeepLabV3Plus
from atldsd_seg.paths import DEEPLABV3PLUS_ROOT

if str(DEEPLABV3PLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPLABV3PLUS_ROOT))

from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr  # noqa: E402
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate  # noqa: E402


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLCS-Net on ATLDSD.")
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--vocdevkit-path", type=str, default=r"D:\dataset\ATLDSD\VOCdevkit")
    parser.add_argument("--backbone", type=str, default="efficientnet_b4")
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--downsample-factor", type=int, default=16, choices=[8, 16])
    parser.add_argument("--input-shape", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--init-epoch", type=int, default=0)
    parser.add_argument("--freeze-epoch", type=int, default=50)
    parser.add_argument("--unfreeze-epoch", type=int, default=150)
    parser.add_argument("--freeze-batch-size", type=int, default=4)
    parser.add_argument("--unfreeze-batch-size", type=int, default=2)
    parser.add_argument("--freeze-train", type=str2bool, default=True)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay-type", type=str, default="cos", choices=["cos", "step"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--eval-period", type=int, default=10)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--leaf-weight", type=float, default=0.4)
    parser.add_argument("--lesion-weight", type=float, default=0.8)
    parser.add_argument("--disease-weight", type=float, default=0.6)
    parser.add_argument("--use-boundary-head", type=str2bool, default=False)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--boundary-pos-weight", type=float, default=5.0)
    parser.add_argument(
        "--disease-class-weights",
        nargs=4,
        type=float,
        default=None,
        metavar=("RUST", "ALTERNARIA", "GRAY", "BROWN"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def read_split(vocdevkit_path: Path, split: str) -> list[str]:
    split_file = vocdevkit_path / "VOC2007" / "ImageSets" / "Segmentation" / f"{split}.txt"
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def freeze_backbone(model: CLCSDeepLabV3Plus, frozen: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = not frozen


def make_loader(
    ids: list[str],
    input_shape: list[int],
    vocdevkit_path: Path,
    train: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = DeeplabDataset(ids, input_shape, NUM_CLASSES, train=train, dataset_path=str(vocdevkit_path))
    return DataLoader(
        dataset,
        shuffle=train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=deeplab_dataset_collate,
    )


def evaluate(
    model: CLCSDeepLabV3Plus,
    loader: DataLoader,
    criterion: CompositionalSegmentationLoss,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks, _labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(imgs)
            loss_dict = criterion(outputs, masks)
            losses.append(float(loss_dict["loss"].item()))
            pred = outputs["final_prob"].argmax(dim=1).detach().cpu().numpy()
            target = masks.detach().cpu().numpy()
            for pred_i, target_i in zip(pred, target):
                hist += compute_confusion_matrix(pred_i, target_i, NUM_CLASSES)
    return float(np.mean(losses)) if losses else 0.0, hist


def write_metrics(report_dir: Path, epoch: int, val_loss: float, hist: np.ndarray) -> dict[str, float]:
    metrics = summarize_metrics(hist)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "epoch": epoch,
        "val_loss": val_loss,
        "pixel_accuracy": metrics.pixel_accuracy,
        "miou_all": metrics.mean_iou_all,
        "mdice_all": metrics.mean_dice_all,
        "miou_foreground": metrics.mean_iou_foreground,
        "mdice_foreground": metrics.mean_dice_foreground,
    }
    (report_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (report_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("class_id,class_name,iou,dice\n")
        for idx, name in enumerate(CLASS_NAMES):
            f.write(f"{idx},{name},{metrics.per_class_iou[idx]},{metrics.per_class_dice[idx]}\n")
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    report_root = save_dir.parent / "reports"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    vocdevkit_path = Path(args.vocdevkit_path)
    train_ids = read_split(vocdevkit_path, "train")
    val_ids = read_split(vocdevkit_path, "val")

    model = CLCSDeepLabV3Plus(
        backbone=args.backbone,
        pretrained=args.pretrained,
        downsample_factor=args.downsample_factor,
        use_boundary_head=args.use_boundary_head,
    ).to(device)
    criterion = CompositionalSegmentationLoss(
        leaf_weight=args.leaf_weight,
        lesion_weight=args.lesion_weight,
        disease_weight=args.disease_weight,
        boundary_weight=args.boundary_weight,
        boundary_pos_weight=args.boundary_pos_weight,
        disease_class_weights=args.disease_class_weights,
    )

    best_miou = -1.0
    best_val_loss = float("inf")
    batch_size = args.freeze_batch_size if args.freeze_train and args.init_epoch < args.freeze_epoch else args.unfreeze_batch_size
    freeze_backbone(model, args.freeze_train and args.init_epoch < args.freeze_epoch)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.init_lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    min_lr = args.min_lr if args.min_lr is not None else args.init_lr * 0.01
    lr_scheduler = get_lr_scheduler(args.lr_decay_type, args.init_lr, min_lr, args.unfreeze_epoch)

    train_loader = make_loader(train_ids, args.input_shape, vocdevkit_path, True, batch_size, args.num_workers)
    val_loader = make_loader(val_ids, args.input_shape, vocdevkit_path, False, batch_size, args.num_workers)

    print("CLCS-Net Ours-A training")
    print(f"Backbone: {args.backbone}")
    print(f"Boundary head: {args.use_boundary_head} | Boundary weight: {args.boundary_weight}")
    print(f"Disease class weights: {args.disease_class_weights}")
    print(f"Device: {device}")
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")
    print(f"Input shape: {args.input_shape} | Epochs: {args.unfreeze_epoch}")
    print(f"Save dir: {save_dir}")

    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        if args.freeze_train and epoch == args.freeze_epoch:
            freeze_backbone(model, False)
            batch_size = args.unfreeze_batch_size
            train_loader = make_loader(train_ids, args.input_shape, vocdevkit_path, True, batch_size, args.num_workers)
            val_loader = make_loader(val_ids, args.input_shape, vocdevkit_path, False, batch_size, args.num_workers)
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.init_lr,
                momentum=args.momentum,
                nesterov=True,
                weight_decay=args.weight_decay,
            )

        set_optimizer_lr(optimizer, lr_scheduler, epoch)
        model.train()
        total_loss = 0.0
        total_final = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.unfreeze_epoch}", mininterval=0.3)
        for step, (imgs, masks, _labels) in enumerate(progress, start=1):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss_dict = criterion(outputs, masks)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_final += float(loss_dict["final_loss"].item())
            progress.set_postfix(
                loss=total_loss / step,
                final=total_final / step,
                lr=optimizer.param_groups[0]["lr"],
            )

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, hist = evaluate(model, val_loader, criterion, device)
        metrics = summarize_metrics(hist)

        print("Finish Validation")
        if (epoch + 1) % args.eval_period == 0 or epoch + 1 == args.unfreeze_epoch:
            print("Get miou.")
            print("Calculate miou.")
            print(f"Num classes {NUM_CLASSES}")
            print(
                f"===> mIoU: {metrics.mean_iou_all * 100:.2f}; "
                f"mPA: {np.nanmean(np.diag(hist) / np.maximum(hist.sum(axis=1), 1)) * 100:.2f}; "
                f"Accuracy: {metrics.pixel_accuracy * 100:.2f}"
            )
            print("Get miou done.")
            write_metrics(report_root / f"ep{epoch + 1:03d}_val", epoch + 1, val_loss, hist)

        print(f"Epoch:{epoch + 1}/{args.unfreeze_epoch}")
        print(f"Total Loss: {train_loss:.3f} || Val Loss: {val_loss:.3f} ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_loss_epoch_weights.pth")
        if metrics.mean_iou_all > best_miou:
            best_miou = metrics.mean_iou_all
            torch.save(model.state_dict(), save_dir / "best_miou_epoch_weights.pth")
        if (epoch + 1) % args.save_period == 0 or epoch + 1 == args.unfreeze_epoch:
            torch.save(model.state_dict(), save_dir / f"ep{epoch + 1:03d}-loss{train_loss:.3f}-val_loss{val_loss:.3f}.pth")
        torch.save(model.state_dict(), save_dir / "last_epoch_weights.pth")

    print(f"Training completed. Best mIoU: {best_miou * 100:.2f}")


if __name__ == "__main__":
    main()
