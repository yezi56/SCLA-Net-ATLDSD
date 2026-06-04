import argparse
import datetime
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.backbone_registry import (
    backbone_has_external_pretrained_url,
    download_backbone_pretrained_weights,
    get_backbone_lr_limits,
    get_backbone_names,
)
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

try:
    from atldsd_seg.paths import DEFAULT_VOCDEVKIT_PATH
except ImportError:
    DEFAULT_VOCDEVKIT_PATH = Path("VOCdevkit")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on VOC-style semantic segmentation datasets.")
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--sync-bn", type=str2bool, default=False)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--backbone", type=str, default="efficientnet_b4", choices=get_backbone_names())
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--downsample-factor", type=int, default=16, choices=[8, 16])
    parser.add_argument("--attention-type", type=str, default="", help="Legacy global attention type for all DeepLabV3+ insertion points.")
    parser.add_argument("--attention-low-type", type=str, default=None, help="Attention for low-level decoder features. Defaults to --attention-type.")
    parser.add_argument("--attention-high-type", type=str, default=None, help="Attention for high-level backbone features. Defaults to --attention-type.")
    parser.add_argument("--attention-aspp-type", type=str, default=None, help="Attention after ASPP. Defaults to --attention-type.")
    parser.add_argument("--attention-decoder-type", type=str, default=None, help="Attention after decoder fusion. Defaults to --attention-type.")
    parser.add_argument("--decoder-conv-type", type=str, default="standard", choices=["standard", "pconv", "repconv"])
    parser.add_argument("--use-ppm", type=str2bool, default=False)
    parser.add_argument("--ppm-bins", nargs="+", type=int, default=[1, 2, 3, 6])
    parser.add_argument("--input-shape", nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--init-epoch", type=int, default=0)
    parser.add_argument("--freeze-epoch", type=int, default=50)
    parser.add_argument("--freeze-batch-size", type=int, default=4)
    parser.add_argument("--unfreeze-epoch", type=int, default=300)
    parser.add_argument("--unfreeze-batch-size", type=int, default=2)
    parser.add_argument("--freeze-train", type=str2bool, default=True)
    parser.add_argument("--init-lr", type=float, default=7e-3)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--optimizer-type", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay-type", type=str, default="cos", choices=["cos", "step"])
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=os.path.join("outputs", "semantic_seg", "weights"))
    parser.add_argument("--log-dir", type=str, default=os.path.join("outputs", "semantic_seg", "logs"))
    parser.add_argument("--eval-flag", type=str2bool, default=True)
    parser.add_argument("--eval-period", type=int, default=10)
    parser.add_argument("--dataset-name", type=str, default="VOC")
    parser.add_argument("--datasets-root", type=str, default=".")
    parser.add_argument("--vocdevkit-path", type=str, default="VOCdevkit")
    parser.add_argument("--dice-loss", type=str2bool, default=False)
    parser.add_argument("--focal-loss", type=str2bool, default=False)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lbft-loss", type=str2bool, default=False, help="Use LBFTLoss: Weighted CE + lambda * Focal Tversky.")
    parser.add_argument("--lbft-lambda", type=float, default=1.0, help="Weight of the Focal Tversky term in LBFTLoss.")
    parser.add_argument("--lbft-alpha", type=float, default=0.3, help="False-positive weight in Focal Tversky.")
    parser.add_argument("--lbft-beta", type=float, default=0.7, help="False-negative weight in Focal Tversky.")
    parser.add_argument("--lbft-gamma", type=float, default=1.33, help="Focusing exponent in Focal Tversky.")
    parser.add_argument("--mix-mode", type=str, default="none", choices=["none", "mixup", "cutmix"])
    parser.add_argument("--mix-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--sclp", type=str2bool, default=False, help="Use severity-controlled lesion copy-paste.")
    parser.add_argument("--sclp-prob", type=float, default=0.0)
    parser.add_argument("--sclp-max-components", type=int, default=3)
    parser.add_argument(
        "--sclp-class-weights",
        nargs=4,
        type=float,
        default=[1.0, 2.0, 2.0, 3.0],
        metavar=("RUST", "ALTERNARIA", "GRAY", "BROWN"),
    )
    parser.add_argument("--component-aux", type=str2bool, default=False, help="Use lesion/boundary/center auxiliary heads.")
    parser.add_argument("--component-lesion-weight", type=float, default=0.4)
    parser.add_argument("--component-boundary-weight", type=float, default=0.2)
    parser.add_argument("--component-center-weight", type=float, default=0.2)
    parser.add_argument("--severity-consistency-loss", type=str2bool, default=False, help="Constrain predicted lesion/leaf ratio to match ground truth severity.")
    parser.add_argument("--severity-consistency-weight", type=float, default=0.1)
    parser.add_argument("--severity-loss-type", type=str, default="l1", choices=["l1", "smooth_l1", "mse"])
    parser.add_argument("--cls-weights", nargs="+", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--auto-export-report", type=str2bool, default=True)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument(
        "--report-checkpoint",
        type=str,
        default="best_miou",
        choices=["best_miou", "best_val_loss", "last", "best_epoch"],
    )
    parser.add_argument("--report-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--report-fps-interval", type=int, default=100)
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Class names used in the final report. Length must match --num-classes.",
    )
    return parser.parse_args()


def build_cls_weights(args):
    if args.cls_weights is None:
        return np.ones([args.num_classes], np.float32)
    if len(args.cls_weights) != args.num_classes:
        raise ValueError("--cls-weights length must match --num-classes")
    return np.array(args.cls_weights, np.float32)


def resolve_dataset_path(args):
    env_vocdevkit_path = os.environ.get("ATLDSD_VOCDEVKIT_PATH")
    if args.vocdevkit_path != "VOCdevkit":
        return args.vocdevkit_path
    if env_vocdevkit_path:
        return env_vocdevkit_path
    if args.datasets_root != ".":
        return os.path.join(args.datasets_root, f"{args.dataset_name}devkit")
    return str(DEFAULT_VOCDEVKIT_PATH)


def default_class_names(num_classes):
    if num_classes == 5:
        return ["background", "Bacterialblight", "Blast", "Brownspot", "Tungro"]
    return [f"class_{idx}" for idx in range(num_classes)]


def auto_export_report(args, dataset_path):
    model_root = Path(__file__).resolve().parent
    lab_root = model_root.parents[2]
    exporter = lab_root / "scripts" / "export_segmentation_report.py"
    if not exporter.exists():
        exporter = lab_root / "scripts" / "export_riceseg_segmentation_report.py"
    if not exporter.exists():
        print(f"[AutoReport] Skip: exporter not found: {exporter}")
        return

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = lab_root / save_dir
    save_dir = save_dir.resolve()
    checkpoint_candidates = {
        "best_miou": [save_dir / "best_miou_weights.pth", save_dir / "best_epoch_weights.pth", save_dir / "last_epoch_weights.pth"],
        "best_val_loss": [save_dir / "best_val_loss_weights.pth", save_dir / "best_epoch_weights.pth", save_dir / "last_epoch_weights.pth"],
        "best_epoch": [save_dir / "best_epoch_weights.pth", save_dir / "last_epoch_weights.pth"],
        "last": [save_dir / "last_epoch_weights.pth"],
    }
    model_path = next((path for path in checkpoint_candidates[args.report_checkpoint] if path.exists()), None)
    if model_path is None or not model_path.exists():
        print(f"[AutoReport] Skip: no checkpoint found in {save_dir}")
        return

    if args.report_dir is None:
        output_dir = save_dir.parent / "reports" / f"{args.report_checkpoint}_{args.report_split}"
    else:
        output_dir = Path(args.report_dir)
        if not output_dir.is_absolute():
            output_dir = lab_root / output_dir
        output_dir = output_dir.resolve()

    class_names = args.class_names or default_class_names(args.num_classes)
    if len(class_names) != args.num_classes:
        raise ValueError("--class-names length must match --num-classes")

    attention_type = args.attention_type if args.attention_type else "none"
    command = [
        sys.executable,
        str(exporter),
        "--model-adapter",
        "deeplabv3plus",
        "--model-root",
        str(model_root),
        "--vocdevkit-path",
        str(Path(dataset_path)),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--split",
        args.report_split,
        "--num-classes",
        str(args.num_classes),
        "--class-names",
        *class_names,
        "--backbone",
        args.backbone,
        "--attention-type",
        attention_type,
        "--use-ppm",
        str(args.use_ppm).lower(),
        "--component-aux",
        str(args.component_aux).lower(),
        "--downsample-factor",
        str(args.downsample_factor),
        "--decoder-conv-type",
        args.decoder_conv_type,
        "--input-shape",
        str(args.input_shape[0]),
        str(args.input_shape[1]),
        "--cuda",
        str(args.cuda).lower(),
        "--fps-interval",
        str(args.report_fps_interval),
        "--leaf-class-id",
        "1",
    ]
    for arg_name, value in [
        ("--attention-low-type", args.attention_low_type),
        ("--attention-high-type", args.attention_high_type),
        ("--attention-aspp-type", args.attention_aspp_type),
        ("--attention-decoder-type", args.attention_decoder_type),
    ]:
        if value is not None:
            command.extend([arg_name, value])
    print("[AutoReport] Exporting final metrics/confusion/complexity report...")
    print("[AutoReport] " + " ".join(command))
    subprocess.run(command, cwd=str(lab_root), check=True)
    print(f"[AutoReport] Report saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    lab_root = Path(__file__).resolve().parents[3]
    if not Path(args.save_dir).is_absolute():
        args.save_dir = str((lab_root / args.save_dir).resolve())
    if not Path(args.log_dir).is_absolute():
        args.log_dir = str((lab_root / args.log_dir).resolve())
    input_shape = list(args.input_shape)
    min_lr = args.min_lr if args.min_lr is not None else args.init_lr * 0.01
    cls_weights = build_cls_weights(args)
    use_cuda = args.cuda and torch.cuda.is_available()
    dataset_path = resolve_dataset_path(args)
    if args.lbft_loss and (args.dice_loss or args.focal_loss):
        raise ValueError("--lbft-loss already combines Weighted CE and Focal Tversky. Set --dice-loss false and --focal-loss false.")

    seed_everything(args.seed)
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        local_rank = 0
        rank = 0

    if args.pretrained and backbone_has_external_pretrained_url(args.backbone):
        if args.distributed:
            if local_rank == 0:
                download_backbone_pretrained_weights(args.backbone)
            dist.barrier()
        else:
            download_backbone_pretrained_weights(args.backbone)

    model = DeepLab(
        num_classes=args.num_classes,
        backbone=args.backbone,
        downsample_factor=args.downsample_factor,
        pretrained=args.pretrained,
        attention_type=args.attention_type,
        attention_low_type=args.attention_low_type,
        attention_high_type=args.attention_high_type,
        attention_aspp_type=args.attention_aspp_type,
        attention_decoder_type=args.attention_decoder_type,
        decoder_conv_type=args.decoder_conv_type,
        use_ppm=args.use_ppm,
        ppm_bins=args.ppm_bins,
        use_component_aux=args.component_aux,
    )
    if not args.pretrained:
        weights_init(model)

    if args.model_path:
        if local_rank == 0:
            print(f"Load weights {args.model_path}.")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "...\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "...\nFail To Load Key Num:", len(no_load_key))
            print("\n\033[1;33;44mHead mismatch is normal when switching to 3 classes, but backbone mismatch is not.\033[0m")

    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(args.log_dir, "loss_" + time_str)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        log_dir = None
        loss_history = None

    if args.fp16:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("sync_bn is only supported in distributed multi-GPU mode.")

    if use_cuda:
        if args.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r", encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            dataset_name=args.dataset_name,
            dataset_path=dataset_path,
            num_classes=args.num_classes,
            backbone=args.backbone,
            attention_type=args.attention_type,
            attention_low_type=args.attention_low_type,
            attention_high_type=args.attention_high_type,
            attention_aspp_type=args.attention_aspp_type,
            attention_decoder_type=args.attention_decoder_type,
            decoder_conv_type=args.decoder_conv_type,
            use_ppm=args.use_ppm,
            ppm_bins=args.ppm_bins,
            model_path=args.model_path,
            input_shape=input_shape,
            Init_Epoch=args.init_epoch,
            Freeze_Epoch=args.freeze_epoch,
            UnFreeze_Epoch=args.unfreeze_epoch,
            Freeze_batch_size=args.freeze_batch_size,
            Unfreeze_batch_size=args.unfreeze_batch_size,
            Freeze_Train=args.freeze_train,
            Init_lr=args.init_lr,
            Min_lr=min_lr,
            optimizer_type=args.optimizer_type,
            momentum=args.momentum,
            lr_decay_type=args.lr_decay_type,
            save_period=args.save_period,
            save_dir=args.save_dir,
            log_dir=log_dir,
            dice_loss=args.dice_loss,
            focal_loss=args.focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            lbft_loss=args.lbft_loss,
            lbft_lambda=args.lbft_lambda,
            lbft_alpha=args.lbft_alpha,
            lbft_beta=args.lbft_beta,
            lbft_gamma=args.lbft_gamma,
            mix_mode=args.mix_mode,
            mix_prob=args.mix_prob,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            sclp=args.sclp,
            sclp_prob=args.sclp_prob,
            sclp_max_components=args.sclp_max_components,
            component_aux=args.component_aux,
            component_lesion_weight=args.component_lesion_weight,
            component_boundary_weight=args.component_boundary_weight,
            component_center_weight=args.component_center_weight,
            severity_consistency_loss=args.severity_consistency_loss,
            severity_consistency_weight=args.severity_consistency_weight,
            severity_loss_type=args.severity_loss_type,
            num_workers=args.num_workers,
            num_train=num_train,
            num_val=num_val,
        )

        wanted_step = 1.5e4 if args.optimizer_type == "sgd" else 0.5e4
        total_step = num_train // args.unfreeze_batch_size * args.unfreeze_epoch
        if total_step <= wanted_step:
            if num_train // args.unfreeze_batch_size == 0:
                raise ValueError("Dataset is too small to continue training.")
            wanted_epoch = wanted_step // (num_train // args.unfreeze_batch_size) + 1
            print(f"\n\033[1;33;44m[Warning] {args.optimizer_type} is usually trained for at least {wanted_step:.0f} steps.\033[0m")
            print(
                f"\033[1;33;44m[Warning] Current setup: num_train={num_train}, "
                f"unfreeze_batch_size={args.unfreeze_batch_size}, epochs={args.unfreeze_epoch}, total_step={total_step}.\033[0m"
            )
            print(f"\033[1;33;44m[Warning] Suggested total epochs: {wanted_epoch}.\033[0m")

    unfreeze_flag = False
    if args.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = args.freeze_batch_size if args.freeze_train else args.unfreeze_batch_size

    nbs = 16
    lr_limit_max, lr_limit_min = get_backbone_lr_limits(args.backbone, args.optimizer_type)
    init_lr_fit = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        "adam": optim.Adam(model.parameters(), init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
        "sgd": optim.SGD(model.parameters(), init_lr_fit, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay),
    }[args.optimizer_type]

    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, init_lr_fit, min_lr_fit, args.unfreeze_epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset is too small to continue training.")

    sclp_class_weights = {
        class_id: weight
        for class_id, weight in zip(range(2, 6), args.sclp_class_weights)
    }
    train_dataset = DeeplabDataset(
        train_lines,
        input_shape,
        args.num_classes,
        True,
        dataset_path,
        sclp=args.sclp,
        sclp_prob=args.sclp_prob,
        sclp_max_components=args.sclp_max_components,
        sclp_class_weights=sclp_class_weights,
    )
    val_dataset = DeeplabDataset(val_lines, input_shape, args.num_classes, False, dataset_path)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
    )

    if local_rank == 0:
        eval_callback = EvalCallback(
            model,
            input_shape,
            args.num_classes,
            val_lines,
            dataset_path,
            log_dir,
            use_cuda,
            eval_flag=args.eval_flag,
            period=args.eval_period,
        )
    else:
        eval_callback = None

    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        if epoch >= args.freeze_epoch and not unfreeze_flag and args.freeze_train:
            batch_size = args.unfreeze_batch_size

            init_lr_fit = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, init_lr_fit, min_lr_fit, args.unfreeze_epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("Dataset is too small to continue training.")

            if args.distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(
                train_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
            )
            gen_val = DataLoader(
                val_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
            )
            unfreeze_flag = True

        if args.distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            args.unfreeze_epoch,
            use_cuda,
            args.dice_loss,
            args.focal_loss,
            cls_weights,
            args.num_classes,
            args.fp16,
            scaler,
            args.save_period,
            args.save_dir,
            local_rank,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            lbft_loss=args.lbft_loss,
            lbft_lambda=args.lbft_lambda,
            lbft_alpha=args.lbft_alpha,
            lbft_beta=args.lbft_beta,
            lbft_gamma=args.lbft_gamma,
            mix_mode=args.mix_mode,
            mix_prob=args.mix_prob,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            component_aux=args.component_aux,
            component_lesion_weight=args.component_lesion_weight,
            component_boundary_weight=args.component_boundary_weight,
            component_center_weight=args.component_center_weight,
            severity_consistency_loss=args.severity_consistency_loss,
            severity_consistency_weight=args.severity_consistency_weight,
            severity_loss_type=args.severity_loss_type,
        )

        if args.distributed:
            dist.barrier()

    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()

    if local_rank == 0 and args.auto_export_report:
        auto_export_report(args, dataset_path)
