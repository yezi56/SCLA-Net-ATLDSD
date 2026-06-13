import argparse
import copy
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    normalized = OrderedDict()
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]
        normalized[key] = value
    return normalized


def normalize_attention_type(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "none", "identity"}:
        return ""
    return value


def build_model(config, device):
    model_root = Path(config["model_root"]).resolve()
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    from nets.deeplabv3_plus import DeepLab

    model = DeepLab(
        num_classes=int(config["num_classes"]),
        backbone=config.get("backbone", "mobilenetv3_large"),
        downsample_factor=int(config.get("downsample_factor", 16)),
        pretrained=False,
        attention_type=normalize_attention_type(config.get("attention_type", "")),
        attention_low_type=normalize_attention_type(config.get("attention_low_type")),
        attention_high_type=normalize_attention_type(config.get("attention_high_type")),
        attention_aspp_type=normalize_attention_type(config.get("attention_aspp_type")),
        attention_decoder_type=normalize_attention_type(config.get("attention_decoder_type")),
        decoder_conv_type=config.get("decoder_conv_type", "standard"),
        decoder_upsample_type=config.get("decoder_upsample_type", "bilinear"),
        use_ppm=bool(config.get("use_ppm", False)),
        use_component_aux=bool(config.get("component_aux", False)),
        use_lbsb=bool(config.get("lesion_boundary_sharpen", False)),
        lbsb_alpha=float(config.get("lesion_boundary_sharpen_alpha", 0.25)),
        use_lcaf=bool(config.get("lesion_cross_scale_fusion", False)),
        lcaf_alpha=float(config.get("lesion_cross_scale_fusion_alpha", 0.5)),
        use_lglc=bool(config.get("lesion_local_global_context", False)),
        lglc_alpha=float(config.get("lesion_local_global_context_alpha", 0.5)),
        use_chfr=bool(config.get("component_high_frequency_refinement", False)),
        chfr_alpha=float(config.get("component_high_frequency_refinement_alpha", 0.2)),
    )
    checkpoint = torch.load(config["model_path"], map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def main_output(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def repconv_modules(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "RepConvBlock"
    ]


def repconv_status(model):
    rows = []
    for name, module in repconv_modules(model):
        rows.append(
            {
                "name": name,
                "has_fuse_for_deploy": hasattr(module, "fuse_for_deploy"),
                "has_switch_to_deploy": hasattr(module, "switch_to_deploy"),
                "has_reparameterize": hasattr(module, "reparameterize"),
                "has_rbr_dense": hasattr(module, "rbr_dense"),
                "has_rbr_1x1": hasattr(module, "rbr_1x1"),
                "has_rbr_identity": getattr(module, "rbr_identity", None) is not None,
                "has_rbr_reparam": hasattr(module, "rbr_reparam"),
            }
        )
    return rows


def fuse_repconv_for_deploy(model):
    fused = 0
    for _, module in repconv_modules(model):
        if hasattr(module, "fuse_for_deploy") and not hasattr(module, "rbr_reparam"):
            module.fuse_for_deploy()
            fused += 1
    return fused


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def benchmark(model, sample, warmup, iterations, device):
    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    time_per_image = elapsed / max(iterations, 1)
    return {
        "iterations": int(iterations),
        "elapsed_seconds": float(elapsed),
        "time_per_image_seconds": float(time_per_image),
        "fps": float(1.0 / time_per_image) if time_per_image > 0 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check DeepLabV3+ RepConv deploy fusion equivalence and speed."
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        required=True,
        help="Path to a report run_config.json generated by export_segmentation_report.py.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def write_markdown(path, result):
    lines = [
        "# RepConv Deploy Fuse Check",
        "",
        f"- model_path: `{result['model_path']}`",
        f"- device: `{result['device']}`",
        f"- input_shape: `{result['input_shape']}`",
        f"- repconv_blocks_before: {result['repconv_blocks_before']}",
        f"- repconv_blocks_fused: {result['repconv_blocks_fused']}",
        f"- repconv_blocks_after: {result['repconv_blocks_after']}",
        f"- max_abs_diff: {result['max_abs_diff']:.8g}",
        f"- mean_abs_diff: {result['mean_abs_diff']:.8g}",
        f"- allclose_atol_1e-4_rtol_1e-4: {result['allclose_atol_1e-4_rtol_1e-4']}",
        f"- params_before: {result['params_before']}",
        f"- params_after: {result['params_after']}",
        f"- fps_before: {result['benchmark_before']['fps']:.4f}",
        f"- fps_after: {result['benchmark_after']['fps']:.4f}",
        f"- time_before_seconds: {result['benchmark_before']['time_per_image_seconds']:.6f}",
        f"- time_after_seconds: {result['benchmark_after']['time_per_image_seconds']:.6f}",
        "",
        "## RepConv Blocks Before",
        "",
    ]
    for row in result["status_before"]:
        lines.append(
            "- `{name}`: fuse_for_deploy={has_fuse_for_deploy}, "
            "switch_to_deploy={has_switch_to_deploy}, reparameterize={has_reparameterize}, "
            "train_branches={train_branches}, deploy_branch={has_rbr_reparam}".format(
                train_branches=row["has_rbr_dense"] and row["has_rbr_1x1"],
                **row,
            )
        )
    lines.extend(["", "## RepConv Blocks After", ""])
    for row in result["status_after"]:
        lines.append(
            "- `{name}`: train_branches={train_branches}, deploy_branch={has_rbr_reparam}".format(
                train_branches=row["has_rbr_dense"] and row["has_rbr_1x1"],
                **row,
            )
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = load_json(args.run_config)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    model = build_model(config, device)
    fused_model = copy.deepcopy(model).to(device).eval()

    input_shape = config.get("input_shape", [384, 384])
    sample = torch.randn(1, 3, int(input_shape[0]), int(input_shape[1]), device=device)

    with torch.no_grad():
        before_logits = main_output(model(sample)).detach()

    status_before = repconv_status(fused_model)
    fused_count = fuse_repconv_for_deploy(fused_model)
    status_after = repconv_status(fused_model)

    with torch.no_grad():
        after_logits = main_output(fused_model(sample)).detach()

    diff = (before_logits - after_logits).abs()
    before_pred = before_logits.argmax(dim=1)
    after_pred = after_logits.argmax(dim=1)
    pred_mismatch = (before_pred != after_pred).float().mean().item()
    benchmark_before = benchmark(model, sample, args.warmup, args.iterations, device)
    benchmark_after = benchmark(fused_model, sample, args.warmup, args.iterations, device)

    result = {
        "run_config": str(args.run_config),
        "model_path": config["model_path"],
        "device": str(device),
        "input_shape": [int(input_shape[0]), int(input_shape[1])],
        "repconv_blocks_before": len(status_before),
        "repconv_blocks_fused": int(fused_count),
        "repconv_blocks_after": len(status_after),
        "status_before": status_before,
        "status_after": status_after,
        "params_before": count_params(model),
        "params_after": count_params(fused_model),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "pred_mismatch_rate": float(pred_mismatch),
        "allclose_atol_1e-4_rtol_1e-4": bool(torch.allclose(before_logits, after_logits, atol=1e-4, rtol=1e-4)),
        "allclose_atol_1e-2_rtol_1e-3": bool(torch.allclose(before_logits, after_logits, atol=1e-2, rtol=1e-3)),
        "benchmark_before": benchmark_before,
        "benchmark_after": benchmark_after,
    }
    write_json(args.output_json, result)
    if args.output_md:
        write_markdown(args.output_md, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
