import argparse
import copy
import json
from pathlib import Path

import torch

from check_deeplab_repconv_deploy import (
    benchmark,
    build_model,
    count_params,
    fuse_repconv_for_deploy,
    load_json,
    main_output,
    repconv_status,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a DeepLabV3+ RepConv checkpoint after deploy fusion."
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        required=True,
        help="Path to the original report run_config.json.",
    )
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-run-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


def write_markdown(path, result):
    lines = [
        "# DeepLabV3+ RepConv Deploy Export",
        "",
        f"- source_checkpoint: `{result['source_checkpoint']}`",
        f"- deploy_checkpoint: `{result['deploy_checkpoint']}`",
        f"- deploy_run_config: `{result['deploy_run_config']}`",
        f"- device: `{result['device']}`",
        f"- input_shape: `{result['input_shape']}`",
        f"- repconv_blocks_before: {result['repconv_blocks_before']}",
        f"- repconv_blocks_fused: {result['repconv_blocks_fused']}",
        f"- repconv_blocks_after: {result['repconv_blocks_after']}",
        f"- params_before: {result['params_before']}",
        f"- params_after: {result['params_after']}",
        f"- max_abs_diff: {result['max_abs_diff']:.8g}",
        f"- mean_abs_diff: {result['mean_abs_diff']:.8g}",
        f"- pred_mismatch_rate: {result['pred_mismatch_rate']:.8g}",
        f"- allclose_atol_1e-4_rtol_1e-4: {result['allclose_atol_1e-4_rtol_1e-4']}",
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
            "- `{name}`: train_branches={train_branches}, deploy_branch={has_rbr_reparam}".format(
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
    deploy_model = copy.deepcopy(model).to(device).eval()

    input_shape = config.get("input_shape", [384, 384])
    sample = torch.randn(1, 3, int(input_shape[0]), int(input_shape[1]), device=device)
    with torch.no_grad():
        before_logits = main_output(model(sample)).detach()

    status_before = repconv_status(deploy_model)
    fused_count = fuse_repconv_for_deploy(deploy_model)
    status_after = repconv_status(deploy_model)
    with torch.no_grad():
        after_logits = main_output(deploy_model(sample)).detach()

    diff = (before_logits - after_logits).abs()
    pred_mismatch = (before_logits.argmax(dim=1) != after_logits.argmax(dim=1)).float().mean().item()
    benchmark_before = benchmark(model, sample, args.warmup, args.iterations, device)
    benchmark_after = benchmark(deploy_model, sample, args.warmup, args.iterations, device)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": deploy_model.cpu().state_dict(),
            "meta": {
                "source_checkpoint": config["model_path"],
                "repconv_deploy_fused": True,
                "repconv_blocks_fused": int(fused_count),
            },
        },
        args.output_checkpoint,
    )

    deploy_config = dict(config)
    deploy_config["model_path"] = str(args.output_checkpoint.resolve())
    deploy_config["repconv_deploy_fused"] = True
    deploy_config["source_model_path"] = config["model_path"]
    write_json(args.output_run_config, deploy_config)

    result = {
        "run_config": str(args.run_config),
        "source_checkpoint": config["model_path"],
        "deploy_checkpoint": str(args.output_checkpoint),
        "deploy_run_config": str(args.output_run_config),
        "device": str(device),
        "input_shape": [int(input_shape[0]), int(input_shape[1])],
        "repconv_blocks_before": len(status_before),
        "repconv_blocks_fused": int(fused_count),
        "repconv_blocks_after": len(status_after),
        "status_before": status_before,
        "status_after": status_after,
        "params_before": count_params(model),
        "params_after": count_params(deploy_model),
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
