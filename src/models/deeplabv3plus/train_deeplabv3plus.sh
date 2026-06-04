#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/semantic_seg/weights outputs/semantic_seg/logs

nohup python train.py > outputs/semantic_seg/train.log 2>&1 &

echo "Training started."
echo "Log file: outputs/semantic_seg/train.log"
echo "Watch with: tail -f outputs/semantic_seg/train.log"
