#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

resolve_vocdevkit() {
  if [[ -n "${ATLDSD_VOCDEVKIT_PATH:-}" ]]; then
    echo "${ATLDSD_VOCDEVKIT_PATH}"
  elif [[ -d "${PROJECT_ROOT}/VOCdevkit" ]]; then
    echo "${PROJECT_ROOT}/VOCdevkit"
  elif [[ -d "${PROJECT_ROOT}/data/VOCdevkit" ]]; then
    echo "${PROJECT_ROOT}/data/VOCdevkit"
  elif [[ -d "/home/liuzhe/dataset/ATLDSD/VOCdevkit" ]]; then
    echo "/home/liuzhe/dataset/ATLDSD/VOCdevkit"
  else
    echo "${PROJECT_ROOT}/VOCdevkit"
  fi
}

VOCDEVKIT_PATH="$(resolve_vocdevkit)"
SPLIT_FILE="${VOCDEVKIT_PATH}/VOC2007/ImageSets/Segmentation/train.txt"
if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "[error] Cannot find VOC split file: ${SPLIT_FILE}" >&2
  echo "[hint] Put VOCdevkit under the project root or export ATLDSD_VOCDEVKIT_PATH=/absolute/path/VOCdevkit" >&2
  exit 1
fi

export ATLDSD_PROJECT_ROOT="${PROJECT_ROOT}"
export ATLDSD_VOCDEVKIT_PATH="${VOCDEVKIT_PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/models/deeplabv3plus:${PROJECT_ROOT}/src/modules${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ROOT="${PROJECT_ROOT}/outputs/atldsd/deeplabv3plus_mobilenetv3_large_150"
mkdir -p "${RUN_ROOT}/weights" "${RUN_ROOT}/logs" "${RUN_ROOT}/reports/best_val"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/models/deeplabv3plus/train.py" \
  --cuda true \
  --seed 11 \
  --num-classes 6 \
  --backbone mobilenetv3_large \
  --pretrained true \
  --downsample-factor 16 \
  --attention-type none \
  --use-ppm false \
  --input-shape 256 256 \
  --init-epoch 0 \
  --freeze-epoch 50 \
  --freeze-batch-size 8 \
  --unfreeze-epoch "${EPOCHS:-150}" \
  --unfreeze-batch-size "${BATCH_SIZE:-4}" \
  --freeze-train true \
  --init-lr 0.003 \
  --optimizer-type sgd \
  --lr-decay-type cos \
  --save-period 10 \
  --eval-period 10 \
  --dataset-name ATLDSD \
  --vocdevkit-path "${VOCDEVKIT_PATH}" \
  --dice-loss true \
  --focal-loss false \
  --num-workers "${NUM_WORKERS:-4}" \
  --auto-export-report true \
  --report-dir "${RUN_ROOT}/reports/best_val" \
  --report-split val \
  --report-fps-interval 100 \
  --save-dir "${RUN_ROOT}/weights" \
  --log-dir "${RUN_ROOT}/logs" \
  --class-names background leaf rust alternaria_leaf_spot gray_spot brown_spot
