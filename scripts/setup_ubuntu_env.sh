#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_CHANNEL="${1:-skip}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

case "${TORCH_CHANNEL}" in
  cu124)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ;;
  cu121)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ;;
  cu118)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ;;
  cpu)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ;;
  skip)
    echo "[setup] Skip torch install. Install torch/torchvision yourself if not already available."
    ;;
  *)
    echo "Usage: $0 [skip|cpu|cu118|cu121|cu124]" >&2
    exit 2
    ;;
esac

python -m pip install -r "${PROJECT_ROOT}/requirements-atldsd.txt"

echo "[setup] Python: $(command -v python)"
python - <<'PY'
try:
    import torch
except ImportError:
    print("[setup] torch is not installed. Install torch/torchvision before training.")
else:
    print(f"[setup] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[setup] gpu={torch.cuda.get_device_name(0)}")
PY
