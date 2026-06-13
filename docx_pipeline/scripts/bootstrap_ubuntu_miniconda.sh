#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${MINICONDA_DIR:-${HOME}/miniconda3}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

case "$(uname -m)" in
  x86_64|amd64)
    INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    ;;
  aarch64|arm64)
    INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    ;;
  *)
    echo "[error] Unsupported CPU architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

if command -v python3 >/dev/null 2>&1; then
  echo "[bootstrap] python3 already exists: $(command -v python3)"
  exit 0
fi

if [[ -x "${INSTALL_DIR}/bin/python" ]]; then
  echo "[bootstrap] Miniconda already exists: ${INSTALL_DIR}"
  echo "[bootstrap] Run: export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
  exit 0
fi

DOWNLOADER=""
if command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl -L"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget -O -"
else
  echo "[error] Need curl or wget to download Miniconda." >&2
  echo "[hint] If you have sudo: sudo apt-get update && sudo apt-get install -y curl" >&2
  exit 1
fi

echo "[bootstrap] Downloading Miniconda from ${INSTALLER_URL}"
INSTALLER="${TMP_DIR}/miniconda.sh"
${DOWNLOADER} "${INSTALLER_URL}" > "${INSTALLER}"

echo "[bootstrap] Installing Miniconda to ${INSTALL_DIR}"
bash "${INSTALLER}" -b -p "${INSTALL_DIR}"

echo "[bootstrap] Done."
echo "[bootstrap] Run this before setup/training:"
echo "export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
