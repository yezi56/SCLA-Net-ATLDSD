#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-sclp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${MODE}" in
  baseline|v3|b0)
    exec "${SCRIPT_DIR}/run_ubuntu_baseline_v3.sh"
    ;;
  sclp|e1|ours)
    exec "${SCRIPT_DIR}/run_ubuntu_sclp_v3.sh"
    ;;
  sclp03|e1.1)
    exec "${SCRIPT_DIR}/run_ubuntu_sclp03_v3.sh"
    ;;
  component_aux|e2)
    exec "${SCRIPT_DIR}/run_ubuntu_component_aux_v3.sh"
    ;;
  component_aux_severity|severity|m2|e3)
    exec "${SCRIPT_DIR}/run_ubuntu_component_aux_severity_v3.sh"
    ;;
  *)
    echo "Usage: $0 [baseline|sclp|sclp03|component_aux|component_aux_severity]" >&2
    exit 2
    ;;
esac
