#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

BIN_TL="${DENDRO_BUILD}/BSSN_GR/tl_gauge_device_test"
if [[ -x "${BIN_TL}" ]]; then
  echo "==== tl_gauge_device_test ====" | tee -a "${LOG}"
  "${BIN_TL}" 1e-6 | tee -a "${LOG}"
  "${BIN_TL}" 1e-7 | tee -a "${LOG}"
  echo TIER31_DEVICE_TL_DONE | tee -a "${LOG}"
else
  echo "tl_gauge_device_test missing — build on login first" | tee -a "${LOG}"
fi
