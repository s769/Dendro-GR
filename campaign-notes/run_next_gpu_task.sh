#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

# Queue #4: hang-path NCU occupancy on __unzip_cg1__2
if [[ -x "${HOME}/Dendro-GR-notes/run_tier2_ncu_hang.sh" ]]; then
  echo "==== queue#4 tier2_ncu_hang ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier2_ncu_hang.sh" | tee -a "${LOG}"
  echo TIER2_NCU_HANG_QUEUE_DONE | tee -a "${LOG}"
else
  echo "run_tier2_ncu_hang.sh missing" | tee -a "${LOG}"
  exit 1
fi
