#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

if [[ -x "${HOME}/Dendro-GR-notes/run_tier31_Bdriver_test.sh" ]]; then
  echo "==== queue#10 Bdriver TL ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier31_Bdriver_test.sh" | tee -a "${LOG}"
elif [[ -x "${HOME}/Dendro-GR-notes/run_tier31_shift_beta.sh" ]]; then
  echo "==== re-gate shift beta ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier31_shift_beta.sh" | tee -a "${LOG}"
elif [[ -x "${HOME}/Dendro-GR-notes/run_tier31_radbc.sh" ]]; then
  bash "${HOME}/Dendro-GR-notes/run_tier31_radbc.sh" | tee -a "${LOG}"
else
  echo "no GPU probe ready" | tee -a "${LOG}"
fi
