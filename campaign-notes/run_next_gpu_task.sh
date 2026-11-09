#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

# Prefer a concrete probe script if present; else re-gate TL q1 (fast, real).
if [[ -x "${HOME}/Dendro-GR-notes/run_tier2_coalesce_smoke.sh" ]]; then
  echo "==== queue#7 coalesce_smoke ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier2_coalesce_smoke.sh" | tee -a "${LOG}"
elif [[ -x "${HOME}/Dendro-GR-notes/run_tier31_q1_tl.sh" ]]; then
  echo "==== re-gate tier31_q1_tl ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier31_q1_tl.sh" | tee -a "${LOG}"
else
  echo "no GPU probe script ready" | tee -a "${LOG}"
fi
