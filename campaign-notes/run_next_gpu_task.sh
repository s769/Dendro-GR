#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

if [[ ! -f "${HOME}/Dendro-GR-notes/tier0/tier32_richardson_q4.log" ]] || \
   ! grep -q TIER32_RICHARDSON_Q4_DONE "${HOME}/Dendro-GR-notes/tier0/tier32_richardson_q4.log" 2>/dev/null; then
  echo "==== queue#12 Tier 3.2 Richardson q4 ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier32_richardson_q4.sh" | tee -a "${LOG}"
else
  echo "no unfinished GPU probe ready" | tee -a "${LOG}"
fi
