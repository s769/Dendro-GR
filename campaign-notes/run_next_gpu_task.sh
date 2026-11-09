#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

if [[ ! -f "${HOME}/Dendro-GR-notes/tier0/tier21_ncu_rhs.log" ]] || \
   ! grep -q TIER21_NCU_RHS_DONE "${HOME}/Dendro-GR-notes/tier0/tier21_ncu_rhs.log" 2>/dev/null; then
  echo "==== B0 NCU eval_rhs3_pt_async ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier21_ncu_rhs.sh" | tee -a "${LOG}"
elif [[ -x "${HOME}/Dendro-GR-notes/run_tier21_staged_ab.sh" ]]; then
  echo "==== B1 staged algebra A/B ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier21_staged_ab.sh" | tee -a "${LOG}"
elif [[ -x "${HOME}/Dendro-GR-notes/run_tier_batch_kcta.sh" ]]; then
  echo "==== B2 K/CTA speedup curve ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier_batch_kcta.sh" | tee -a "${LOG}"
else
  echo "no unfinished GPU probe ready" | tee -a "${LOG}"
fi
