#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

# Tier 3.3 GPU measure only after the orbits knob exists.
if [[ -x "${HOME}/Dendro-GR-notes/run_tier33_orbits_q8.sh" ]]; then
  echo "==== queue#14 Tier 3.3 orbits cost ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier33_orbits_q8.sh" | tee -a "${LOG}"
else
  echo "no unfinished GPU probe ready (codegen / 3.3 knob on login)" | tee -a "${LOG}"
fi
