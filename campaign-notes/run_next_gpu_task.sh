#!/usr/bin/env bash
# Run the next open GPU task from OPEN_TASK_QUEUE.md (no filler).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/next_gpu_task.log"
echo "host $(hostname) $(date -Is) next_gpu_task" | tee -a "${LOG}"

# Queue #5 placeholder: frozen-mesh unzip reuse needs coding first on login.
# When a runnable GPU probe exists, wire it here. For now report and exit 0
# so watchers do not spin filler.
if [[ -x "${HOME}/Dendro-GR-notes/run_tier2_frozen_unzip_probe.sh" ]]; then
  echo "==== queue#5 frozen_unzip_probe ====" | tee -a "${LOG}"
  bash "${HOME}/Dendro-GR-notes/run_tier2_frozen_unzip_probe.sh" | tee -a "${LOG}"
else
  echo "queue#5: no GPU probe yet — code frozen-mesh unzip reuse on login" | tee -a "${LOG}"
fi
