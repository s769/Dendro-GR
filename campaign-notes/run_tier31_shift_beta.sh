#!/usr/bin/env bash
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/tier31_shift_beta.log"
BIN="${DENDRO_BUILD}/BSSN_GR/tl_shift_beta_test"
echo "host $(hostname) $(date -Is) tier31_shift_beta" | tee "${LOG}"
[[ -x "${BIN}" ]] || { echo "missing ${BIN}"; exit 1; }
"${BIN}" 1e-6 | tee -a "${LOG}"
"${BIN}" 1e-7 | tee -a "${LOG}"
echo TIER31_SHIFT_BETA_DONE | tee -a "${LOG}"
