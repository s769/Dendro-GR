#!/usr/bin/env bash
# Tier 3.1 radiative-BC TL gate (device).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/tier31_radbc.log"
BIN="${DENDRO_BUILD}/BSSN_GR/tl_gauge_radbc_test"
echo "host $(hostname) $(date -Is) tier31_radbc" | tee "${LOG}"
if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN}" | tee -a "${LOG}"
  exit 1
fi
"${BIN}" 1e-6 | tee -a "${LOG}"
"${BIN}" 1e-7 | tee -a "${LOG}"
echo TIER31_RADBC_DONE | tee -a "${LOG}"
