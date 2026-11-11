#!/usr/bin/env bash
# Tier 3.1 #3: dump one interior unzipped q1 gauge block, then device TL vs FD.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
TEST="${DENDRO_BUILD}/BSSN_GR/tl_gauge_q1_test"
PAR0="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
REC="${HOME}/Dendro-GR-runs/tier14/mesh_rec"
OUT="${HOME}/Dendro-GR-runs/tier31_q1_tl"
DUMP="${OUT}/tl_gauge_blk.bin"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier31_q1_tl.log"

mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
cp "${PAR0}" "${OUT}/local.par.toml"
ln -sfn "${TPID}" "${OUT}/rit_q1_tpid_sol.bin"
: > "${LOG}"
echo "host $(hostname) $(date -Is) tier31_q1_tl" | tee -a "${LOG}"

if [[ ! -x "${BIN}" ]]; then
  echo "missing ${BIN}" | tee -a "${LOG}"
  exit 1
fi
if [[ ! -x "${TEST}" ]]; then
  echo "missing ${TEST} — build tl_gauge_q1_test first" | tee -a "${LOG}"
  exit 1
fi

rm -f "${DUMP}"
cd "${OUT}"
mpirun -np 1 --bind-to none env -u CUDA_VISIBLE_DEVICES \
  DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY="${REC}" \
  DENDRO_DUMP_TL_GAUGE="${DUMP}" \
  "${BIN}" "${OUT}/local.par.toml" 1 2>&1 | tee -a "${LOG}"

if [[ ! -s "${DUMP}" ]]; then
  echo "dump missing after solver run" | tee -a "${LOG}"
  exit 2
fi

echo "==== tl_gauge_q1_test ====" | tee -a "${LOG}"
"${TEST}" "${DUMP}" 1e-6 | tee -a "${LOG}"
"${TEST}" "${DUMP}" 1e-7 | tee -a "${LOG}"
echo TIER31_Q1_TL_DONE | tee -a "${LOG}"
