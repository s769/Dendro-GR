#!/usr/bin/env bash
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier21_lb1_ab.log"
echo "==== confirm2 $(date -Is) ====" | tee -a "${LOG}"
cd "${HOME}/Dendro-GR-runs/tier21_lb1_ab/ASYNC"
START=$(date +%s)
mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA" local.par.toml 1 >run2.log 2>&1
END=$(date +%s)
ETS=$(grep -E 'ETS time \(max\)' run2.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
echo "ASYNC_lb1_r2 wall=$((END-START))s ets=${ETS:-NA}" | tee -a "${LOG}"
cmake --build "${DENDRO_BUILD}" -j16 --target rhsgpu_test 2>&1 | tail -5 | tee -a "${LOG}"
DENDRO_TIME_PT_ASYNC=1 "${DENDRO_BUILD}/BSSN_GR/rhsgpu_test" 2048 20 0 2>&1 | tee -a "${LOG}" | grep pt_async || true
echo LB1_CONFIRM_DONE | tee -a "${LOG}"
