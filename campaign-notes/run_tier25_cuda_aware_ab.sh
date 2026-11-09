#!/usr/bin/env bash
# Compare q8/4GPU ETS: this binary built with -DMPIX_CUDA_AWARE_SUPPORT=ON
# vs recorded host-staged baseline (~2.99–3.03s).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile 2>/dev/null || true
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/q8_cuda_aware"
LOGDIR="${HOME}/Dendro-GR-notes/tier0"
LOG="${LOGDIR}/tier25_q8_cuda_aware.log"
SUMMARY="${LOGDIR}/tier25_cuda_aware_ab.log"
rm -rf "${OUT}"
mkdir -p "${OUT}"/{aeh,cp,dat,vtu} "${LOGDIR}"
cp "${PAR0}" "${OUT}/local.par.toml"
python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml")
t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.5", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_TIME_STEP_OUTPUT_FREQ = [0-9]+", "BSSN_TIME_STEP_OUTPUT_FREQ = 1", t)
p.write_text(t)
PY
ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
cd "${OUT}"
echo "host $(hostname) $(date -Is) MPIX_CUDA_AWARE_SUPPORT=ON" | tee "${SUMMARY}"
# UCX/OpenMPI CUDA transport hints
export UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"
START=$(date +%s)
mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
END=$(date +%s)
WALL=$((END-START))
ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
echo "cuda_aware wall_sec=${WALL} ets=${ETS:-NA}" | tee -a "${SUMMARY}"
echo "baseline_host_staged ets≈2.99–3.03 (tier23_baseline / tier25_q8_np4)" | tee -a "${SUMMARY}"
grep -E 'ETS time|error|Error|CUDA|cuda' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -30 | tee -a "${SUMMARY}" || true
echo TIER25_CUDA_AWARE_AB_DONE | tee -a "${SUMMARY}"
