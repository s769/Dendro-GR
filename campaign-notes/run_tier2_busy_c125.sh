#!/usr/bin/env bash
# Keep a 4-GPU idev busy: nsys (deferred) + q8 1GPU ETS + q8 4GPU ETS.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"
export DENDRO_RHS_PT_ASYNC=1
export DENDRO_UNZIP_BATCH_SYNC=0

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
LOGDIR="${HOME}/Dendro-GR-notes/tier0"
SUMMARY="${LOGDIR}/tier2_busy_c125.log"
mkdir -p "${LOGDIR}"
echo "host $(hostname) $(date -Is) busy_pipeline" | tee "${SUMMARY}"

prep() {
  local OUT="$1" TEND="${2:-0.5}"
  rm -rf "${OUT}"
  mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${OUT}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = ${TEND}", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_TIME_STEP_OUTPUT_FREQ = [0-9]+", "BSSN_TIME_STEP_OUTPUT_FREQ = 1", t)
t=re.sub(r"BSSN_ASYNC_COMM_K = [0-9]+", "BSSN_ASYNC_COMM_K = 1", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
}

# 1) nsys q8/4GPU short
OUT="${HOME}/Dendro-GR-runs/q8_nsys_busy"
NSYS_OUT="${LOGDIR}/tier24_q8_busy"
prep "${OUT}" 0.5
cd "${OUT}"
NSYS=$(command -v nsys || echo /usr/local/bin/nsys)
echo "==== nsys ====" | tee -a "${SUMMARY}"
"${NSYS}" profile -o "${NSYS_OUT}" --force-overwrite=true \
  --trace=cuda,nvtx,osrt,mpi --mpi-impl=openmpi --stats=true \
  mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 DENDRO_UNZIP_BATCH_SYNC=0 \
  "${BIN}" "${OUT}/local.par.toml" 1 \
  > "${LOGDIR}/tier24_q8_busy_run.log" 2>&1 || true
grep -E 'ETS time' "${LOGDIR}/tier24_q8_busy_run.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | tee -a "${SUMMARY}" || true
"${NSYS}" stats --force-export=true --report cuda_gpu_kern_sum \
  "${NSYS_OUT}.nsys-rep" > "${LOGDIR}/tier24_q8_busy_stats.txt" 2>&1 || true
grep -E 'eval_rhs|unzip|stock|write_sb|zip_cg' "${LOGDIR}/tier24_q8_busy_stats.txt" | head -12 | tee -a "${SUMMARY}" || true

# 2) 1GPU ETS reference
OUT="${HOME}/Dendro-GR-runs/q8_np1_busy"
prep "${OUT}" 0.5
cd "${OUT}"
echo "==== np1 ====" | tee -a "${SUMMARY}"
START=$(date +%s)
mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 > "${LOGDIR}/tier25_q8_np1_busy.log" 2>&1 || true
ETS=$(grep -E 'ETS time \(max\)' "${LOGDIR}/tier25_q8_np1_busy.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
echo "tag=np1 wall_sec=$(( $(date +%s) - START )) ets=${ETS:-NA}" | tee -a "${SUMMARY}"

# 3) 4GPU ETS
OUT="${HOME}/Dendro-GR-runs/q8_np4_busy"
prep "${OUT}" 0.5
cd "${OUT}"
echo "==== np4 ====" | tee -a "${SUMMARY}"
START=$(date +%s)
mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 > "${LOGDIR}/tier25_q8_np4_busy.log" 2>&1 || true
ETS=$(grep -E 'ETS time \(max\)' "${LOGDIR}/tier25_q8_np4_busy.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
echo "tag=np4 wall_sec=$(( $(date +%s) - START )) ets=${ETS:-NA}" | tee -a "${SUMMARY}"

# 4) longer 4GPU for SNR
OUT="${HOME}/Dendro-GR-runs/q8_np4_t2_busy"
prep "${OUT}" 2.0
cd "${OUT}"
echo "==== np4 TIME_END=2.0 ====" | tee -a "${SUMMARY}"
START=$(date +%s)
mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 > "${LOGDIR}/tier25_q8_np4_t2_busy.log" 2>&1 || true
ETS=$(grep -E 'ETS time \(max\)' "${LOGDIR}/tier25_q8_np4_t2_busy.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
echo "tag=np4_t2 wall_sec=$(( $(date +%s) - START )) ets=${ETS:-NA}" | tee -a "${SUMMARY}"

echo TIER2_BUSY_C125_DONE | tee -a "${SUMMARY}"
