#!/usr/bin/env bash
# Tier 2.4: nsys on q8 4-GPU short evolve (honest multi-GPU, no CVD pin).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile 2>/dev/null || true
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
OUT="${HOME}/Dendro-GR-runs/q8_nsys_4gpu"
LOGDIR="${HOME}/Dendro-GR/campaign-notes/tier0"
NSYS_OUT="${LOGDIR}/tier24_q8_4gpu"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
PREFIX=mesh_q8

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
ln -sfn "${SOL}" "${OUT}/${PREFIX}_tpid_sol.bin"
cd "${OUT}"

NSYS=$(command -v nsys || echo /usr/local/bin/nsys)
START=$(date +%s)
# cuda + NVTX + MPI; keep report under notes/tier0
"${NSYS}" profile -o "${NSYS_OUT}" --force-overwrite=true \
  --trace=cuda,nvtx,osrt,mpi \
  --mpi-impl=openmpi \
  --stats=true \
  mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 \
  > "${LOGDIR}/tier24_q8_4gpu_nsys_run.log" 2>&1 || true
END=$(date +%s)
echo "wall_sec $((END-START))" | tee -a "${LOGDIR}/tier24_q8_4gpu_nsys_run.log"
grep -E 'ETS time|C_HAM|error|Error' "${LOGDIR}/tier24_q8_4gpu_nsys_run.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | tail -20 || true
# Extract top CUDA GPU kernels / memops from stats if present
if [[ -f "${NSYS_OUT}.nsys-rep" ]]; then
  "${NSYS}" stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
    "${NSYS_OUT}.nsys-rep" > "${LOGDIR}/tier24_q8_4gpu_nsys_stats.txt" 2>&1 || true
  head -80 "${LOGDIR}/tier24_q8_4gpu_nsys_stats.txt" || true
fi
echo TIER24_NSYS_Q8_4GPU_DONE
