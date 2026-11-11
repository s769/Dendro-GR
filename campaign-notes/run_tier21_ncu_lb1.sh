#!/usr/bin/env bash
# NCU regs/local-mem on PT_ASYNC after launch_bounds(343,1).
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

NCU="${NCU:-/home1/apps/nvidia/Linux_aarch64/26.5/profilers/13.2/Nsight_Compute/ncu}"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/q8_ncu_lb1"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier21_ncu_lb1.log"
REP="${HOME}/Dendro-GR/campaign-notes/tier0/rhs_pt_async_lb1"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')

mkdir -p "${OUT}"
cp "${PAR0}" "${OUT}/local.par.toml"
python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.25", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
p.write_text(t)
PY
ln -sfn "${SOL}" "${OUT}/${PREFIX}_tpid_sol.bin"
cd "${OUT}"
: > "${LOG}"
echo "host $(hostname) $(date -Is) ncu_lb1 NCU=${NCU}" | tee -a "${LOG}"

# Match templated demangled name; skip remesh/unzip warm-up launches.
"${NCU}" --kernel-name-base demangled \
  --kernel-name regex:eval_rhs3_pt_async \
  --launch-skip 8 --launch-count 4 \
  --metrics launch__registers_per_thread,launch__shared_mem_per_block_dynamic,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum \
  -o "${REP}" --force-overwrite \
  mpirun -np 1 --bind-to none env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    "${BIN}" "${OUT}/local.par.toml" 1 >ncu_run.log 2>&1 \
  || { echo "ncu/mpirun failed"; tail -30 ncu_run.log | tee -a "${LOG}"; }

if [[ -f "${REP}.ncu-rep" ]]; then
  "${NCU}" --import "${REP}.ncu-rep" --print-summary per-kernel 2>&1 | tee -a "${LOG}"
  "${NCU}" --import "${REP}.ncu-rep" --page raw --csv 2>/dev/null | head -80 | tee -a "${LOG}" || true
else
  echo "NO_REPORT" | tee -a "${LOG}"
  grep -E 'Available Kernels|PROF|WARNING|ERROR' ncu_run.log | tee -a "${LOG}" || true
fi
echo NCU_LB1_DONE | tee -a "${LOG}"
