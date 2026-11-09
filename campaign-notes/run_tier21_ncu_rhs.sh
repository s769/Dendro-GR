#!/usr/bin/env bash
# Tier 2.1 prerequisite: NCU LaunchStats/Occupancy on eval_rhs3_pt_async (regs/spills).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

NCU="${NCU:-/home1/apps/nvidia/Linux_aarch64/26.5/profilers/13.2/Nsight_Compute/ncu}"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/q8_ncu_rhs"
LOG="${HOME}/Dendro-GR-notes/tier0/tier21_ncu_rhs.log"
REP="${HOME}/Dendro-GR-notes/tier0/rhs_pt_async_occ"

rm -rf "${OUT}"
mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
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
ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
cd "${OUT}"
: > "${LOG}"
echo "host $(hostname) $(date -Is) ncu_rhs NCU=${NCU}" | tee -a "${LOG}"
[[ -x "${NCU}" ]] || { echo "ncu missing"; exit 1; }

"${NCU}" --set full --kernel-name-base demangled \
  --kernel-name regex:eval_rhs3_pt_async \
  --launch-skip 4 --launch-count 6 \
  --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  -o "${REP}" \
  mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    "${BIN}" "${OUT}/local.par.toml" 1 \
  2>&1 | tee -a "${LOG}"

echo TIER21_NCU_RHS_DONE | tee -a "${LOG}"
# extract key lines for the note
grep -E 'Registers Per Thread|Local Memory Spill|Theoretical Occupancy|Achieved Occupancy|Memory Throughput|Compute \(SM\) Throughput|eval_rhs3_pt_async' \
  "${LOG}" | tee -a "${LOG}.summary" || true
