#!/usr/bin/env bash
# Tier 2 hang-path: NCU Occupancy/LaunchStats on __unzip_cg1__2 (q8, 1 GPU, short).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

NCU="${NCU:-/home1/apps/nvidia/Linux_aarch64/26.5/profilers/13.2/Nsight_Compute/ncu}"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/q8_ncu_hang"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier2_ncu_hang.log"
REP="${HOME}/Dendro-GR/campaign-notes/tier0/unzip_hang_occ"

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
p.write_text(t)
PY
ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
cd "${OUT}"
: > "${LOG}"
echo "host $(hostname) $(date -Is) ncu_hang NCU=${NCU}" | tee -a "${LOG}"
if [[ ! -x "${NCU}" ]]; then
  echo "ncu missing at ${NCU}" | tee -a "${LOG}"
  exit 1
fi

# Profile a few unzip launches only (kernel name filter). Hang-majority mesh.
# SPLIT=2 so hang CTA list is what we care about; still same kernel symbol.
"${NCU}" --set full --kernel-name-base demangled \
  --kernel-name regex:unzip_cg1 \
  --launch-skip 2 --launch-count 4 \
  --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
  -o "${REP}" \
  mpirun -np 1 --bind-to none env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    DENDRO_UNZIP_SPLIT=2 \
    "${BIN}" "${OUT}/local.par.toml" 1 \
  2>&1 | tee -a "${LOG}"

"${NCU}" --import "${REP}.ncu-rep" 2>/dev/null | tee -a "${LOG}" || true
echo TIER2_NCU_HANG_DONE | tee -a "${LOG}"
