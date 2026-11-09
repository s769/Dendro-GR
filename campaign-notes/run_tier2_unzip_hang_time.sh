#!/usr/bin/env bash
# Measure hang vs nh unzip kernel time (DENDRO_UNZIP_SPLIT=2 TIMING=1).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/q8_unzip_hang_time"
LOG="${HOME}/Dendro-GR-notes/tier0/tier2_unzip_hang_time.log"
rm -rf "${OUT}"; mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
cp "${PAR0}" "${OUT}/local.par.toml"
python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.5", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
p.write_text(t)
PY
ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
cd "${OUT}"
echo "host $(hostname) $(date -Is) hang_time" | tee "${LOG}"
mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  DENDRO_UNZIP_SPLIT=2 DENDRO_UNZIP_TIMING=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 >> "${LOG}" 2>&1 || true
grep -E 'unzip-time|ETS time|unzip-split' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tee -a "${LOG}.summary"
echo TIER2_UNZIP_HANG_TIME_DONE | tee -a "${LOG}.summary"
