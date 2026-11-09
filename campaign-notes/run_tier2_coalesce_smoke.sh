#!/usr/bin/env bash
# Queue #7: hang gather unique-CG stats (compression headroom for coalesce plan).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/q8_gather_stats"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier2_gather_stats.log"
rm -rf "${OUT}"; mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
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
echo "host $(hostname) $(date -Is) gather_stats" | tee -a "${LOG}"
mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  DENDRO_UNZIP_SPLIT=2 DENDRO_UNZIP_GATHER_STATS=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 >> "${LOG}" 2>&1 || true
grep -E 'unzip-gather-stats|unzip-split|ETS time' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tee "${LOG}.summary"
echo TIER2_GATHER_STATS_DONE | tee -a "${LOG}.summary"
