#!/usr/bin/env bash
# A/B: current binary (lb=1 after rebuild) ETS on q8 meshprobe.
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier21_lb1_ab"
LOG="${HOME}/Dendro-GR-notes/tier0/tier21_lb1_ab.log"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
rm -rf "${OUT}"; mkdir -p "${OUT}/ASYNC"/{aeh,cp,dat,vtu}
cp "${PAR0}" "${OUT}/ASYNC/local.par.toml"
python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/ASYNC/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.25", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
p.write_text(t)
PY
ln -sfn "${SOL}" "${OUT}/ASYNC/${PREFIX}_tpid_sol.bin"
cd "${OUT}/ASYNC"
echo "host $(hostname) $(date -Is) lb1_ab" | tee "${LOG}"
START=$(date +%s)
mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" local.par.toml 1 >run.log 2>&1
END=$(date +%s)
ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
echo "ASYNC_lb1 wall=$((END-START))s ets=${ETS:-NA} (baseline lb3 was ~2.45)" | tee -a "${LOG}"
echo LB1_AB_DONE | tee -a "${LOG}"
