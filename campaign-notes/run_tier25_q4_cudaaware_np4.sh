#!/usr/bin/env bash
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile 2>/dev/null || true
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q4.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/tpid_tier12/q4_a/mesh_q4_a_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/q4_cudaaware_np4"
LOG="${HOME}/Dendro-GR-notes/tier0/tier25_q4_cudaaware_np4.log"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
rm -rf "${OUT}"
mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
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
START=$(date +%s)
mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
END=$(date +%s)
ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
echo "q4_cudaaware_np4 wall_sec=$((END-START)) ets=${ETS:-NA} (host-staged was 1.453)" | tee -a "${HOME}/Dendro-GR-notes/tier0/tier25_scale_q4.log"
