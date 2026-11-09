#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
REC="${HOME}/Dendro-GR-runs/tier14/mesh_rec"
OUT="${HOME}/Dendro-GR-runs/tier14/nf_pert"
rm -rf "${OUT}"
mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
cp "${PAR}" "${OUT}/local.par.toml"
python3 - <<PY
from pathlib import Path
import re
p = Path("${OUT}/local.par.toml")
t = p.read_text()
t = re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>0.4830", t, count=1)
t = re.sub(r"(BSSN_BH2 = \{ MASS = )[0-9.eE+-]+", r"\g<1>0.4824", t, count=1)
p.write_text(t)
print("BH1", re.search(r"BSSN_BH1 = \{ MASS = ([^,]+)", t).group(1))
print("BH2", re.search(r"BSSN_BH2 = \{ MASS = ([^,]+)", t).group(1))
PY
ln -sfn "${TPID}" "${OUT}/rit_q1_tpid_sol.bin"
cd "${OUT}"
env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY="${REC}" \
  "${BIN}" "${OUT}/local.par.toml" 1 > run.log 2>&1
grep -E 'resolved inversion|m\+=|Frozen-mesh|Grid Info|C_HAM|Remesh|NOTE: BSSN' run.log \
  | sed 's/\x1b\[[0-9;]*m//g' | head -50
echo NF_DONE
