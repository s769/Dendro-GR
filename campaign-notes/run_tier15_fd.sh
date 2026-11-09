#!/usr/bin/env bash
# Tier 1.5 smoke: FD d(C_HAM_final)/d(m+) on frozen mesh + frozen TP sol.
# Not a full waveform FD at two resolutions — first plateau scout.
set -euo pipefail
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
REC="${HOME}/Dendro-GR-runs/tier14/mesh_rec"
OUT="${HOME}/Dendro-GR-runs/tier15_fd"
rm -rf "${OUT}"
mkdir -p "${OUT}"

run_mass() {
  local tag="$1" mass="$2"
  local d="${OUT}/${tag}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${PAR}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p = Path("${d}/local.par.toml")
t = p.read_text()
t = re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>${mass}", t, count=1)
p.write_text(t)
PY
  ln -sfn "${TPID}" "${d}/rit_q1_tpid_sol.bin"
  cd "${d}"
  env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY="${REC}" \
    "${BIN}" "${d}/local.par.toml" 1 > run.log 2>&1
  # last C_HAM from Constraints.dat
  python3 - <<PY
from pathlib import Path
rows=[]
for i,l in enumerate(Path("dat/bssn_profile_Constraints.dat").read_text().splitlines()):
    if i==0 or not l.strip(): continue
    rows.append(l.split())
ham=float(rows[-1][2])
print(f"${tag}\tmass=${mass}\tC_HAM_final={ham:.16e}\tsteps={len(rows)}")
Path("ham.txt").write_text(str(ham))
PY
}

M0=0.4824
run_mass m0 "${M0}"
for eps in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  # python for mass arithmetic
  mp=$(python3 -c "print(f'{${M0}+${eps}:.16g}')")
  mm=$(python3 -c "print(f'{${M0}-${eps}:.16g}')")
  run_mass "p_${eps}" "${mp}"
  run_mass "m_${eps}" "${mm}"
done

python3 - <<'PY'
from pathlib import Path
import math
OUT=Path("/home1/08435/srvenkat/Dendro-GR-runs/tier15_fd")
m0=float((OUT/"m0/ham.txt").read_text())
print("eps\tFD_central\tFD_forward")
for eps in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    tag=f"{eps:g}" if False else None
    # directory names use literal eps strings from bash
    pass
eps_list=["1e-4","3e-4","1e-3","3e-3","1e-2"]
print("eps\tFD_central\tFD_forward")
for e in eps_list:
    eps=float(e)
    hp=float((OUT/f"p_{e}/ham.txt").read_text())
    hm=float((OUT/f"m_{e}/ham.txt").read_text())
    fc=(hp-hm)/(2*eps)
    ff=(hp-m0)/eps
    print(f"{e}\t{fc:.8e}\t{ff:.8e}")
print("TIER15_FD_DONE")
PY
