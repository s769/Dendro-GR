#!/usr/bin/env bash
# Longer 1.5 FD: same frozen mesh, TIME_END=1.0 (~15 steps), fewer eps for wall-clock.
set -euo pipefail
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
REC="${HOME}/Dendro-GR-runs/tier14/mesh_rec"
OUT="${HOME}/Dendro-GR-runs/tier15_gw_fd_long"
rm -rf "${OUT}"
mkdir -p "${OUT}"

run_mass() {
  local tag="$1" mass="$2"
  local d="${OUT}/${tag}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p = Path("${d}/local.par.toml")
t = p.read_text()
t = re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>${mass}", t, count=1)
t = re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 1.0", t)
p.write_text(t)
PY
  ln -sfn "${TPID}" "${d}/rit_q1_tpid_sol.bin"
  cd "${d}"
  env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY="${REC}" \
    "${BIN}" "${d}/local.par.toml" 1 > run.log 2>&1
  python3 - <<PY
from pathlib import Path
import math, re
gw = Path("dat/bssn_profile_GW_L2.dat")
ham = Path("dat/bssn_profile_Constraints.dat")
val=None; src=None
if gw.exists():
    lines=[l for l in gw.read_text().splitlines() if l.strip()]
    if len(lines)>1:
        comps=re.findall(r"\(([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?),([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)", lines[-1])
        if comps:
            re0,im0=map(float,comps[0]); val=math.hypot(re0,im0); src="GW_L2_r50_mag"
rows=[]
for i,l in enumerate(ham.read_text().splitlines()):
    if i==0 or not l.strip(): continue
    rows.append(l.split())
hamv=float(rows[-1][2])
Path("obs.txt").write_text(f"{val if val is not None else hamv}\n{src or 'C_HAM'}\n{hamv}\n")
print(f"${tag}\tmass=${mass}\t{(src or 'C_HAM')}={(val if val is not None else hamv):.16e}\tC_HAM={hamv:.16e}\tsteps={len(rows)}")
PY
}

M0=0.4824
run_mass m0 "${M0}"
for eps in 3e-4 1e-3 3e-3; do
  mp=$(python3 -c "print(f'{${M0}+float(\"${eps}\"):.16g}')")
  mm=$(python3 -c "print(f'{${M0}-float(\"${eps}\"):.16g}')")
  run_mass "p_${eps}" "${mp}"
  run_mass "m_${eps}" "${mm}"
done

python3 - <<'PY'
from pathlib import Path
OUT=Path("/home1/08435/srvenkat/Dendro-GR-runs/tier15_gw_fd_long")
def load(tag):
    lines=(OUT/tag/"obs.txt").read_text().splitlines()
    return float(lines[0]), lines[1], float(lines[2])
m0,src,h0=load("m0")
print(f"LONG evolve TIME_END=1.0  observable={src}  m0={m0}  C_HAM0={h0}")
print("eps\tFD_central_obs\tFD_central_HAM")
for e in ["3e-4","1e-3","3e-3"]:
    eps=float(e)
    hp,_,hhp=load(f"p_{e}"); hm,_,hhm=load(f"m_{e}")
    print(f"{e}\t{(hp-hm)/(2*eps):.8e}\t{(hhp-hhm)/(2*eps):.8e}")
print("TIER15_GW_FD_LONG_DONE")
PY
