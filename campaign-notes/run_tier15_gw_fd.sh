#!/usr/bin/env bash
# Tier 1.5: FD of GW L2 (psi4 proxy) w.r.t. m+ on frozen mesh, 1 GPU.
set -euo pipefail
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
REC="${HOME}/Dendro-GR-runs/tier14/mesh_rec"
OUT="${HOME}/Dendro-GR-runs/tier15_gw_fd"
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
  python3 - <<PY
from pathlib import Path
import math, re
# GW_L2 lines: step t (re,im)_r50 (re,im)_r60 ...
gw = Path("dat/bssn_profile_GW_L2.dat")
ham = Path("dat/bssn_profile_Constraints.dat")
val = None
src = None
if gw.exists():
    lines=[l for l in gw.read_text().splitlines() if l.strip()]
    if len(lines) > 1:
        last=lines[-1]
        comps=re.findall(r"\(([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?),([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)", last)
        if comps:
            re0, im0 = map(float, comps[0])  # innermost radius
            val = math.hypot(re0, im0)
            src = "GW_L2_r50_mag"
if val is None:
    rows=[]
    for i,l in enumerate(ham.read_text().splitlines()):
        if i==0 or not l.strip(): continue
        rows.append(l.split())
    val=float(rows[-1][2]); src="C_HAM"
print(f"${tag}\tmass=${mass}\t{src}={val:.16e}")
Path("obs.txt").write_text(f"{val}\n{src}\n")
PY
}

M0=0.4824
run_mass m0 "${M0}"
for eps in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  mp=$(python3 -c "print(f'{${M0}+float(\"${eps}\"):.16g}')")
  mm=$(python3 -c "print(f'{${M0}-float(\"${eps}\"):.16g}')")
  run_mass "p_${eps}" "${mp}"
  run_mass "m_${eps}" "${mm}"
done

python3 - <<'PY'
from pathlib import Path
OUT=Path("/home1/08435/srvenkat/Dendro-GR-runs/tier15_gw_fd")
m0=float((OUT/"m0/obs.txt").read_text().splitlines()[0])
src=(OUT/"m0/obs.txt").read_text().splitlines()[1]
print(f"observable={src}  m0_val={m0}")
print("eps\tFD_central\tFD_forward")
for e in ["1e-4","3e-4","1e-3","3e-3","1e-2"]:
    eps=float(e)
    hp=float((OUT/f"p_{e}/obs.txt").read_text().splitlines()[0])
    hm=float((OUT/f"m_{e}/obs.txt").read_text().splitlines()[0])
    print(f"{e}\t{(hp-hm)/(2*eps):.8e}\t{(hp-m0)/eps:.8e}")
print("TIER15_GW_FD_DONE")
PY
