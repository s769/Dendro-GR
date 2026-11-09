#!/usr/bin/env bash
# Tier 3.2: Richardson sample at q=4 — same pars except BSSN_MAXDEPTH / BH*_MAX_LEV.
# Record mesh per depth, short evolve, report C_HAM L2 as the observable pair.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q4.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/tpid_tier12/q4_a/mesh_q4_a_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier32_richardson_q4"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier32_richardson_q4.log"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
COARSE_DEPTH=13
FINE_DEPTH=14
TIME_END=0.27   # ~4 ETS at q4 meshprobe CFL

echo "host $(hostname) $(date -Is) tier32_richardson_q4" | tee "${LOG}"
[[ -x "${BIN}" ]] || { echo "missing ${BIN}"; exit 1; }
[[ -f "${PAR0}" && -f "${SOL}" ]] || { echo "missing par/sol"; exit 1; }
rm -rf "${OUT}"
mkdir -p "${OUT}"

mk_par() {
  local dest="$1" depth="$2"
  cp "${PAR0}" "${dest}"
  python3 - <<PY
from pathlib import Path
import re
p = Path("${dest}")
t = p.read_text()
t = re.sub(r"BSSN_MAXDEPTH = \d+", "BSSN_MAXDEPTH = ${depth}", t)
t = re.sub(r"BSSN_BH1_MAX_LEV = \d+", "BSSN_BH1_MAX_LEV = ${depth}", t)
t = re.sub(r"BSSN_BH2_MAX_LEV = \d+", "BSSN_BH2_MAX_LEV = ${depth}", t)
t = re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = ${TIME_END}", t)
t = re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t = re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t = re.sub(r'BSSN_PROFILE_FILE_PREFIX = "[^"]*"',
           'BSSN_PROFILE_FILE_PREFIX = "dat/bssn_profile"', t)
# GPU driver writes Constraints via write_vtu on GW extract cadence
if "BSSN_GW_EXTRACT_FREQ" in t:
    t = re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 1", t)
else:
    t += "\nBSSN_GW_EXTRACT_FREQ = 1\n"
p.write_text(t)
PY
}

run_one() {
  local tag="$1" depth="$2"
  local d="${OUT}/${tag}"
  local mesh="${OUT}/mesh_${tag}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu} "${mesh}"
  mk_par "${d}/local.par.toml" "${depth}"
  ln -sfn "${SOL}" "${d}/${PREFIX}_tpid_sol.bin"
  cd "${d}"
  echo "==== record+evolve ${tag} depth=${depth} ====" | tee -a "${LOG}"
  if ! mpirun -np 1 env -u CUDA_VISIBLE_DEVICES \
      DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_RECORD="${mesh}" \
      "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1; then
    echo "FAIL ${tag} exit=$?" | tee -a "${LOG}"
    tail -60 run.log | tee -a "${LOG}"
    return 1
  fi
  echo "==== ${tag} finished $(wc -c <run.log)B log ====" | tee -a "${LOG}"
  python3 - <<PY | tee -a "${LOG}"
from pathlib import Path
import re
rows=[]
cands=[Path("dat/bssn_profile_Constraints.dat"), Path("bssn_profile_Constraints.dat")]
p=next((c for c in cands if c.exists()), None)
if p is None:
    print("${tag}", "NO_CONSTRAINTS")
else:
    for l in p.read_text().splitlines():
        if not l.strip() or l.startswith("TimeStep") or "C_HAM" in l.split()[:3]:
            continue
        parts=l.split()
        try: rows.append(float(parts[2]))
        except (ValueError, IndexError): continue
    ham = rows[-1] if rows else float("nan")
    print("${tag}", "C_HAM", ham, "n", len(rows), "file", p)
    Path("ham.txt").write_text(str(ham))
# always record grid meta from run.log
text=Path("run.log").read_text(errors="ignore")
lms=re.findall(r"lmin:\s*(\d+)\s+lmax:(\d+)", text)
ets=re.findall(r"ETS time \(max\)\s*:\s*([0-9.eE+-]+)", text)
dxs=re.findall(r"dx:\s*([0-9.eE+-]+)", text)
meta={"lmax_after": int(lms[-1][1]) if lms else -1,
      "lmin_after": int(lms[-1][0]) if lms else -1,
      "ets": float(ets[-1]) if ets else float("nan"),
      "dx": float(dxs[-1]) if dxs else float("nan")}
print("${tag}", "GRID", meta)
Path("grid_meta.txt").write_text(repr(meta))
PY
}

run_one "coarse_d${COARSE_DEPTH}" "${COARSE_DEPTH}"
run_one "fine_d${FINE_DEPTH}" "${FINE_DEPTH}"

python3 - <<PY | tee -a "${LOG}"
from pathlib import Path
OUT=Path("${OUT}")
def load_ham(tag):
    p=OUT/tag/"ham.txt"
    return float(p.read_text()) if p.exists() else float("nan")
def load_meta(tag):
    p=OUT/tag/"grid_meta.txt"
    return eval(p.read_text()) if p.exists() else {}
hc=load_ham("coarse_d${COARSE_DEPTH}")
hf=load_ham("fine_d${FINE_DEPTH}")
mc=load_meta("coarse_d${COARSE_DEPTH}")
mf=load_meta("fine_d${FINE_DEPTH}")
print(f"GRID coarse(d=${COARSE_DEPTH})={mc}")
print(f"GRID fine(d=${FINE_DEPTH})={mf}")
import math
if not (math.isnan(hc) or math.isnan(hf)):
    r=2.0; p=2.0
    cont=(r**p * hf - hc)/(r**p - 1.0)
    print(f"C_HAM coarse={hc:.8e} fine={hf:.8e}")
    print(f"Richardson_p2 continuum≈{cont:.8e}  |fine-coarse|={abs(hf-hc):.8e}")
else:
    print("C_HAM unavailable — grid meta is the resolution-knob sample")
print("TIER32_RICHARDSON_Q4_DONE")
PY
