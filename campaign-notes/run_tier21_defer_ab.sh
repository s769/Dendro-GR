#!/usr/bin/env bash
# B1: PT_ASYNC vs PT_ASYNC_DEFER (deferred DD seconds) — ETS + C_HAM bit-id.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier21_defer_ab"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier21_defer_ab.log"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
TIME_END=0.25

echo "host $(hostname) $(date -Is) tier21_defer_ab" | tee "${LOG}"
rm -rf "${OUT}"; mkdir -p "${OUT}"

run_mode() {
  local mode="$1"  # ASYNC|DEFER
  local d="${OUT}/${mode}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${d}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = ${TIME_END}", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${d}/${PREFIX}_tpid_sol.bin"
  cd "${d}"
  local env_rhs=()
  case "${mode}" in
    ASYNC) env_rhs=(DENDRO_RHS_PT_ASYNC=1) ;;
    DEFER) env_rhs=(DENDRO_RHS_PT_ASYNC_DEFER=1) ;;
    *) echo "bad mode"; return 1 ;;
  esac
  echo "==== ${mode} ====" | tee -a "${LOG}"
  START=$(date +%s)
  if ! mpirun -np 1 --bind-to none env -u CUDA_VISIBLE_DEVICES "${env_rhs[@]}" \
      "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1; then
    echo "FAIL ${mode}" | tee -a "${LOG}"
    tail -40 run.log | tee -a "${LOG}"
    return 1
  fi
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
  HAM=$(grep -E 'Constraint C_HAM L2' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 || true)
  echo "${mode} wall=$((END-START))s ets=${ETS:-NA} ${HAM}" | tee -a "${LOG}"
}

run_mode ASYNC
run_mode DEFER
python3 - <<PY | tee -a "${LOG}"
from pathlib import Path
import re
OUT = Path("${OUT}")
def last_ham(mode):
    t = (OUT / mode / "run.log").read_text(errors="ignore")
    vals = []
    for line in t.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if "Constraint C_HAM L2" in line:
            m = re.search(r"C_HAM L2\s*:\s*\(([^\)]+)\)", line)
            if m:
                try:
                    vals.append(float(m.group(1).strip()))
                except Exception:
                    pass
    return vals
a = last_ham("ASYNC"); d = last_ham("DEFER")
print("ASYNC_n", len(a), "last", a[-1] if a else None)
print("DEFER_n", len(d), "last", d[-1] if d else None)
if a and d:
    da = a[-1] - d[-1]
    print(f"delta_last_C_HAM {da:.6e}")
    print("BITID", "PASS" if a[-1] == d[-1] else ("NEAR" if abs(da) < 1e-18 else "FAIL"))
print("TIER21_DEFER_AB_DONE")
PY
