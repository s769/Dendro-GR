#!/usr/bin/env bash
# Generic A/B on q8 meshprobe (np=4 default) + q1 smoke_cons bit-id check.
# usage: run_ab_env.sh LABEL "A_ENVS" "B_ENVS" [reps=2] [np=4]
#   envs are space-separated KEY=VAL strings, e.g. "DENDRO_BDY_STREAM=0" "DENDRO_BDY_STREAM=1"
set -uo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null
unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
LABEL="$1"; A="$2"; B="$3"; REPS="${4:-2}"; NP="${5:-4}"
BIN="${BINPATH:-${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA}"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/ab_${LABEL}.log"; : > "$LOG"
echo "host $(hostname) $(date -Is) $LABEL np=$NP bin=$BIN" | tee -a "$LOG"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"; SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
runq8() { # tag envs
  local d="${HOME}/Dendro-GR-runs/ab_${LABEL}_$1"; mkdir -p "$d"/{aeh,cp,dat,vtu}; cp "$PAR0" "$d/local.par.toml"
  python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
for k,v in [("BSSN_RK_TIME_END","0.5"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0"),("BSSN_TIME_STEP_OUTPUT_FREQ","1")]:
    t=re.sub(k+r" = [0-9.eE+-]+",k+" = "+v,t)
p.write_text(t)
PY
  ln -sfn "$SOL" "$d/mesh_q8_tpid_sol.bin"; cd "$d"
  mpirun -np $NP env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 $2 "$BIN" local.par.toml 1 > run.log 2>&1
  echo "$1 [$2] ETS=$(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time (max)' | tail -1 | awk '{print $NF}')" | tee -a "$LOG"
}
for i in $(seq 1 $REPS); do runq8 "A$i" "$A"; runq8 "B$i" "$B"; done
# bit-id: q1 smoke_cons, 1 GPU, compare Constraints output byte-for-byte
runq1() {
  local d="${HOME}/Dendro-GR-runs/ab_${LABEL}_q1_$1"; mkdir -p "$d"/{aeh,cp,dat,vtu}; cd "$d"
  ln -sfn "${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin" rit_q1_tpid_sol.bin
  mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 $2 "$BIN" "${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml" 1 > run.log 2>&1
}
if [[ -z "${SKIP_BITID:-}" ]]; then runq1 A "$A"; runq1 B "$B"; fi
DA="${HOME}/Dendro-GR-runs/ab_${LABEL}_q1_A/dat"; DB="${HOME}/Dendro-GR-runs/ab_${LABEL}_q1_B/dat"
for f in $( [[ -z "${SKIP_BITID:-}" ]] && echo bssn_profile_Constraints.dat bssn_profile_GW_l2_m2.dat ); do
  if cmp -s "$DA/$f" "$DB/$f"; then echo "BITID $f IDENTICAL" | tee -a "$LOG"; else echo "BITID $f DIFFER" | tee -a "$LOG"; fi
done
echo AB_DONE | tee -a "$LOG"
