#!/usr/bin/env bash
# Accuracy-vs-cost study for the inverse problem: forward psi4 runs (production physics: init-mesh fix, SSL, CAHD, BH tracking)
# usage: run_resstudy.sh "name:cap:tol:gpu name:cap:tol:gpu ..."  (one single-rank process per GPU, run concurrently)
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
TEND=${TEND:-26}; OUT=$HOME/Dendro-GR-runs/resstudy; mkdir -p $OUT
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
RADII='[10.0, 12.0, 15.0, 18.0, 20.0, 25.0]'
for spec in $1; do
  IFS=: read name cap tol gpu <<< "$spec"
  d=$OUT/${name}_T${TEND}; rm -rf $d; mkdir -p $d/{aeh,cp,dat,vtu}; cp $HOME/Dendro-GR-runs/q1.smoke_cons.par.toml $d/local.par.toml
  python3 - <<PY
import re
from pathlib import Path
p=Path("$d/local.par.toml"); t=p.read_text()
def sub(k,v):
    global t
    t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0:
        m=re.search(r"^\[",t,flags=re.M); ins=k+" = "+v+"\n\n"
        t=(t[:m.start()]+ins+t[m.start():]) if m else t+"\n"+ins
for k,v in [("BSSN_MAXDEPTH","$cap"),("BSSN_WAVELET_TOL","$tol"),("BSSN_RK_TIME_END","float($TEND)".replace("float(","").replace(")","")+".0" if "." not in "$TEND" else "$TEND"),
            ("BSSN_REMESH_TEST_FREQ","50"),("BSSN_GW_EXTRACT_FREQ","8"),("BSSN_GW_RADAII","$RADII"),("BSSN_CAHD_C","0.06"),
            ("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]: sub(k,v)
p.write_text(t)
PY
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin
  ( cd $d; s=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 DENDRO_TIME_REMESH=1 $BIN local.par.toml 1 > run.log 2>&1
    echo "$name cap=$cap tol=$tol gpu=$gpu wall=$(( $(date +%s)-s ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1)" >> $OUT/summary_T${TEND}.txt ) &
done
wait
echo RESSTUDY_DONE >> $OUT/summary_T${TEND}.txt
