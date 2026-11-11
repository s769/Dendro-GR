#!/usr/bin/env bash
# CFL-factor test: q1 production adaptivity, cap CAP, T=TEND; one run per CFL on GPUs 0..; compare psi4_22 vs CFL=0.25 with analyze_res.py
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
CAP=${CAP:-12}; TEND=${TEND:-70}; RMFS="${RMFS:-50 100 200}"; GPU0=${GPU0:-0}
OUT=$HOME/Dendro-GR-runs/resstudy2; BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
TPB=$OUT/tp/tp_q001_tpid_sol.bin; i=0
for r in $RMFS; do c=0.25
  d=$OUT/rmf${r}c${CAP}_T${TEND}; rm -rf $d; mkdir -p $d/{aeh,bah,cp,dat,vtu}; cp $HOME/Dendro-GR/BSSN_GR/pars/q1.par.toml $d/local.par.toml; ln -sfn $TPB $d/tp_q001_tpid_sol.bin
  gwf=$(python3 -c "print(max(1,round(8*2**($CAP-11)*0.25/$c)))")
  rmf=$r
  python3 - <<PY
import re
from pathlib import Path
p=Path("$d/local.par.toml"); t=p.read_text()
def sub(k,v):
    global t
    t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0:
        m=re.search(r"^\[",t,flags=re.M); ins=k+" = "+v+"\n\n"; t=(t[:m.start()]+ins+t[m.start():]) if m else t+"\n"+ins
for k,v in [("BSSN_RESTORE_SOLVER","0"),("BSSN_MAXDEPTH","$CAP"),("BSSN_BH1_MAX_LEV","$CAP"),("BSSN_BH2_MAX_LEV","$CAP"),("BSSN_RK_TIME_END","$TEND.0"),
            ("BSSN_CFL_FACTOR","$c"),("BSSN_REMESH_TEST_FREQ","$rmf"),("BSSN_GW_EXTRACT_FREQ","$gwf"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]: sub(k,v)
p.write_text(t)
PY
  ( cd $d; s=$(date +%s); CUDA_VISIBLE_DEVICES=$((GPU0+i)) mpirun -np 1 env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 $BIN local.par.toml 1 > run.log 2>&1
    echo "rmf$r wall=$(( $(date +%s)-s ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1)" >> $OUT/summary_rmf.txt ) &
  i=$((i+1))
done; wait; echo RMF_DONE >> $OUT/summary_rmf.txt
