#!/usr/bin/env bash
# checkpoint/restart test: q1 cap 13 T=14, CHECKPT_FREQ=200 steps; kill at ~step KILLSTEP; restart with RESTORE=1; compare with an uninterrupted run
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA; d=$HOME/Dendro-GR-runs/ckpt_test; rm -rf $d; mkdir -p $d/{aeh,bah,cp,dat,vtu}; cd $d
cp $HOME/Dendro-GR/BSSN_GR/pars/q1.par.toml local.par.toml; ln -sf $HOME/Dendro-GR-runs/resstudy2/tp/tp_q001_tpid_sol.bin .
setp() { python3 - "$1" <<'PY'
import re,sys
from pathlib import Path
p=Path("local.par.toml"); t=p.read_text(); restore=sys.argv[1]
def sub(k,v):
    global t
    t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0:
        m=re.search(r"^\[",t,flags=re.M); ins=k+" = "+v+"\n\n"; t=(t[:m.start()]+ins+t[m.start():]) if m else t+"\n"+ins
for k,v in [("BSSN_RESTORE_SOLVER",restore),("BSSN_MAXDEPTH","13"),("BSSN_BH1_MAX_LEV","13"),("BSSN_BH2_MAX_LEV","13"),("BSSN_RK_TIME_END","14.0"),
            ("BSSN_GW_EXTRACT_FREQ","32"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","200")]: sub(k,v)
p.write_text(t)
PY
}
setp 0
E="env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1"
CUDA_VISIBLE_DEVICES=${GPU:-0} mpirun -np 1 --bind-to none $E $BIN local.par.toml 1 > run1.log 2>&1 &
PID=$!
until sed 's/\x1b\[[0-9;]*m//g' run1.log | grep -q "Executing step :  *6[0-9][0-9]\b"; do sleep 3; done
pkill -P $PID; kill $PID 2>/dev/null; sleep 3; pkill -u $USER -f "bssnSolverCUDA local.par.toml" 2>/dev/null
echo "killed after: $(sed 's/\x1b\[[0-9;]*m//g' run1.log | grep -o 'Executing step : *[0-9]*' | tail -1)" > status.txt
ls cp >> status.txt
setp 1
CUDA_VISIBLE_DEVICES=${GPU:-0} mpirun -np 1 --bind-to none $E $BIN local.par.toml 1 > run2.log 2>&1
echo "run2: $(sed 's/\x1b\[[0-9;]*m//g' run2.log | grep -E 'ETS time|restore|Restor' | tail -3 | tr '\n' '|')" >> status.txt
echo CKPT_DONE >> status.txt
