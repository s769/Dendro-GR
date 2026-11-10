#!/usr/bin/env bash
# Production-depth q8 (pars/q8.par.toml: maxdepth 17, remesh/50, GW/100, ckpt/100) with TPID grid matched to the reusable meshprobe solution.
# usage: run_prod_q8.sh TAG TEND "ENVS" [nsys=0]
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
TAG="$1"; TEND="$2"; ENVS="$3"; NSYS="${4:-0}"; OMPT="${OMPT:-1}"
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
d=$HOME/Dendro-GR-runs/prod_$TAG; mkdir -p $d; find $d -mindepth 1 -delete 2>/dev/null; mkdir -p $d/{aeh,cp,dat,vtu}
cp $HOME/Dendro-GR/BSSN_GR/pars/q8.par.toml $d/local.par.toml
python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
def sub(k,v):
    global t; t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M); assert n>=1,k
sub("BSSN_RESTORE_SOLVER","0"); sub("BSSN_IO_OUTPUT_FREQ","0"); sub("BSSN_RK_TIME_END","$TEND")
sub("TPID_NPOINTS_A","60"); sub("TPID_NPOINTS_B","60"); sub("TPID_NPOINTS_PHI","60")
sub("TPID_TARGET_M_PLUS","0.11111111111111111"); sub("TPID_TARGET_M_MINUS","0.88888888888888888")
sub("TPID_FILEPREFIX",'"mesh_q8"')
import os
for kv in [x for x in os.environ.get("PAR_OVR","").split(",") if x]:
    k,v=kv.split("="); sub(k,v)
p.write_text(t)
PY
ln -sfn $HOME/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin $d/mesh_q8_tpid_sol.bin; cd $d
CMD="mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=$OMPT env OMP_NUM_THREADS=$OMPT DENDRO_RHS_PT_ASYNC=1 DENDRO_TIME_EXTRACT=1 $ENVS $BIN local.par.toml 1"
if [[ "$NSYS" == 1 ]]; then nsys profile -o $d/rep --force-overwrite=true --trace=cuda,nvtx --stats=false $CMD > run.log 2>&1; else $CMD 2>&1 | python3 -u -c "import sys,time
for l in sys.stdin: sys.stdout.write('%.3f %s'%(time.time(),l))" > run.log; fi
sed 's/\x1b\[[0-9;]*m//g' run.log | grep -E "lmin|lmax:|dt:|ETS time|Executing step|extract-time|Remesh is triggered|ERROR|Error" | awk '/Executing step/{n++; if(n%20==1)print; next} {print}' | tail -25
echo PROD_DONE
