#!/usr/bin/env bash
# TwoPunctures solve: cold vs warm start (DENDRO_TPID_WARM_START) and thread scaling. q1 smoke setup, 60^3 spectral grid.
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
TPID=$DENDRO_BUILD/BSSN_GR/tpid; BASE=$HOME/Dendro-GR-runs/tpwarm; mkdir -p $BASE
run() { # tag mass nthreads [warm_sol]
  d=$BASE/$1; mkdir -p $d/{aeh,dat}; find $d -mindepth 1 -maxdepth 1 -name '*.bin' -delete 2>/dev/null; cp $HOME/Dendro-GR-runs/q1.smoke_cons.par.toml $d/local.par.toml
  python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
t=re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>$2", t, count=1)
t=re.sub(r'TPID_FILEPREFIX = "[^"]*"','TPID_FILEPREFIX = "tp_$1"',t)
p.write_text(t)
PY
  cd $d; S=$(date +%s)
  if [[ -n "${4:-}" ]]; then DENDRO_TPID_WARM_START=$4 $TPID local.par.toml $3 > tpid.log 2>&1; else $TPID local.par.toml $3 > tpid.log 2>&1; fi
  echo "$1 mass=$2 threads=$3 warm=${4:+yes} wall=$(( $(date +%s)-S ))s $(grep -c 'Newton: it=' tpid.log) Newton lines; $(grep 'TP solver time' tpid.log | tail -1); $(grep -E 'Newton: it=0' tpid.log | tail -1); last: $(grep -E 'Newton: it=' tpid.log | tail -1); $(grep 'warm start' tpid.log | head -1)"
}
run fid 0.4824 36
FID=$BASE/fid/tp_fid_tpid_sol.bin
run pert_cold36 0.4834 36
run pert_warm36 0.4834 36 $FID
run pert_cold8 0.4834 8
run pert_warm8 0.4834 8 $FID
python3 - <<'PY'
import numpy as np,os
B=os.path.expanduser('~/Dendro-GR-runs/tpwarm')
def v0(tag):
    f=f'{B}/{tag}/tp_{tag}_tpid_sol.bin'
    with open(f,'rb') as fh:
        n=np.fromfile(fh,dtype=np.int32,count=1)[0]; np.fromfile(fh,dtype=np.float64,count=n-1)
        np.fromfile(fh,dtype=np.float64,count=10*n); return np.fromfile(fh,dtype=np.float64,count=n)
a=v0('pert_cold36'); b=v0('pert_warm36'); c=v0('pert_cold8'); d=v0('pert_warm8')
print('max|v_cold36 - v_warm36| %.3e (max|v| %.3e); cold8 vs cold36 %.3e; warm8 vs warm36 %.3e'%(abs(a-b).max(),abs(a).max(),abs(c-a).max(),abs(d-b).max()))
PY
echo TPWARM_DONE
