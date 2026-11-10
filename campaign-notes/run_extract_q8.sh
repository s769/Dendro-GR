#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
cfg() { # tag omp shell cache
  d=$HOME/Dendro-GR-runs/q8x_$1; mkdir -p $d; find $d -mindepth 1 -delete 2>/dev/null; mkdir -p $d/{aeh,cp,dat,vtu}
  cp $HOME/Dendro-GR-runs/q8.meshprobe.par.toml $d/local.par.toml
  python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
for k,v in [("BSSN_RK_TIME_END","0.5"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0"),("BSSN_GW_EXTRACT_FREQ","4"),("BSSN_REMESH_TEST_FREQ","20")]:
    t=re.sub(k+r" = [0-9.eE+-]+",k+" = "+v,t)
p.write_text(t)
PY
  ln -sfn $HOME/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin $d/mesh_q8_tpid_sol.bin; cd $d
  mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=$2 env OMP_NUM_THREADS=$2 DENDRO_RHS_PT_ASYNC=1 DENDRO_TIME_EXTRACT=1 DENDRO_GW_SHELL_ONLY=$3 DENDRO_GW_CACHE_SAMPLER=$4 $BIN local.par.toml 1 > run.log 2>&1
  echo "== $1 (OMP=$2 shell=$3 cache=$4): $(grep extract-time run.log | tail -1 | cut -c1-175) | $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time (max)')"
}
cfg base 1 0 0
cfg new 8 1 1
cd $HOME/Dendro-GR-runs
python3 - <<'PY'
import re,numpy as np
def load(d,lm):
    z=[]
    for l in open(f'{d}/bssn_profile_GW_{lm}.dat').read().splitlines()[1:]:
        p=l.split()
        if len(p)>2: z.append([complex(*map(float,re.findall(r'[-+0-9.eE]+',c))) for c in p[2:]])
    return np.array(z)
for lm in ['l2_m2','l2_m1','l3_m2','l4_m4']:
    try:
        a=load('q8x_base',lm); b=load('q8x_new',lm); n=min(len(a),len(b))
        print(lm,'rows',len(a),len(b),'max rel diff %.3e'%np.max(abs(a[:n]-b[:n])/np.maximum(abs(a[:n]),1e-300)),' |a|max %.2e'%abs(a).max())
    except Exception as e: print(lm,'ERR',e)
PY
echo Q8X_DONE
