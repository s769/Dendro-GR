#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
cfg() { # tag omp shell cache
  d=$HOME/Dendro-GR-runs/ex3_$1; mkdir -p $d; find $d -mindepth 1 -delete 2>/dev/null; mkdir -p $d/{aeh,cp,dat,vtu}
  cp $HOME/Dendro-GR-runs/exab_full/local.par.toml $d/local.par.toml
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
  CUDA_VISIBLE_DEVICES=0 mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=$2 env OMP_NUM_THREADS=$2 DENDRO_RHS_PT_ASYNC=1 DENDRO_TIME_EXTRACT=1 DENDRO_GW_SHELL_ONLY=$3 DENDRO_GW_CACHE_SAMPLER=$4 $BIN local.par.toml 1 > run.log 2>&1
  echo "== $1 (OMP=$2 shell=$3 cache=$4): $(grep extract-time run.log | tail -1 | cut -c1-175) | $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time (max)')"
}
cfg base 1 0 0
cfg cache 4 0 1
cfg shellcache 4 1 1
cfg shellcache8 8 1 1
cd $HOME/Dendro-GR-runs
python3 - <<'PY'
import re,numpy as np
def load(d,lm):
    z=[]
    for l in open(f'{d}/dat/bssn_profile_GW_{lm}.dat').read().splitlines()[1:]:
        p=l.split()
        if len(p)>2: z.append([complex(*map(float,re.findall(r'[-+0-9.eE]+',c))) for c in p[2:]])
    return np.array(z)
for m in ['cache','shellcache','shellcache8']:
  for lm in ['l2_m2','l3_m2','l4_m4']:
    a=load('ex3_base',lm); b=load('ex3_'+m,lm); n=min(len(a),len(b))
    print(m,lm,'rows',len(a),len(b),'max rel diff %.3e'%np.max(abs(a[:n]-b[:n])/np.maximum(abs(a[:n]),1e-300)))
PY
echo EX3_DONE
