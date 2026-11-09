#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
for mode in m2 m1000; do
  d=$HOME/Dendro-GR-runs/exab_$mode; mkdir -p $d/{aeh,cp,dat,vtu}; cp $HOME/Dendro-GR-runs/exab_full/local.par.toml $d/local.par.toml
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
  M=2; [[ $mode == m1000 ]] && M=1000
  CUDA_VISIBLE_DEVICES=0 mpirun -np 1 env DENDRO_RHS_PT_ASYNC=1 DENDRO_TIME_EXTRACT=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_GW_SHELL_MARGIN=$M $BIN local.par.toml 1 > run.log 2>&1
  echo "== $mode: $(grep extract-time run.log | tail -1)"; sed 's/\x1b\[[0-9;]*m//g' run.log | grep "ETS time (max)"
done
cd $HOME/Dendro-GR-runs
python3 - <<'PY'
import re,numpy as np
def load(d,lm):
    z=[]
    for l in open(f'{d}/dat/bssn_profile_GW_{lm}.dat').read().splitlines()[1:]:
        p=l.split()
        if len(p)>2: z.append([complex(*map(float,re.findall(r'[-+0-9.eE]+',c))) for c in p[2:]])
    return np.array(z)
for m in ['m2','m1000']:
  for lm in ['l2_m2','l3_m2']:
    a=load('exab_full',lm); b=load('exab_'+m,lm); n=min(len(a),len(b))
    print(m,lm,'max rel diff',np.max(abs(a[:n]-b[:n])/np.maximum(abs(a[:n]),1e-300)))
PY
echo MARGIN_DONE
