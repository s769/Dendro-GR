#!/usr/bin/env bash
# full vs shell-only extraction: timers + psi4 identity (single-GPU each, run sequentially for clean timing)
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
for mode in full shell; do
  d=$HOME/Dendro-GR-runs/exab_$mode; mkdir -p $d/{aeh,cp,dat,vtu}; cp $HOME/Dendro-GR-runs/q1.smoke_cons.par.toml $d/local.par.toml
  python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+","BSSN_RK_TIME_END = 4.0",t)
t=re.sub(r"BSSN_REMESH_TEST_FREQ = [0-9]+","BSSN_REMESH_TEST_FREQ = 20",t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+","BSSN_GW_EXTRACT_FREQ = 4",t)
t=re.sub(r"BSSN_GW_RADAII = \[[^\]]*\]","BSSN_GW_RADAII = [10.0,12.0,15.0,18.0,20.0,25.0]",t)
p.write_text(t)
PY
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
  E="DENDRO_GW_SHELL_ONLY=0"; [[ $mode == shell ]] && E="DENDRO_GW_SHELL_ONLY=1"
  CUDA_VISIBLE_DEVICES=0 mpirun -np 1 env DENDRO_RHS_PT_ASYNC=1 DENDRO_TIME_EXTRACT=1 $E $BIN local.par.toml 1 > run.log 2>&1
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
for lm in ['l2_m2','l3_m2','l4_m4']:
    a=load('exab_full',lm); b=load('exab_shell',lm); n=min(len(a),len(b))
    print(lm,'rows',len(a),len(b),'max rel diff',np.max(abs(a[:n]-b[:n])/np.maximum(abs(a[:n]),1e-300)),'max abs',np.max(abs(a[:n]-b[:n])))
PY
echo EXAB_DONE
