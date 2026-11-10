#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null; unset CUDA_VISIBLE_DEVICES
rm -f $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | tail -1 ) > tier0/paramcheck_build.log 2>&1
S=$HOME/Dendro-GR-runs/statecmp_cahd_s0_rk2/gpu
for V in ok bad_lambda bad_eta; do
  d=$HOME/Dendro-GR-runs/paramcheck_$V; rm -rf $d; mkdir -p $d/{aeh,cp,dat,vtu}; cp $S/local.par.toml $d/
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
  sed -i -E "s/^BSSN_RK_TIME_END = .*/BSSN_RK_TIME_END = 0.04/" local.par.toml
  [ $V = bad_lambda ] && sed -i -E 's/^BSSN_LAMBDA = .*/BSSN_LAMBDA = [1, 1, 1, 0]/' local.par.toml
  [ $V = bad_eta ] && python3 - <<PY
import re
from pathlib import Path
p=Path("local.par.toml"); s=p.read_text(); m=re.search(r"^\[",s,flags=re.M); p.write_text(s[:m.start()]+"RIT_ETA_WIDTH = 30.0\n\n"+s[m.start():])
PY
  timeout 300 mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=8 env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
  echo "$V: exit=$? $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep -E 'FATAL|ETS time' | head -1 | cut -c1-150)" >> $HOME/Dendro-GR/campaign-notes/tier0/paramcheck.out
  cd $HOME/Dendro-GR/campaign-notes
done
echo D > tier0/paramcheck.done
