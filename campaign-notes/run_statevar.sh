#!/usr/bin/env bash
# usage: run_statevar.sh NAME "sed-expr"   (binaries already built; no rebuild!)
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null; unset CUDA_VISIBLE_DEVICES
NAME=$1; EXPR=$2
R=$HOME/Dendro-GR-runs/statecmp_$NAME; rm -rf $R; mkdir -p $R
for M in cpu gpu; do
  d=$R/$M; mkdir -p $d/{aeh,cp,dat,vtu,dump}; cp $HOME/Dendro-GR-runs/phys_gf_fix/local.par.toml $d/
  sed -i -E 's/^BSSN_RK_TIME_END = .*/BSSN_RK_TIME_END = 0.04/; s/^BSSN_REMESH_TEST_FREQ = .*/BSSN_REMESH_TEST_FREQ = 0/' $d/local.par.toml
  sed -i -E "$EXPR" $d/local.par.toml
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin
  cd $d
  if [ $M = cpu ]; then
    DENDRO_DUMP_STATE=$d/dump:1 mpirun -np 8 --bind-to none env OMP_NUM_THREADS=1 $DENDRO_BUILD/BSSN_GR/bssnSolver local.par.toml 1 > run.log 2>&1
  else
    DENDRO_DUMP_STATE=$d/dump:1 mpirun -np 8 --bind-to none -x OMP_NUM_THREADS=1 -x DENDRO_DUMP_STATE env DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
  fi
done
echo DONE > $HOME/Dendro-GR/campaign-notes/tier0/statevar_$NAME.done
