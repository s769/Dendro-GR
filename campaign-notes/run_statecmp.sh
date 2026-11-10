#!/usr/bin/env bash
# dump full state after 1,2,5 steps from CPU (np=8) and GPU (np=8 on 1 GPU), same mesh; diff in compare_state.py
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null; unset CUDA_VISIBLE_DEVICES
( cd $DENDRO_BUILD && make -j16 bssnSolver bssnSolverCUDA 2>&1 | tail -2 ) > tier0/statecmp_build.log 2>&1
R=$HOME/Dendro-GR-runs/statecmp; rm -rf $R; mkdir -p $R
for M in cpu gpu; do
  d=$R/$M; mkdir -p $d/{aeh,cp,dat,vtu,dump}; cp $HOME/Dendro-GR-runs/phys_gf_fix/local.par.toml $d/
  sed -i -E 's/^BSSN_RK_TIME_END = .*/BSSN_RK_TIME_END = 0.1/; s/^BSSN_REMESH_TEST_FREQ = .*/BSSN_REMESH_TEST_FREQ = 0/' $d/local.par.toml
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin
  cd $d
  if [ $M = cpu ]; then
    DENDRO_DUMP_RHS=$d/dump:0 DENDRO_DUMP_STATE=$d/dump:0,1,2,5 mpirun -np 8 --bind-to none env OMP_NUM_THREADS=1 $DENDRO_BUILD/BSSN_GR/bssnSolver local.par.toml 1 > run.log 2>&1
  else
    DENDRO_DUMP_RHS=$d/dump:0 DENDRO_DUMP_STATE=$d/dump:0,1,2,5 mpirun -np 8 --bind-to none -x OMP_NUM_THREADS=1 -x DENDRO_DUMP_STATE -x DENDRO_DUMP_RHS env DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
  fi
done
echo DONE > $HOME/Dendro-GR/campaign-notes/tier0/statecmp.done
