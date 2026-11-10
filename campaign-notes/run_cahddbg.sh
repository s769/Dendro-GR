#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null; unset CUDA_VISIBLE_DEVICES
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | tail -1 ) > tier0/cahddbg_build.log 2>&1
d=$HOME/Dendro-GR-runs/statecmp_cahd/gpu_dbg; rm -rf $d; mkdir -p $d; cp -r $HOME/Dendro-GR-runs/statecmp_cahd/gpu/{aeh,cp,dat,vtu,local.par.toml} $d/ 2>/dev/null; ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin
sed -i -E 's/^BSSN_RK_TIME_END = .*/BSSN_RK_TIME_END = 0.04/' $d/local.par.toml; cd $d
mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=8 env OMP_NUM_THREADS=8 DENDRO_CAHD_DEBUG=1 DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
grep cahd-debug run.log > ../../../Dendro-GR/campaign-notes/tier0/cahddbg.out
echo D > $HOME/Dendro-GR/campaign-notes/tier0/cahddbg.done
