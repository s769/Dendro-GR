#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null; unset CUDA_VISIBLE_DEVICES
d=$HOME/Dendro-GR-runs/unzipcmp2; rm -rf $d; mkdir -p $d/{aeh,cp,dat,vtu}; cp $HOME/Dendro-GR-runs/phys_gf_fix/local.par.toml $d/
sed -i -E 's/^BSSN_RK_TIME_END = .*/BSSN_RK_TIME_END = 0.04/; s/^BSSN_REMESH_TEST_FREQ = .*/BSSN_REMESH_TEST_FREQ = 0/' $d/local.par.toml
ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=8 env OMP_NUM_THREADS=8 DENDRO_UNZIP_CMP=1 DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
echo DONE > $HOME/Dendro-GR/campaign-notes/tier0/unzipcmp2.done
