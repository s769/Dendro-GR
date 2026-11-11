#!/usr/bin/env bash
# Does reusing the fiducial TwoPunctures solution for a perturbed BH1 mass corrupt the FD sensitivity?
# Replay the coarse +0.001 member with (a) the reused file [existing run p_0.001] and (b) the correctly solved perturbed file.
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
OUT=$HOME/Dendro-GR-runs/tier15_wave_coarse_T26_cahd; REC=$OUT/mesh_rec; BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
d=$OUT/p_0.001_tpfix; rm -rf $d; mkdir -p $d/{aeh,cp,dat,vtu}; cp $OUT/p_0.001/local.par.toml $d/
SOL=$HOME/Dendro-GR-runs/tpwarm/pert_warm36/tp_pert_warm36_tpid_sol.bin
ln -sfn $SOL $d/rit_q1_tpid_sol.bin
( cd $d; s=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 mpirun -np 1 --bind-to none env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY=$REC $BIN local.par.toml 1 > run.log 2>&1
  echo "tpfix wall=$(( $(date +%s)-s ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1)" > $OUT/tpfix.done )
