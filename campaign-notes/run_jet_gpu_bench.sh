#!/usr/bin/env bash
#SBATCH -p debug -N 1 -t 0:30:00 -J jetgpu -o /home1/08435/srvenkat/Dendro-GR/campaign-notes/jet_gpu_bench.out
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source $HOME/Dendro-GR/campaign-notes/env_build_horizon.sh >/dev/null
unset CUDA_VISIBLE_DEVICES
$SCRATCH/jet_gpu_bench 4194304 20
$SCRATCH/jet_gpu_bench 1048576 50
echo done
