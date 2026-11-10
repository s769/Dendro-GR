#!/usr/bin/env bash
# Corrected headline speedup at production depth (q8, T=0.3 ~ 300 steps incl. 5 remesh + 3 extractions), full 4-GPU node.
#  BASE  : original GPU port behaviour (stock RHS, no overlap/boundary stream, host WAMR, serial init, eager host vectors,
#          original physics = no SSL/CAHD/init-refmode)
#  OPTO  : all performance work, ORIGINAL physics (pure speedup)
#  OPTF  : all performance work + corrected physics (init-refmode, SSL, CAHD, per-step BH tracking)  <- production
cd $HOME/Dendro-GR/campaign-notes
ORIG='DENDRO_GPU_INIT_REFMODE=0 DENDRO_GPU_SSL=0 DENDRO_GPU_CAHD=0'
OMPT=1  bash ./run_prod_q8.sh SP_BASE 0.3 "env -u DENDRO_RHS_PT_ASYNC DENDRO_GPU_BH_WAMR=0 DENDRO_GPU_WAMR=0 DENDRO_UNZIP_OVERLAP=0 DENDRO_BDY_STREAM=0 DENDRO_INIT_OMP=0 DENDRO_EAGER_HOST_UNZIP=1 DENDRO_GW_CACHE_SAMPLER=0 $ORIG" 0 > tier0/prod_SP_BASE.out 2>&1
OMPT=36 bash ./run_prod_q8.sh SP_OPTO 0.3 "DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1 $ORIG" 0 > tier0/prod_SP_OPTO.out 2>&1
OMPT=36 bash ./run_prod_q8.sh SP_OPTF 0.3 "DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1 DENDRO_BH_TRACK_FREQ=1" 0 > tier0/prod_SP_OPTF.out 2>&1
echo D > tier0/speedup3.done
