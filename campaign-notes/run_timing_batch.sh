#!/usr/bin/env bash
# re-measure production-depth cost after the physics fixes (36 OMP threads/rank)
cd $HOME/Dendro-GR/campaign-notes
E='DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1'
OMPT=36 bash ./run_prod_q8.sh NEW   0.3 "$E" 0 > tier0/prod_NEW_${TAGN:-a}.out 2>&1
OMPT=36 bash ./run_prod_q8.sh OLD   0.3 "$E DENDRO_GPU_INIT_REFMODE=0 DENDRO_GPU_SSL=0 DENDRO_GPU_CAHD=0" 0 > tier0/prod_OLD_${TAGN:-a}.out 2>&1
OMPT=36 bash ./run_prod_q8.sh TRK1  0.3 "$E DENDRO_BH_TRACK_FREQ=1" 0 > tier0/prod_TRK1_${TAGN:-a}.out 2>&1
echo D > tier0/timing_batch.done
