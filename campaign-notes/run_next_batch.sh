#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
# 1) frozen-mesh CPU vs GPU (identical mesh) physics check
bash ./run_physcheck.sh cpu0 6.4 > tier0/phys_cpu0.out 2>&1
# 2) OpenMP thread-count sweep at production depth (init_grid, remesh, extraction scale)
for T in 4 8 16 36; do OMPT=$T bash ./run_prod_q8.sh TS$T 0.3 'DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1' 0 > tier0/prod_TS$T.out 2>&1; done
OMPT=8 bash ./run_prod_q8.sh TS8E 0.3 'DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1 DENDRO_EAGER_HOST_UNZIP=1' 0 > tier0/prod_TS8E.out 2>&1
# 3) TwoPunctures cold vs warm, thread scaling
bash ./run_tp_warm.sh > tier0/tp_warm.out 2>&1
echo BATCH_DONE > tier0/batch.done
