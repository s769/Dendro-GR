#!/usr/bin/env bash
# bisect GPU-vs-CPU physics gap on frozen mesh (T=6.4): which GPU feature moves BH velocity / step-0 C_HAM
cd $HOME/Dendro-GR/campaign-notes
NP=1 EXTRA="DENDRO_INIT_OMP=0 DENDRO_RHS_PT_ASYNC=1" bash ./run_physcheck.sh gf_ptinit0 6.4 > tier0/phys_gf_ptinit0.out 2>&1
NP=1 EXTRA="DENDRO_INIT_OMP=0" bash ./run_physcheck.sh gf_stockinit0 6.4 > tier0/phys_gf_stockinit0.out 2>&1
NP=1 EXTRA="DENDRO_INIT_OMP=1" bash ./run_physcheck.sh gf_stock 6.4 > tier0/phys_gf_stock.out 2>&1
echo BISECT_DONE > tier0/bisect.done
