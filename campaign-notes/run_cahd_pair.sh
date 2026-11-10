#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
if [ "$1" = cpu ]; then PAR_OVR="BSSN_CAHD_C=0.06" bash ./run_physcheck.sh fzcpu_cahd 6.4 > tier0/phys_fzcpu_cahd.out 2>&1; echo D > tier0/cahd_cpu.done
else PAR_OVR="BSSN_CAHD_C=0.06" NP=1 bash ./run_physcheck.sh fzgpu_cahd 6.4 > tier0/phys_fzgpu_cahd.out 2>&1; echo D > tier0/cahd_gpu.done; fi
