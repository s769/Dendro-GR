#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
bash ./run_physcheck.sh gpu 20.0  > tier0/phys_gpu.out 2>&1
bash ./run_physcheck.sh gpu0 20.0 > tier0/phys_gpu0.out 2>&1
timeout 1500 bash ./run_physcheck.sh cpu 20.0 > tier0/phys_cpu.out 2>&1
echo PHYS_DONE > tier0/phys.done
