#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
OMPT=8 bash ./run_prod_q8.sh G4 0.5 'DENDRO_GW_SHELL_ONLY=1' 0 > tier0/prod_G4.out 2>&1
OMPT=8 bash ./run_prod_q8.sh G5 0.5 'DENDRO_GW_SHELL_ONLY=1 DENDRO_UNZIP_VB=4' 0 > tier0/prod_G5.out 2>&1
echo G45_DONE > tier0/g45.done
