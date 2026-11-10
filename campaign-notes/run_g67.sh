#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
OMPT=8 bash ./run_prod_q8.sh G6 0.5 'DENDRO_GW_SHELL_ONLY=1 DENDRO_UNZIP_VB=2' 0 > tier0/prod_G6.out 2>&1
OMPT=8 bash ./run_prod_q8.sh G7 0.5 'DENDRO_GW_SHELL_ONLY=1 DENDRO_UNZIP_VB=8' 0 > tier0/prod_G7.out 2>&1
OMPT=8 bash ./run_prod_q8.sh G8 0.5 'DENDRO_GW_SHELL_ONLY=1' 0 > tier0/prod_G8.out 2>&1
echo G67_DONE > tier0/g67.done
