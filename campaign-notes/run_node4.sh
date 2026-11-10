#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
bash ./run_tp_warm.sh > tier0/tp_warm.out 2>&1
echo DONE > tier0/node4.done
