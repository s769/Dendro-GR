#!/usr/bin/env bash
D=$HOME/Dendro-GR/campaign-notes
until [ -n "$(squeue -j 89193 -h -t R -o %N 2>/dev/null)" ]; do sleep 15; done
N=$(squeue -j 89193 -h -t R -o %N | head -1); sleep 10
rm -f $HOME/Dendro-GR-runs/tier15_wave_coarse_T26_cahd/tpfix.done
ssh -o StrictHostKeyChecking=no $N "nohup bash $D/run_tpfix_test.sh > /dev/null 2>&1 &"
echo "$(date -Is) tpfix test on $N (89193)" >> $D/tier0/watch_dev1.log
