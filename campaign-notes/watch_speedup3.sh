#!/usr/bin/env bash
D=$HOME/Dendro-GR/campaign-notes
until [ -n "$(squeue -j 88890 -h -t R -o %N 2>/dev/null)" ]; do sleep 15; done
N=$(squeue -j 88890 -h -t R -o %N | head -1); sleep 15
ssh -o StrictHostKeyChecking=no $N "nohup bash $D/run_speedup3.sh > /dev/null 2>&1 &"
echo "$(date -Is) run_speedup3.sh on $N (job 88890)" >> $D/tier0/watch_speedup3.log
