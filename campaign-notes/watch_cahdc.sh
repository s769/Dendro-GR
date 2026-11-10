#!/usr/bin/env bash
# run run_cahd_c.sh once, on the first of these jobs that starts
D=$HOME/Dendro-GR/campaign-notes
while true; do
  for j in 88890 88891 88893; do
    N=$(squeue -j $j -h -t R -o %N 2>/dev/null | head -1)
    if [ -n "$N" ]; then sleep 15; ssh -o StrictHostKeyChecking=no $N "nohup bash $D/run_cahd_c.sh > /dev/null 2>&1 &"; echo "$(date -Is) run_cahd_c.sh on $N (job $j)" >> $D/tier0/watch_cahdc.log; exit 0; fi
  done
  sleep 10
done
