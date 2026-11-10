#!/usr/bin/env bash
# usage: watch_job_run.sh JOBID SCRIPT : run SCRIPT on the node of JOBID once it is RUNNING
D=$HOME/Dendro-GR/campaign-notes
while true; do
  N=$(squeue -j $1 -h -t R -o "%N" 2>/dev/null | head -1)
  if [[ -n "$N" ]]; then sleep 25; ssh -o BatchMode=yes -o StrictHostKeyChecking=no $N "nohup bash $D/$2 > /dev/null 2>&1 &" && echo "$(date -Is) $2 launched on $N (job $1)" >> $D/tier0/watch_$2.log; exit 0; fi
  sleep 15
done
