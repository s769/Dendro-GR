#!/usr/bin/env bash
# launch run_bisect.sh on the first RUNNING idv job node, once
D=$HOME/Dendro-GR/campaign-notes
while true; do
  N=$(squeue -u $USER -h -t R -o "%N" | grep -v "^$" | head -1)
  if [[ -n "$N" ]]; then sleep 25; ssh -o BatchMode=yes -o StrictHostKeyChecking=no $N "nohup bash $D/run_bisect.sh > /dev/null 2>&1 &" && echo "$(date -Is) launched on $N" >> $D/tier0/watch_bisect.log; exit 0; fi
  sleep 15
done
