#!/usr/bin/env bash
# usage: watch_job.sh JOBID "ENV=.. ENV=.." script   -> when JOBID runs, ssh to its node and start script with env (once)
J="$1"; ENVS="$2"; S="$3"; LOG="$HOME/Dendro-GR/campaign-notes/tier0/watch_job_$J.log"
while true; do
  read -r st host < <(squeue -j "$J" -h -o '%T %N' 2>/dev/null) || true
  [[ -z "${st:-}" ]] && { echo "$(date -Is) job gone" >> "$LOG"; exit 0; }
  if [[ "$st" == RUNNING ]]; then
    echo "$(date -Is) start on $host" >> "$LOG"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "$ENVS nohup bash $S > $HOME/Dendro-GR/campaign-notes/tier0/job_${J}.log 2>&1 &"; exit 0
  fi; sleep 20
done
