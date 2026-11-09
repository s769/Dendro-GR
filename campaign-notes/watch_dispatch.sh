#!/usr/bin/env bash
# Dispatch $1 (script) once to the first idev job that is RUNNING; then exit.
S="${1:?script}"; LOG="$HOME/Dendro-GR/campaign-notes/tier0/watch_dispatch.log"
while true; do
  read -r jid host < <(squeue -u "$USER" -t R -o '%i %N' --noheader | grep -E ' c' | head -1 || true) || true
  if [[ -n "${host:-}" ]]; then
    echo "$(date -Is) dispatch $S to $host (job $jid)" >> "$LOG"
    ssh -o BatchMode=yes "$host" "nohup bash $S > $HOME/Dendro-GR/campaign-notes/tier0/dispatch_$host.out 2>&1 &"
    exit 0
  fi
  sleep 30
done
