#!/usr/bin/env bash
set -euo pipefail
# Wait for N8 TPID sol, then run cost table.
for i in $(seq 1 120); do
  if [[ -f /home1/08435/srvenkat/Dendro-GR-runs/tpid_tier33_orbits/N8/rit_q1_N8_tpid_sol.bin ]]; then
    echo "N8_sol_ready $(date -Is)"
    break
  fi
  sleep 10
done
# Wait until retpid script finished writing N8 wall line (avoid racing tpid)
for i in $(seq 1 60); do
  if grep -q 'N8 wall=' /home1/08435/srvenkat/Dendro-GR/campaign-notes/tier0/tier33_retpid_orbits.log 2>/dev/null; then
    break
  fi
  sleep 5
done
bash /home1/08435/srvenkat/Dendro-GR/campaign-notes/run_tier33_cost_orbits.sh N3 N5 N8
