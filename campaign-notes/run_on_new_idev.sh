#!/usr/bin/env bash
# When a pending idev job becomes R, ssh in and run the next queued GPU task.
set -euo pipefail
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/watch_new_idev.log"
mkdir -p "$(dirname "${LOG}")"
echo "$(date -Is) watch_new_idev start" | tee -a "${LOG}"

seen_running=""
while true; do
  while read -r jid host; do
    [[ -z "${jid}" || -z "${host}" ]] && continue
    key="${jid}:${host}"
    if [[ " ${seen_running} " == *" ${key} "* ]]; then
      continue
    fi
    seen_running="${seen_running} ${key}"
    echo "$(date -Is) NEW_R job=${jid} host=${host}" | tee -a "${LOG}"
    # Prefer completing retpid N=8, then short cost ETS with new sols
    ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}" "nohup bash -c '
      set -euo pipefail
      source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
      source \$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh
      unset CUDA_VISIBLE_DEVICES
      export OMP_NUM_THREADS=1
      L=\$HOME/Dendro-GR/campaign-notes/tier0/new_idev_${jid}.log
      echo host=\$(hostname) job=${jid} \$(date -Is) | tee \$L
      # Finish N=8 retpid if missing
      if [[ ! -f \$HOME/Dendro-GR-runs/tpid_tier33_orbits/N8/rit_q1_N8_tpid_sol.bin ]]; then
        bash \$HOME/Dendro-GR/campaign-notes/run_tier33_retpid_orbits.sh 8.0 2>&1 | tee -a \$L
      fi
      # Short cost ETS for N=3 and N=5 if sols exist
      BIN=\$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
      PAR0=\$HOME/Dendro-GR-runs/q1.smoke_cons.par.toml
      for tag in N3 N5 N8; do
        sol=\$HOME/Dendro-GR-runs/tpid_tier33_orbits/\$tag/rit_q1_\${tag}_tpid_sol.bin
        [[ -f \$sol ]] || continue
        d=\$HOME/Dendro-GR-runs/tier33_cost_\$tag
        mkdir -p \$d/{aeh,cp,dat,vtu}
        cp \$PAR0 \$d/local.par.toml
        Nf=\$(grep BSSN_TARGET_ORBITS \$HOME/Dendro-GR-runs/tpid_tier33_orbits/\$tag/local.par.toml | awk \"{print \\\$3}\")
        python3 - <<PY
from pathlib import Path
import re
p=Path(\"$d/local.par.toml\"); t=p.read_text()
t=re.sub(r\"BSSN_RK_TIME_END = [0-9.eE+-]+\", \"BSSN_RK_TIME_END = 0.15\", t)
t=re.sub(r\"BSSN_IO_OUTPUT_FREQ = [0-9]+\", \"BSSN_IO_OUTPUT_FREQ = 0\", t)
t=re.sub(r\"BSSN_CHECKPT_FREQ = [0-9]+\", \"BSSN_CHECKPT_FREQ = 0\", t)
n=\"$Nf\"
line=f\"BSSN_TARGET_ORBITS = {float(n)}\"
if \"BSSN_TARGET_ORBITS\" in t: t=re.sub(r\"BSSN_TARGET_ORBITS = [0-9.eE+-]+\", line, t)
else: t=line+\"\\n\"+t
t=re.sub(r\"TPID_FILEPREFIX = \\\"[^\\\"]*\\\"\", f\"TPID_FILEPREFIX = \\\"rit_q1_$tag\\\"\", t)
p.write_text(t)
PY
        ln -sfn \$sol \$d/rit_q1_\${tag}_tpid_sol.bin
        cd \$d
        echo ==== cost \$tag ==== | tee -a \$L
        START=\$(date +%s)
        mpirun -np 1 --bind-to none env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
          \$BIN local.par.toml 1 >run.log 2>&1 || { echo FAIL_\$tag | tee -a \$L; continue; }
        END=\$(date +%s)
        ETS=\$(grep -E \"ETS time \\(max\\)\" run.log | sed \"s/\\x1b\\[[0-9;]*m//g\" | tail -1 | awk \"{print \\\$NF}\")
        LMAX=\$(grep -E \"lmax:\" run.log | sed \"s/\\x1b\\[[0-9;]*m//g\" | tail -1)
        echo \"\$tag wall=\$((END-START))s ets=\${ETS:-NA} \$LMAX\" | tee -a \$L
      done
      echo NEW_IDEV_TASKS_DONE | tee -a \$L
    ' > \$HOME/Dendro-GR/campaign-notes/tier0/new_idev_${jid}_nohup.out 2>&1 &"
  done < <(squeue -u srvenkat -t R -o '%i %N' --noheader | grep -E 'idv' || true)
  sleep 30
done
