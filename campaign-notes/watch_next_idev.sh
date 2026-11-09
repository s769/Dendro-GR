#!/usr/bin/env bash
# When JOB becomes RUNNING, launch busy pipeline on that node.
JOB="${1:-86880}"
SCRIPT="${HOME}/Dendro-GR-notes/run_tier2_busy_c125.sh"
LOG="${HOME}/Dendro-GR-notes/tier0/watch_idev_${JOB}.log"
echo "$(date -Is) watching ${JOB}" | tee -a "${LOG}"
while true; do
  st=$(timeout 12 squeue -j "${JOB}" -h -o '%T %N' 2>/dev/null || true)
  t=$(echo "$st" | awk '{print $1}')
  n=$(echo "$st" | awk '{print $2}')
  echo "$(date -Is) state=${t:-NONE} node=${n:-}" | tee -a "${LOG}"
  if [[ "$t" == "RUNNING" && -n "$n" ]]; then
    echo "$(date -Is) launching busy on ${n}" | tee -a "${LOG}"
    ssh -F /dev/null -o BatchMode=yes "$n" \
      "nohup ${SCRIPT} >${HOME}/Dendro-GR-notes/tier0/busy_${n}.out 2>&1 & echo PID=\$!" \
      | tee -a "${LOG}"
    exit 0
  fi
  if [[ -z "$t" || "$t" == "COMPLETED" || "$t" == "CANCELLED" || "$t" == "FAILED" || "$t" == "TIMEOUT" ]]; then
    echo "$(date -Is) job gone state=${t}" | tee -a "${LOG}"
    exit 0
  fi
  sleep 45
done
