#!/usr/bin/env bash
# When JOB is RUNNING, run the next *real* GPU queue item (not filler).
JOB="${1:-86880}"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/watch_idev_${JOB}.log"
NEXT="${HOME}/Dendro-GR/campaign-notes/run_next_gpu_task.sh"
echo "$(date -Is) watching ${JOB} for next GPU task" | tee -a "${LOG}"
while true; do
  st=$(timeout 12 squeue -j "${JOB}" -h -o '%T %N' 2>/dev/null || true)
  t=$(echo "$st" | awk '{print $1}')
  n=$(echo "$st" | awk '{print $2}')
  echo "$(date -Is) state=${t:-NONE} node=${n:-}" | tee -a "${LOG}"
  if [[ "$t" == "RUNNING" && -n "$n" ]]; then
    echo "$(date -Is) dispatch next GPU task on ${n}" | tee -a "${LOG}"
    ssh -F /dev/null -o BatchMode=yes "$n" \
      "nohup ${NEXT} >${HOME}/Dendro-GR/campaign-notes/tier0/next_gpu_${n}.out 2>&1 & echo PID=\$!" \
      | tee -a "${LOG}"
    exit 0
  fi
  if [[ -z "$t" || "$t" == "COMPLETED" || "$t" == "CANCELLED" || "$t" == "FAILED" || "$t" == "TIMEOUT" ]]; then
    echo "$(date -Is) job gone state=${t}" | tee -a "${LOG}"
    exit 0
  fi
  sleep 45
done
