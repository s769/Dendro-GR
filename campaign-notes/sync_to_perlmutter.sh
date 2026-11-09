#!/usr/bin/env bash
# Copy Dendro-GR handoff notes (+ local Cursor rule) to Perlmutter home.
#
# NERSC auth (MFA sshproxy), then:
#   bash ~/Dendro-GR/campaign-notes/sync_to_perlmutter.sh
# Optional full tree:
#   SYNC_FULL_REPO=1 bash ~/Dendro-GR/campaign-notes/sync_to_perlmutter.sh
set -euo pipefail

HOST="${PERLMUTTER_HOST:-srvenkat@perlmutter.nersc.gov}"
REMOTE_HOME="${PERLMUTTER_HOME:-/global/homes/s/srvenkat}"
SRC_NOTES="${HOME}/Dendro-GR/campaign-notes"
SRC_RULE="${HOME}/Dendro-GR/.cursor/rules/gpu-opt-context.mdc"
SRC_REPO="${HOME}/Dendro-GR"

echo "Target: ${HOST}:${REMOTE_HOME}"

ssh -o BatchMode=yes "$HOST" "mkdir -p '${REMOTE_HOME}/Dendro-GR/campaign-notes' '${REMOTE_HOME}/Dendro-GR/.cursor/rules'"

rsync -avP "${SRC_NOTES}/" "${HOST}:${REMOTE_HOME}/Dendro-GR/campaign-notes/"
if [[ -f "$SRC_RULE" ]]; then
  rsync -avP "$SRC_RULE" "${HOST}:${REMOTE_HOME}/Dendro-GR/.cursor/rules/"
fi

if [[ "${SYNC_FULL_REPO:-0}" == "1" ]]; then
  rsync -avP "${SRC_REPO}/" "${HOST}:${REMOTE_HOME}/Dendro-GR/"
else
  echo "Skipped full repo (set SYNC_FULL_REPO=1 to rsync). On Perlmutter you can also:"
  echo "  git clone https://github.com/paralab/Dendro-GR.git ~/Dendro-GR"
fi

ssh "$HOST" "ls -la '${REMOTE_HOME}/Dendro-GR/campaign-notes' '${REMOTE_HOME}/Dendro-GR/.cursor/rules'"
echo "Done. On Perlmutter agent: read ~/Dendro-GR/campaign-notes/GPU_OPT_BRIEF.md first."
