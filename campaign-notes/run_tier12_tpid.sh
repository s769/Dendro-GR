#!/usr/bin/env bash
# Tier 1.2: dual tpid bit-id + unequal-mass bare-mass check
set -euo pipefail
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
TPID="${DENDRO_BUILD}/BSSN_GR/tpid"
OUT="${HOME}/Dendro-GR-runs/tpid_tier12"
rm -rf "${OUT}"
mkdir -p "${OUT}"

run_pair() {
  local tag="$1" par="$2" prefix="$3"
  for ab in a b; do
    local d="${OUT}/${tag}_${ab}"
    mkdir -p "${d}"
    cd "${d}"
    cp "${par}" local.par.toml
    # force unique FILEPREFIX
    if grep -q '^TPID_FILEPREFIX' local.par.toml; then
      sed -i "s|^TPID_FILEPREFIX.*|TPID_FILEPREFIX = \"${prefix}_${ab}\"|" local.par.toml
    else
      echo "TPID_FILEPREFIX = \"${prefix}_${ab}\"" >> local.par.toml
    fi
    echo "[run] ${tag}_${ab}"
    "${TPID}" local.par.toml 4 > tpid.log 2>&1
    test -f "${prefix}_${ab}_tpid_sol.bin"
  done
  local md5_a md5_b
  md5_a=$(md5sum "${OUT}/${tag}_a/${prefix}_a_tpid_sol.bin" | awk '{print $1}')
  md5_b=$(md5sum "${OUT}/${tag}_b/${prefix}_b_tpid_sol.bin" | awk '{print $1}')
  echo "${tag}: a=${md5_a} b=${md5_b}"
  if [[ "${md5_a}" != "${md5_b}" ]]; then
    echo "${tag}: MISMATCH"
    exit 1
  fi
  echo "${tag}: BIT_IDENTICAL"
  grep -A6 'resolved inversion vector' "${OUT}/${tag}_a/tpid.log" || true
  grep -E 'two puncture masses|NOTE: BSSN_BH|WARNING: GIVE' "${OUT}/${tag}_a/tpid.log" || true
}

run_pair q1 "${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml" rit_q1
run_pair q4 "${HOME}/Dendro-GR-runs/q4.meshprobe.par.toml" mesh_q4
echo '--- q4 bare masses (expect 0.2 / 0.8) ---'
grep -E 'm\+=|bare mass:|two puncture masses' "${OUT}/q4_a/tpid.log" | head -10
echo 'TIER12_TPID_OK'
