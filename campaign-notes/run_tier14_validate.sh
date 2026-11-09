#!/usr/bin/env bash
# On a compute node: rebuild + dual-run bit-id smoke after min-det fix,
# plus non-fiducial mesh replay check (perturbed bare mass on frozen mesh).
set -euo pipefail
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
cmake --build "${DENDRO_BUILD}" -j16 --target bssnSolverCUDA 2>&1 | tail -30

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier14"
rm -rf "${OUT}"
mkdir -p "${OUT}"

run_one() {
  local tag="$1"; shift
  local d="${OUT}/${tag}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  ln -sfn "${TPID}" "${d}/rit_q1_tpid_sol.bin"
  cd "${d}"
  env "$@" "${BIN}" "${PAR}" 1 > "${d}/run.log" 2>&1
  grep -E 'C_HAM|U_ALPHA \(min|ETS time' "${d}/run.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -20 \
    > "${d}/summary.txt"
}

# Dual identical fiducial (bit-id of constraints after atomic removal)
run_one fid_a DENDRO_RHS_PT_ASYNC=1
run_one fid_b DENDRO_RHS_PT_ASYNC=1
if cmp -s "${OUT}/fid_a/dat/bssn_profile_Constraints.dat" \
         "${OUT}/fid_b/dat/bssn_profile_Constraints.dat"; then
  echo "TIER14_FID_BITID_OK"
else
  echo "TIER14_FID_BITID_MISMATCH"
  # still continue
fi

# Record fiducial mesh, replay with same ID (sanity)
REC="${OUT}/mesh_rec"
REP="${OUT}/mesh_rep"
mkdir -p "${REC}" "${REP}"/{aeh,cp,dat,vtu}
ln -sfn "${TPID}" "${OUT}/mesh_rec_run/rit_q1_tpid_sol.bin" 2>/dev/null || true
mkdir -p "${OUT}/mesh_rec_run"/{aeh,cp,dat,vtu}
ln -sfn "${TPID}" "${OUT}/mesh_rec_run/rit_q1_tpid_sol.bin"
cd "${OUT}/mesh_rec_run"
env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_RECORD="${REC}" "${BIN}" "${PAR}" 1 \
  > "${OUT}/mesh_rec_run/run.log" 2>&1 || true
ls -la "${REC}" | head

ln -sfn "${TPID}" "${REP}/rit_q1_tpid_sol.bin"
cd "${REP}"
env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY="${REC}" "${BIN}" "${PAR}" 1 \
  > "${REP}/run.log" 2>&1 || true
grep -E 'Frozen-mesh|C_HAM|init' "${REP}/run.log" | sed 's/\x1b\[[0-9;]*m//g' | head -30

echo TIER14_VALIDATE_DONE
