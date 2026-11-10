# GPU-vs-CPU physics check (2026-11-10) — RESOLVED root causes

Reference: CPU `bssnSolver` (36 ranks), q1 smoke (maxdepth 13, BH_WAMR), same TP solution file.
Method: frozen mesh (no remesh), per-step state / unzip / stage-1 RHS dumps from both solvers
(`DENDRO_DUMP_STATE=<dir>:<steps>`, `DENDRO_DUMP_RHS=<dir>:<step>`, `compare_state.py`, `analyze_unzip.py`).

## Findings (all in the ORIGINAL GPU port, none from the campaign optimizations)
1. **Init refinement ignored `BSSN_USE_SET_REF_MODE_FOR_INITIAL_CONVERGE`.** CPU refines the init mesh with the
   configured mode (BH_WAMR: 7512 elements); GPU used plain WAMR (4964). Fixed in `BSSNCtxGPU::initialize`
   (`DENDRO_GPU_INIT_REFMODE=0` restores old). Step-0 C_HAM now identical (9.2068383e-6).
2. **GPU RHS lacked the slow-start-lapse (SSL) term.** CPU build has `BSSN_ENABLE_SSL_HD=ON`
   (`bssneqs_SSL_HD_dxsq.cpp`): `a_rhs += -sqrt(chi)*h*(alpha-sqrt(chi))*exp(-t^2/(2 sig^2))`, h=0.6, sig=20
   (hard-coded defaults, not in the par files). Added as an elementwise pass after the RHS kernels
   (`DENDRO_GPU_SSL=0` disables). Found by diffing stage-1 RHS on identical input: only alpha differed.
3. **Not implemented on GPU (guarded):** CAHD (`BSSN_CAHD_C`, default 0 -> GPU now aborts if nonzero) and CAKO
   (`BSSN_CAKO_ENABLED`; driver enables it in some merger configs -> GPU prints a warning, uses plain KO).
   Neither is on in the current par files.
4. **BH tracking cadence differs:** CPU updates BH locations every step; GPU only at GW-extraction steps (coarser
   integration). Waveform unaffected (frozen mesh); trajectory differs ~2.6e-3 abs, sep 3.5e-4 rel. Affects BH-based
   refinement regions in adaptive runs. OPEN.

## Ruled out
Unzip / hanging-node interpolation (device unzip differs from CPU only by the state difference), dx, KO stencils
(identical), advection, eta damping, OpenMP init, stock vs PT_ASYNC RHS (agree to 1e-11 with each other).

## Result after fixes (frozen mesh, t<=6.35)
psi4 l2m2 3e-11, l3m2 2e-11 rel-to-max (was 3e-3 / 1.5e-2); state after 5 steps 1.5e-11 (was 1e-4);
C_HAM final equal. Remaining: BH trajectory (item 4).

## Consequence for earlier numbers
GPU runs before this fix started on a coarser init mesh (4964 vs 7512 elements) and solved a different gauge.
Earlier GPU timings (esp. init-phase and early-step costs) should be re-measured; ensemble FD tables
(`TIER1_FD.md`) were produced with the old GPU physics and need to be redone.
