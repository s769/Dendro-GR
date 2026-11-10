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
4. **BH tracking cadence:** CPU updates BH locations every step; the ORIGINAL GPU driver only at GW-extraction steps
   (verified in git f790d8a, same as ours). Trajectory differs ~2.6e-3 abs. New option `DENDRO_BH_TRACK_FREQ=N`
   (default 0 = baseline behaviour; shift-only D2H every N steps; 1 = CPU cadence): N=1 reproduces the CPU trajectory
   to 4e-15, N=5 gives 5.9e-4; waveform unaffected; no measurable wall-time cost on the smoke case.

Note: CAHD / CAKO / SSL are absent from the ORIGINAL GPU kernels too (git f790d8a). SSL was added to match the CPU;
`DENDRO_GPU_SSL=0` and `DENDRO_GPU_INIT_REFMODE=0` give original-GPU behaviour. CAHD/CAKO: guard only.

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

## Adaptive validation (remesh test every 10 steps, BH_WAMR, GPU 4 ranks vs CPU 36 ranks, to t=12.5)
psi4 l2m2 1.5e-11, l3m2 3.4e-11 rel-to-max; C_HAM 1.0100e-05 both (max differs 1.5e-4 rel);
remesh rate 6.0 vs 6.1 per time unit. The earlier concern (constraints rising with remeshing) was a symptom of the
gauge/mesh mismatches, not of remeshing.

## CAHD effect (production par files set BSSN_CAHD_C=0.06; q32: 0.077). Frozen mesh, t<=6.35, CPU 36 ranks
CPU with CAHD vs CPU without: psi4 l2m2 7.5e-5, l3m2 2.1e-4 rel-to-max; C_HAM +1% (9.37e-6 -> 9.47e-6);
BH position 8.7e-4. GPU (no CAHD, as in the original port) vs CPU with CAHD: identical psi4 difference
(7.5e-5 / 2.1e-4) -> the GPU-vs-CPU gap with production pars is exactly the CAHD effect (plus tracking cadence).
Effect at early time only; accumulation over a full inspiral NOT yet measured. Not implemented on GPU (baseline also
lacks it); GPU prints a warning (DENDRO_GPU_CAHD_STRICT=1 aborts).
Harness bug found: run_physcheck.sh appended missing keys inside [AEH_PARAMS]; fixed (insert before first table).

## Production-depth timing after the fixes (q8, T=0.3, 36 OMP/rank, single samples)
old physics: startup 43 s, ETS 28.5 s | new defaults: 61 s, 30.0 s (+5%) | new + BH_TRACK_FREQ=1: 57 s, 31.2 s (+4%).
