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

## CAHD on the GPU (implemented 2026-11-10; production par files set BSSN_CAHD_C=0.06)
Adaptive CPU (BH_WAMR, remesh/10) with CAHD vs without, t<=14: psi4 l2m2 2.5e-3, l3m2 5.2e-3 rel-to-max, C_HAM +14%,
BH sep 2.8e-3 -> CAHD matters and grows with time (7.5e-5 / 2.1e-4 at t=6.4), so it is implemented, default ON when C!=0
(`DENDRO_GPU_CAHD=0` = original-GPU behaviour, with a warning).
Implementation: cuda/gen/gen_cahd.py generates the device expression from src/bssneqs_SSL_HD_dxsq.cpp (identical by
construction: chi_rhs += -C*dx_i^2*(chi/12)*B/(dt*(10 dx_i^2+1)), dx_i = min block spacing, dt = BSSN_RK45_TIME_STEP_SIZE;
skipped on the physical-boundary face layers where the CPU radiative BC overwrites the RHS).
Pass 1 (`cahd_derivs`) reuses the stock per-variable derivative device code for chi, Gt, gt only (72 arrays, batched
workspace of 128 blocks); pass 2 (`cahd_chi_rhs`) evaluates the term.
Bug found/fixed during verification: derivative CTAs were selected with wrong variable IDs (assumed gt at 3-8, Gt at 18-20;
actual grDef.h layout: chi=1, Gt=3-5, beta=6-8, B=9-11, gt=12-17, At=18-23). Stage 1 hid it because the initial data is
conformally flat (all gt/Gt derivatives are zero); stage 2 exposed it (chi RHS off by 1e-5). Lesson: verify at stage >=2 /
evolved state, not only at step 0.
Verified (CPU vs GPU, identical stage-2 inputs 1e-15): stage-2 chi RHS 2.3e-14, state after 1/2/5 steps 1.5e-11.
Note: the stock GPU RHS kernels hard-code lambda={1,1,1,1}, lambda_f={1,0}; production par files use the same values
(a different parameter file would silently diverge).

## CAHD cost at production depth (q8, T=0.3 ~300 steps, 36 OMP/rank, same node, ETS = stepping phase)
CAHD off: 29.5 s | two-kernel (derivative workspace) first version: 41.7 s (+40%) | trimmed (chi/Gt/gt only, no KO, batch 512):
38.4 s (+30%; 37.8 s in the later A/B) | **fused single kernel (default, DENDRO_GPU_CAHD_FUSED=0 -> two-kernel): 36.0 s (+22%)**.
nsys (first version): cahd_chi_rhs 3.6 s + cahd_derivs 3.1 s = 29% of all GPU kernel time, i.e. about the cost of the whole
main RHS (eval_rhs3_pt_async 6.1 s). The derivative pass was HBM-write bound (72 full 13^3 arrays per block), but the fused
kernel only gains ~24% of the overhead: it uses 168 regs and spills ~2.4 KB stack/thread (622 temporaries + 72 derivative
values live), so the expression itself is the bottleneck. Further reduction needs the Hamiltonian expression restructured
(shorter live ranges / hand-written conformal Ricci scalar) -- not done. Physical-boundary blocks (bflag != 0) use the
two-kernel path (one-sided stencils). Accuracy of the fused kernel vs CPU: stage-2 chi RHS 2.9e-14, state 1.5e-11 after 5 steps.

## End-to-end (adaptive, remesh/10, BH_WAMR, t<=13.8): GPU (init-mesh fix + SSL + CAHD + BH_TRACK_FREQ=1) vs CPU (CAHD)
BH positions 5e-14, C_HAM 5e-11, psi4 l2m2 1.5e-11, l3m2 2.8e-11; 86 remeshes on both; GPU 165 s vs CPU 1262 s (36 ranks).
(GPU before the fixes, no CAHD: psi4 2.5e-3 / 5.2e-3 off.)

## CAHD theory notes (for kernel design)  -- source: arXiv:2404.01137 (Eq. 26) + CodeGen/bssn.py
- Added to the conformal-factor equation: chi_rhs += C * chi * (dx_i^2/(1+10 dx_i^2)) / dt * H  (the "WKB" variant used by
  the CPU code; paper form: -C*dt_n*[CFL0*ds_n/dt_n]*H_-).  H = chi*gt^{jk}R_jk - At_ij At^ij + (2/3)K^2 (vacuum).
- Why: H violations act like an effective energy density and are largest at the sharp lapse pulse / strong field near the
  punctures and persist at refinement boundaries; the term damps a constraint-violating mode exponentially and, because
  H -> 0 in the continuum limit, vanishes with resolution (prefactor ~ dx^2/dt ~ dx/CFL on fine levels; saturates at
  ~0.1/dt on coarse levels, hence the softening 1/(1+10dx^2)).  The dt-based scaling keeps the damping stable at any
  resolution (an alpha*H term would have a parabolic CFL limit ~dx^2).
- Consequence for cost: the term needs H pointwise EVERY RK stage from the stage state; H is cheap in contracted form
  (inverse metric, Christoffels, a few contractions). The generated 622-temporary expression was the problem (spills).
- Possible further savings (NOT exact vs CPU, opt-in only if measured error is acceptable): freeze H over the 4 RK stages
  (smooth slowly varying source), or skip levels where chi*dx^2*H/dt is below a threshold.  Not implemented.
- Exact algebraic identities used in the contracted H (validated to ~4e-13 vs the generated bracket, cuda/gen/ham_contracted.py):
  gi^{ij} g_ki d_j Gt^k = div Gt (only d_x Gt^x + d_y Gt^y + d_z Gt^z needed, not all 9);  gi^{ij} C1_{ijk} = (1/2) d_k ln det g;
  gi^{ij} Rphi_ij = (2/chi)(L - CalGt.grad chi) - 5/(2 chi^2) Q   (L = gi^{lm} d_l d_m chi, Q = gi^{lm} d_l chi d_m chi).
