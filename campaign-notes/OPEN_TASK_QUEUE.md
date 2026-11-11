# Open task queue (work in order; GPU only when the task needs it)

Do **not** run filler ETS loops. When a GPU task finishes, start the next item that needs the alloc. Login-side coding can proceed while waiting for PD nodes.

**Batching doctrine:** see `BATCHING.md`. Bank on K/CTA latency hiding + pack; do **not** sell FP64 TLM GEMM as speedup; POD/DEIM rejected; FP32 tangents only after plateau test.

| # | Task | Needs GPU? | Status |
|---|------|------------|--------|
| 1–16 | Tier 3.1 gauge→χ→gt→K→At TL gates + 3.2 Richardson | mixed | **done** (see prior rows in git history / TIER3_TL_PLAN) |
| 17 | Tier 3.3 sep-cost scout (reused TPID) | yes | **provisional** (`TIER33_ORBITS.md`) |
| **B0** | **2.1 NCU on `eval_rhs3_pt_async`** (regs/spills/occ) | yes (short) | **done** (56 regs; **local spill heavy**; occ 17% **smem-limited**) |
| **B1a** | PT_GMEM / F32SU A/B vs ASYNC | yes | **done** (GMEM **NO-GO**; F32SU −8% ETS, **not** C_HAM bit-id) |
| **B1-defer** | deferred DD seconds in PT_ASYNC | yes | **NO-GO** (+7.7% ETS on GB200; same as A100) |
| **B1-scoped** | section-scoped PT (restencil late stages) | yes | **NO-GO** (+123% ETS) |
| **B1-lb** | `launch_bounds(343,3)→(343,1)` on PT_ASYNC | yes | **done** (~**−15%** ETS; keep) |
| **B1** | **2.1 live-range/CSE without restencil** (for K/CTA headroom) | mixed | **CSE NO-GO**; **while-hot keep opt-in** (q8 −0.7%, q1 −0.23%; `tier21_whilehot_ab.md`) |
| **B2** | **K members/CTA interleave** — speedup/member vs K @ 64/256/2048 | yes | **baseline done** (`tier0/tier21_ensK_baseline.md`; small-nblocks only) |
| **B3** | Pack K ghost msgs + one RHS launch; measure fine-level | yes | **RHS done** (2.89×@32bl); pack API in; **q8+q1 pack NO-GO** (0.5–0.9×) — park pack wire until few-block LTS |
| 16b | Gt TL + Ricci/lap pieces; fused emit | mixed | **Gt+lap+toyRicci+fused emit 16/16**; full Ricci/C³ next |
| 18 | Tier 3.3: re-TPID per sep / TARGET_ORBITS | mixed | **done scaffold+re-TPID N=3/5/8 + short cost**; longer cost next |
| T_fp32 | FP32-tangent plateau vs Tier 1.5 FD plot | yes | later (after B2; not first) |

Automation: when a new idev goes `R`, run the **next queued GPU task**, not a keep-alive cycle.

## Update 2026-11-09 (campaign continuation)
| item | status |
|------|--------|
| B2 K/CTA interleave | **closed** (measured ~1.0-1.15x at production block counts; see BATCHING.md) |
| B3 pack K ghosts | closed (no-go at these sizes) |
| Remesh: GPU BH_WAMR + device WAMR + block-setup O(N^2) fix | **done, default ON** (adaptive production-depth q8 380.8 -> 190.4 s) |
| Extraction (shell-only, OMP, cached sampler) | done (opt-in `DENDRO_GW_SHELL_ONLY=1`) |
| init_grid initial-data evaluation | **done**: 584 s -> 74 s (`DENDRO_INIT_OMP`, default on) |
| var-batched unzip CTA | **done, no-go** (+5..13% slower; `DENDRO_UNZIP_VB`, off) |
| FP32-tangent plateau test | later |
| Tier 3.3 q8 orbits-vs-cost | not started |
| fine-resolution waveform FD | not started |
| GPU-vs-CPU long-run physics comparison with BH_WAMR | not started |

## 2026-11-09 late: queued, awaiting nodes (cluster queue congested ~1h)
Auto-launch armed (`watch_batch.sh` -> `run_next_batch.sh`, logs `tier0/watch_batch.log`, `tier0/batch.done`):
1. frozen-mesh CPU vs GPU physics (`run_physcheck.sh cpu0 6.4`) — see PHYSICS_CHECK.md
2. OpenMP thread sweep at production depth, OMP=4/8/16/36 (`prod_TS*`): init_grid, remesh Mesh ctor, extraction constraint loop scaling; plus eager-vs-lazy host unzip vectors (`TS8E`)
3. TwoPunctures cold vs warm start (`run_tp_warm.sh`; `DENDRO_TPID_WARM_START=<fiducial *_tpid_sol.bin>`), thread scaling 8 vs 36 — a cold TP solve at 4 threads is ~475 s/member (3 Newton iterations from |F|=5.5e-3); warm start expected ~1 iteration.
Code built but UNTESTED: TPID warm start; lazy host unzip vectors after remesh (`DENDRO_EAGER_HOST_UNZIP=1` restores eager).
Then: fine-resolution waveform FD (needs recorded fine fiducial + 4 replays; est >1 h of node time).

## Storage rule
All run output goes to `$SCRATCH` (`~/Dendro-GR-runs` is a symlink to `$SCRATCH/Dendro-GR-runs`). Home quota is ~46 GB.

## INVERSION ROADMAP (added 2026-11-10) -- ON HOLD: the user is deciding the inference strategy (goal: full-order-model inference or another guaranteed-high-fidelity route, NOT lower-order surrogates). Do not plan Phases C-E further until they come back; Phase B/F items remain valid prerequisites.
Priority order: (1) non-eccentric injection-recovery, (2) real-event replication (high-mass, e.g. GW150914-like), (3) eccentricity as an inferred parameter.

### Phase A -- finish the solver side (IN PROGRESS)
- A1 resolution/cost study batch 3 (cap-13 ref, cap-14 to T=80, GW_REFINE_WTOL and AMR_FAC variants; see RESOLUTION_STUDY.md) -> choose the cheapest config that meets an error target
  (target needs the user's SNR/noise assumption: mismatch ~ 1/(2 SNR^2)).
- A2 junk-radiation / observing window decision; multi-radius extrapolation of psi4; check starting separation vs cost (q1: inspiral ~190 M).
- A3 FD Jacobian at the chosen config for the parameters actually inferred first: mass ratio, then aligned spins; extend to merger+ringdown window; choose eps; re-record policy for the frozen mesh.
- A4 RIT/SXS catalog comparison of one waveform (needs a catalog file from the user) -> real model-error number.
- A5 engineering: checkpoint/restart for long runs, unzip+ghost kernels (~34% of GPU time), remesh interval, parameterize R_orbit (currently hard-coded 8 in tolerance function 6).

### Phase B -- NR waveform product (psi4 -> h)
- B1 pipeline psi4_lm(t; r) -> extrapolate across radii -> h_lm(t) (fixed-frequency integration), junk cut, taper, alignment; test vs resolution.
- Scaling facts to exploit: total mass M and all extrinsic parameters (distance, inclination, polarization, sky position, time, phase) are ANALYTIC for a given dimensionless NR waveform; only the intrinsic dimensionless
  parameters need simulations: q (+ spins, + eccentricity/anomaly later).  Non-eccentric aligned-spin case: p = 1..3 intrinsic parameters, so a Jacobian by central FD is only 3..7 runs.

### Phase C -- waveform model for the inference code (bilby-style)
- Plug into bilby as a custom time-domain (or frequency-domain) source model that returns h_plus/h_cross from h_lm and the analytic extrinsics (bilby supports custom WaveformGenerator source models; verify the current API:
  its own time-domain example was reported broken (issue 290) and time-domain models can mismatch the frequency array/f_max (LIGO DCC note G2200203) -- test both).
- Direct NR-in-loop per likelihood call is impossible (hours per run), so the model used inside a sampler must be one of: (i) local linearized model h(theta0)+J dtheta from the FD Jacobian (Fisher/Laplace/Gauss-Newton,
  valid near the injection), (ii) an emulator trained on an ensemble of NR runs over a parameter box (surrogate-style), (iii) ensemble methods with NR in the loop at a limited number of evaluations.
### Phase D -- injection-recovery (first inversion target)
- D1 zero-noise injection, same model: recovers truth; compare Fisher widths with posterior widths.
- D2 zero-noise injection from a HIGHER-resolution NR run recovered with the lower-resolution model: the bias is the NR error budget.
- D3 Gaussian-noise injection (design PSD, SNR ~20-30), repeated for a P-P plot (calibration).
### Phase E -- real event replication (after D): high-mass event (few in-band cycles fit NR length), GWOSC data, PSD estimation, compare with LVK posterior.
### Phase F -- eccentricity as an inferred parameter (after the non-eccentric inversion works)
- F1 initial-data generator (e, mean anomaly, q, spins) -> positions/velocities (BSSN_BH1/BH2 X,V; TwoPunctures momenta); id.py is the low-eccentricity PN estimator only. Verify the V->momentum mapping.
- F2 R_orbit parameterized + resolution check for wider orbits; F3 eccentricity estimator from BH trajectories/waveform; F4 validation vs an RIT/SXS eccentric waveform;
  F5 cost (wider start separation => longer inspiral); F6 add (e, anomaly) to the FD Jacobian / emulator.

## SOLVER OPTIMIZATION PRIORITIES (current focus; forward solve + sensitivities, minimize time to solution)  -- written 2026-11-10
Measured shares (production depth q8, nsys, GPU kernel time 18.7 s over ~300 steps, contracted CAHD):
  main fused RHS 32% (6.1 s) + boundary RHS 6% | unzip (__unzip_cg1__2) 22% | ghost pack/unpack (__read_rb/__write_sb) 23% | CAHD 13% | zip+axpy 3%.
  Host side: remesh ~16% of stepping (1.1 s per remesh, every 50 steps; 58% of it state-independent mesh build), startup ~57 s.
  The cap-12 case on ONE GPU (~10k blocks) is GPU-bound: ~0.8 s/step => ~75 us per block-step.
Candidates, ordered by (expected payoff x ease):
 1. Numerics knobs that change the STEP COUNT (time to solution ~ 1/CFL): CFL 0.25 -> 0.35..0.5 (RK stability + waveform accuracy test), remesh interval 50 -> 100 (-8%), KO/AMR settings. Cheap, needs a short accuracy check.
 2. Fuse CAHD into the main RHS kernel (the main kernel already forms the Ricci pieces): removes most of the 13%.
 3. Unzip + ghost exchange (45% of kernel time, memory-bound gathers over padded 13^3 blocks for 7^3 interior points, ~6.4x redundancy):
    - larger blocks (19^3 = 2x2x2 elements merged: ~2.2x redundancy) = the old plan item 2.6 (not started),
    - exchange only the variables whose derivatives are needed, fuse zip/unzip/ghost packing around RK stages.
 4. Remesh: device-resident grid transfer, avoid no-op remeshes, share mesh construction across ensemble members (<= ~9%).
 5. Sensitivities: with frozen-mesh replay the 2p+1 FD members are independent, so Jacobian wall time ~ fiducial + one replay if enough GPUs are available
    => optimizing a SINGLE forward solve is the same lever as optimizing the sensitivities; strong-scaling one run over more GPUs (efficiency 0.56-0.64 at 4 GPUs, earlier
    measurement) trades throughput for time-to-solution; a tangent-linear model would give ~one-run wall time too (decided not needed for now).
 6. Keep: accuracy-vs-cost study (resolution/knobs) because it sets how many blocks are needed at all.

## HIGHER-ORDER DERIVATIVES (user request 2026-11-10: build towards Hessians; RMHMC would need 3rd order) -- design notes
Cost model for p inferred parameters, each forward member ~ one full solve (independent members; frozen-mesh replay):
 - FD gradient (central): 2p+1 members.  FD Hessian (central, function values): ~2p^2+1 (p=3: 19; p=9: 163).  FD 3rd derivative: O(p^3) (p=3: ~100; p=9: >1000).
 - Gauss-Newton / Fisher needs only J (first order).  RMHMC with a Fisher metric needs dG/dtheta = second derivatives of the MODEL (waveform), i.e. the Hessian of a vector output.
 - Forward-mode tangents: J costs p tangent solves; Hessian costs ~p(p+1)/2 second-order tangents => same O(p^2) as FD (no cost win, but exact: no eps/noise tuning).
 - Reverse mode (adjoint) wins asymptotically only for a SCALAR objective (gradient ~O(1) solves; Hessian-vector products ~O(1) each => Hessian ~p HVPs), but needs checkpointed trajectories
   (~1e5 steps), the adjoint of remeshing/grid transfer (frozen mesh => transpose of fixed linear maps), CAHD/SSL/KO/BCs, and adjoint stability of the puncture gauge is unknown.
 - Exact derivatives of the discrete solver can be generated from the sympy RHS (CodeGen), as started in the tangent-linear stubs (cuda/src/tl_*); second order = differentiate the tangent expressions (expression swell: not measured).
 - Smoothness prerequisite: frozen mesh + deterministic runs; the first-derivative plateau is clean (4.8e-5, fine, r=25); second derivatives amplify noise ~ roundoff/eps^2, so a plateau test is needed.
Tasks:
 - H1 (cheap, soon): second-derivative FD plateau test with p=2 (BH1 mass, BH2 mass): 9-run stencil per eps (center, +-e1, +-e2, 4 corners) at two eps, coarse then fine; report diag/mixed Hessian plateau vs eps.
 - H2: size the tangent/second-order codegen (count ops of d(RHS)/d(theta) and d2): decide AD vs FD after the user's inference strategy.
 - H3: keep options open: keep kernels sympy-generated where possible, keep frozen-mesh replay and determinism, FP64.

## Step-count knob tests (2026-11-10, cap 12, q1 production adaptivity, T=70; 3 runs concurrent on one node)
- CFL 0.25 / 0.35 / 0.5 (remesh interval scaled to constant time): ETS 1578 / 1463 / 1336 s -> only -7% / -15% despite 2x fewer steps, so wall time is NOT dominated by step count (remesh/mesh rebuild + extraction per remesh dominate, same count of remeshes per unit time). Waveform vs CFL0.25: max|dphase| 0.38 / 0.30 rad, max|dA/A| 0.42 / 0.44 at r=25 -- large, but cap 12 is unconverged and sensitive; no evidence that CFL is a free win. Verdict: do not raise CFL; next lever is remesh cost/interval (test run_rmf.sh: interval 50/100/200).
- Remesh interval 50/100/200 (CFL 0.25, cap 12, T=70): ETS 1444 / 1331 / 1259 s (-8% / -13%). Error vs cap-13 reference (psi4_22 r=25, t in 20..70): max|dphase| 0.29 / 0.82 / 0.81 rad -> intervals >50 make phase error ~3x worse at cap 12; reject. CFL 0.35/0.5 vs p13: 0.20/0.16 rad (i.e. within the cap-12 scatter, 0.29 for the baseline). Conclusion: remesh interval stays 50; CFL not worth raising on this evidence (<=15% gain); a proper test needs cap 13 (revisit only after the FD/accuracy decisions). Wall time is dominated by per-remesh/mesh work, so the real lever is cheaper remeshing (device grid transfer, avoid no-op remeshes), not fewer steps.

## Derivative order needed by Riemannian-manifold samplers (for the full-adjoint vs second-order decision)
Model h(theta) (waveform), log L = -1/2 |d - h|^2_n. Fisher metric G = J^T C^-1 J, J = dh/dtheta (first derivatives only).
- MALA with constant/preconditioned metric: needs grad log L = J^T C^-1 (d-h): first order.
- Simplified manifold MALA / RMHMC with the (expected) Fisher metric: need dG/dtheta_k = (dJ/dtheta_k)^T C^-1 J + J^T C^-1 (dJ/dtheta_k), i.e. the model Hessian d2h/dtheta2 (second-order sensitivities of the waveform), NOT third derivatives of log L. The "3rd order" requirement only arises if G is the observed Hessian of -log L (then dG needs d3 log L, i.e. also d3h-free but includes d2h terms and products; still only up to d2h plus J if one uses Gauss-Newton and drops the residual-curvature term).
- So: with Gauss-Newton/Fisher metric, p intrinsic parameters need J (2p+1 solves by central FD) and d2h (p(p+1)/2 mixed second differences ~ 2p^2+1 solves). For p=2 (q, chi_eff): 5 and 9 solves; p=3: 7 and 19. Extrinsic parameters (distance, phase, time, inclination, polarization, sky, total mass) have analytic derivatives, so they do not add solves. A full adjoint is NOT required for this route; it would only help if p grows past ~6-8 (then reverse mode gives gradients at O(1) solves but second derivatives still need forward-over-reverse, ~p solves).
- FD second-derivative accuracy is the open issue: need h2 = O(eps^-2) error amplification; plateau test H1 (cap 13, p=2) will decide; if FD d2h is too noisy, forward-mode tangent+second tangent (exact, same asymptotic cost) is the fallback.

## !! CPU-pinning bug in single-rank launches (found 2026-11-11 00:45)
`mpirun -np 1 ...` (Open MPI/PRRTE default --bind-to core) pinned each solver to ONE core; with 4 concurrent solvers per node they all landed on CPU 0 (Cpus_allowed_list: 0, ~25% CPU each; OMP_NUM_THREADS=8 threads time-sliced on one core). Fix: `mpirun -np 1 --bind-to none` (now in all run_*.sh using -np 1; solver then runs at ~630% CPU).
Consequences: (1) every *wall-time* number from concurrent single-rank runs (fiducial cap 13 = 5916 s, cap 12 replays 4750 s, CFL/rmf tests) is inflated by host-side (mesh/remesh/OMP) serialization; GPU-only microbenchmarks and per-kernel timings are unaffected. (2) The "wall time is dominated by remesh, not steps" conclusion drawn from the CFL/rmf tests above is NOT established -- the remesh work was running on one shared core. Accuracy results (phase/amplitude errors) are unaffected. Redo timing with proper binding before choosing step-count knobs. (3) Earlier thread-tuning conclusions ("36 OMP threads good") may also have been measured under pinning; re-check.

## Step-count knobs REDONE at cap 13 with correct binding (2026-11-11); supersedes the cap-12 pinned results above
Reference cap 14 (p14u). psi4_22 r=25, t in 20..110. ETS = stepping wall on one GPU (3 concurrent runs/node).
| config | ETS [s] | max|dphase| [rad] | max|dA/A| |
| CFL 0.25, remesh 50 (baseline p13u) | 784 | 0.0073 | 0.0063 |
| CFL 0.35, remesh 36 steps | 598 | 0.038 | 0.011 |
| CFL 0.5, remesh 25 steps | 495 (-37%) | 0.0029 | 0.0033 |
| CFL 0.25, remesh 100 | 680 (-13%) | 0.0072 | 0.0062 |
- CFL 0.5 stable through merger+ringdown and no less accurate than baseline; wall time now scales with steps (the pinned runs hid this). CFL 0.35 is a worse outlier (0.038 rad): shows adaptation/mesh-timing scatter of a few 1e-2 rad between nominally equivalent configs -- must be quantified before trusting any step-count knob (more samples: CFL 0.4/0.6/0.7 running).
- Remesh interval 100 steps at cap 13 costs nothing in accuracy (unlike cap 12) and saves 13%.
- More cap-13 variants (same reference p14u; error measured only where |z_ref|>10% of max, since the interpolated phase at the inspiral null t~47 (|z|~1e-8) is meaningless -- the "0.038 rad" CFL 0.35 outlier above was that artifact):
| config | ETS [s] | max|dphase| (masked) | max|dz|/max|z| |
| CFL 0.25 rm50 (baseline) | 784 | 0.0073 | 0.0049 |
| CFL 0.4 rm36 (time-scaled) | 563 (-28%) | 0.0032 | 0.0031 |
| CFL 0.5 rm25 (time-scaled) | 495 (-37%) | 0.0022 | 0.0023 |
| CFL 0.5 rm50 | 415 (-47%) | 0.0023 | 0.0023 |
| CFL 0.4 rm80 | 464 (-41%) | 0.0032 | 0.0031 |
| CFL 0.6 rm20 | 454 (-42%) | 0.0100 | 0.0059 |
| CFL 0.7 | crashed (unstable) | - | - |
Recommendation (q1 only so far): CFL 0.4 + remesh 80 steps (-41%, error below baseline) as the new production default; CFL 0.5 + remesh 50 is -47% but only 1.2x from the instability edge (0.7 fails, 0.6 degrades). Before making it the default for other q/spins: repeat for q=4 (and one spinning case), since the shift/advection CFL margin depends on the configuration. Note the FD harness is unaffected (members replay the fiducial's mesh but use the same CFL; the fiducial and members must share the CFL).
- q=4 check (production q4.par.toml, cap 14, T=50, TwoPunctures bare masses 0.2/0.8; run_q4_cfl.sh): CFL 0.4+rm80 and CFL 0.5+rm50 are stable and ETS drops 1129 -> 676 s (-40%) / 606 s (-46%); psi4_22 agrees with the baseline to 2e-6 / 3e-6 relative. CAVEAT: T=50 only covers the junk-radiation window (|psi4| ~3e-8, no inspiral signal yet), so this verifies stability and cost, not inspiral accuracy; q=4 inspiral accuracy (hundreds of M) is still to be tested before the CFL change becomes a default for q != 1.

## Eccentricity prep, step 1 (2026-11-11): BSSN_WAVELET_TOL_R_ORBIT
The strictest-refinement radius of wavelet-tolerance function 6 (formerly hard-coded `R_orbit = 8`) is now a parameter `BSSN_WAVELET_TOL_R_ORBIT` (default 8.0, optional in the par file; BSSN_GR/src/grUtils.cpp, parameters.cpp/.h). Regression: default reproduces the old binary at 2e-11 relative (psi4_22, T=12, cap 13). For an eccentric binary set it to ~ the apocenter radius + margin. (dataUtils.cpp has a second, separate R_orbit = f*dBH + 8 in the initial-grid refinement; it follows the orbit separation and needs no change for now.)

## Full-step GPU profile, production q1 cap 13, T=14 (nsys, 1 GPU, 861 steps, 3444 RHS calls) -- 2026-11-11
GPU kernel time shares (total 67.7 s): main RHS (PT_ASYNC interior) 37.3% (7.3 ms/call) | unzip `__unzip_cg1__2` 25.3% (4.9 ms) | boundary RHS `eval_rhs3` 13.3% (2.6 ms) | CAHD 18.5% = `cahd_fused_c` 11.8% (2.3 ms) + `cahd_chi_rhs` 3.6% + `cahd_derivs` 3.1% (boundary-block path) | axpy 2.2% | zip 2.0% | SSL 0.7% | enforce 0.6% | WAMR 0.1%.
Kernel time/step = 79 ms vs 116 ms wall/step in the cap-13 run (ETS 784 s / 6752 steps) -> ~30% of wall is outside GPU kernels (host syncs, extraction, remesh + grid transfer, MPI single-rank overhead): next profile item (nsys timeline gaps, DENDRO_TIME_REMESH totals over a full run).
Priorities from this table: (1) unzip (25%): bandwidth-bound gather of padded 13^3 blocks; options 19^3 merged blocks / only unzip evolved vars needed / fuse with zip; (2) CAHD 18.5% -- the boundary-path pieces (chi_rhs+derivs, 6.7%) can be folded into one pass, and a fused-into-main-RHS version could remove the extra read of the 24 variables (~half of the 11.8%); (3) boundary RHS 13%; (4) host/remesh 30% of wall. Main RHS (37%) was already tuned (launch bounds, PT_ASYNC).

## Pinned host-memory pool (patches/dendrolib/pinned_pool.patch; dendrolib include/device.h) -- 2026-11-11
nsys showed cudaMallocHost 6.4 s (143 calls, up to 270 ms) + cudaFreeHost 1.1 s = ~7.5% of a 100 s cap-13 run: `host_malloc/host_free` allocate/free pinned buffers at every remesh. Added a size-bucketed (x1.25) pool with an eviction cap (DENDRO_PINNED_POOL_MAX_GB, default 24; DENDRO_PINNED_POOL=0 disables). A/B on the same binary, cap 13 T=14 on one GPU: ETS 98.4 s -> 92.0 s (-6.5%), waveform identical to roundoff (max |dz| 1.7e-17 at |z| 7e-7, the same level as run-to-run noise). On by default.

## Where the non-kernel 30% goes (nsys gap analysis, cap 13 T=14)
GPU span 99.8 s, kernel-busy 67.7 s, idle 32 s: 27.8k+16.9k gaps <100 us total 0.5 s (launch gaps are negligible), 843 gaps of 1-10 ms 1.5 s, and 58 gaps of 0.1-1 s totalling 29.6 s = the remesh events (mesh reconstruction + grid transfer on the host, ~0.5 s each at this size, ~1.8 s late in the full run: 135 remeshes ~ 250 s of the 784 s cap-13 ETS). => remesh host cost is ~30% of wall (not 9%), and it is what the FD replay members also pay (mesh rebuild from recorded flags). Levers: (a) fewer remeshes (interval 100 = -13%); (b) cheaper rebuild (device-side or incremental Mesh construction / grid transfer); (c) shared Mesh across K ensemble members in one process per GPU: GPU-seconds per member 1 -> (0.7 + 0.3/K), i.e. -22% at K=3, -25% at K=6 (needs the ensemble context, not just a batched RHS).

## Remesh host cost: OpenMP intergrid transfer (patches/dendrolib/igt_omp.patch) -- 2026-11-11
Per-remesh phases (cap 13, DENDRO_TIME_REMESH=1): balance+partition 0.2-0.26 s, Mesh ctor 0.17-0.36 s, interGridTransferSendRecvCompute 0-0.41 s, host intergrid transfer 0.4-0.5 s (serial loops over 24 variables x 10k elements).
Findings: (1) **the `.cu` translation units were compiled WITHOUT OpenMP** (CMAKE_CUDA_FLAGS lacked `-Xcompiler -fopenmp`), so every `#pragma omp` in the dendrolib headers instantiated from CUDA code (e.g. intergrid transfer) was inert. Added `-Xcompiler -fopenmp` to the build (`cmake -DCMAKE_CUDA_FLAGS='-I... -Xcompiler -fopenmp'`); with the fast path disabled results are identical to the old binary (2e-17) and timing unchanged. (2) New single-rank INJECTION fast path in `Mesh::interGridTransfer` parallel over variables with per-thread scratch (must use the thread-safe `getElementNodalValues(..., isDGVec, im1, im2)` overload: the first attempt shared the RefElement scratch and corrupted the mesh -- 7176 -> 48105 elements at the 2nd remesh). Fixed version: total intergrid transfer 9.5 s -> 2.35 s over the 20 remeshes of the T=14 test (4x), waveform identical to the old binary at 2e-17, ETS 90.5 -> 86.9 s (-4%; ~ -7% expected over a full 135-remesh run). DENDRO_IGT_OMP=0 disables; DENDRO_PINNED_POOL=0 disables the pool.
Remaining per-remesh host cost (balance+partition, Mesh ctor, SendRecvCompute ~0.4-0.9 s) is the next target; the Mesh ctor/balance are Dendro-internal octree code.

## Full-length validation of the optimized binary (pinned pool + OpenMP intergrid transfer), cap 13, T=110 -- 2026-11-11
| config | ETS [s] | vs cap-14 reference max|dphase| / max|dz|/max|z| |
| CFL 0.25, remesh 50, old binary (p13u) | 784 | 0.0073 / 0.0049 |
| CFL 0.25, remesh 50, new binary | 634 (-19%) | 0.0073 / 0.0049 (waveform identical to p13u: max|dz| 1e-16 at |z|~3e-5) |
| CFL 0.4, remesh 80, new binary | 408 (-48%) | 0.0032 / 0.0031 |
=> one cap-13 q1 forward solve to T=110 is ~7 min of GPU stepping (down from 13 min), 5-6 min with the replay-based FD members. Software-only changes are bit-for-bit (roundoff) neutral; the CFL/remesh change is a measured, accuracy-neutral (q1) schedule change that still needs the q != 1 inspiral check before becoming a default.

## q=2 check of the CFL 0.4 / remesh 80 schedule (cap 13, T=110; run_q_cfl.sh) -- 2026-11-11
ETS 1087 s (CFL 0.25, rm50) -> 690 s (CFL 0.4, rm80), -37%; but the two waveforms differ by max|dphase| 0.026 rad, max|dz|/max|z| 0.025 (r=25 shell, |z|>10% of max; q=2 merger at t~102) -- 3x-8x larger than at q=1 (0.003). Cap 13 is itself less converged for q=2 (smaller BH), so this difference may just be cap-13 discretization noise, or a genuine schedule effect: NOT validated. Policy: keep CFL 0.25 / remesh 50 as the default for every q != 1 until a per-q convergence run (cap 14 reference, below) shows the schedule change is below the cap-13 error. (q=1: validated, 0.0032 rad.)

## RESOLUTION of the schedule question (2026-11-11): CFL must stay 0.25; remesh interval 80-100 is free
Same-resolution comparisons against the baseline (CFL 0.25, remesh 50), psi4_22 r=25 shell, |z|>10% of max, T=110:
| case | change | max|dphase| | max|dz|/max|z| | ETS change |
| q2 cap 13 | remesh 80 only (CFL 0.25) | 0.0001 | 0.0001 | -(~10%) |
| q2 cap 13 | CFL 0.4 + remesh 80 | 0.026 | 0.025 | -37% |
| q4 cap 14 | CFL 0.4 + remesh 80 | 0.048 | 0.020 | -38% (1924 vs 3089 s) |
| q1 cap 13 (vs cap-14 ref) | CFL 0.4 + remesh 80 | 0.0032 | 0.0031 (baseline 0.0073) | -48% |
=> the remesh interval is harmless (1e-4 rad at q=2, 7e-3 = baseline at q=1); a larger CFL changes the answer by 0.03-0.05 rad for q = 2, 4 (a systematic RK4 time-integration shift), and the *apparent improvement* at q=1 (error vs cap 14 decreasing with CFL up to 0.5) is a compensation of a negative time error against the positive cap-13 spatial error, not convergence. Policy: CFL 0.25 (or lower) in all production runs; remesh interval may be raised to 80-100 steps (-10 to -13% wall). Do not use the CFL speedup. (A genuine step-count lever would need a higher-order/different time integrator or LTS -- not pursued.)

## !! Resolution requirement is q-dependent (2026-11-11): cap 13 is NOT enough at q=2
q=2, baseline schedule, psi4_22 r=25 shell, |z|>10% of max, T=110 (merger t~102): cap 13 vs cap 14 reference: max|dphase| 0.175 rad, max|dz|/max|z| 0.177 (at q=1 the same comparison is 0.007 rad). ETS cap 13 1087 s, cap 14 1786 s. The relevant resolution is that of the smaller BH (mass 1/3 at q=2 vs 1/2 at q=1): roughly one extra level per factor 2 in 1/m_small (cap_needed ~ 13 + log2(0.5/m_small)); the production q4/q8 pars use MAXDEPTH 16 for that reason. Earlier conclusions ("cap 13 production, cap 14 verification, 0.007 rad") hold only for q=1. For an inversion over mass ratio the resolution must be set per q (or per m_small) with its own convergence check; cap 15 for q=2 is running (q2cfl/base_c15_T110) to get the q=2 error of cap 14.
- Mass-aware refinement levels work (q=2, T=110, baseline schedule): "eq13" = BH1 (m=1/3) at level 14, BH2 (m=2/3) at level 13 (level_k = 13 + log2(0.5/m_k), small BH rounded up) reproduces the uniform level-14 run (both BHs at 14) to 1e-4 rad / 5e-4 relative, at 1459 s ETS vs 1786 s (-18%). It is far from uniform level 13 (0.19 rad). So the resolution that matters is per-BH points-per-mass: use L_k = L_ref + log2(0.5/m_k) (L_ref = q=1-equivalent level) for BH{1,2}_MAX_LEV, MAXDEPTH = max_k L_k. Production pars q2/q4/q8 do a similar thing (16/15, 16/14, 17/14) but with a larger L_ref (~15). run_q_cfl.sh now takes per-BH levels (5th/6th variant fields).
- q=4 mass-aware levels, first attempt FAILED: levels (BH1=15, BH2=12) [L_ref=13] crashed at t=25.5 (signal after a remesh; mesh ran away 57k -> 64k -> 70k elements with 99.9% hanging) -- too coarse a level for the heavy BH (m=0.8) in the initial/junk phase, or a refinement runaway. (16,13) was too slow to finish (t=18 after 55 min; killed). Mass-aware levels beyond a one-level reduction of the large BH need care: keep the large BH within 1-2 levels of the small one (production pars use (16,14) at q=4), and treat q>=4 as a separate (expensive, multi-hour per run) regime; the q>=4 convergence study needs a proper long allocation and checkpointing.


## !! CORRECTION 2026-11-11 08:45 -- see STATE_2026-11-11.md top: the T=110 comparisons above test the junk-radiation transient (shell r=100, retarded time <= 10 M; BHs have not merged: q=1 r=3.6 at T=110), not inspiral phase accuracy. Redo with T=260 and retarded-time windows is running (p1{2,3,4}L_T260).

## Root cause of the >65535-element crash (2026-11-11 11:45): zip kernel gridDim.y
`MeshGPU::zip_cg/zip_dg` launch `dim3(m_zip_grid[0], m_zip_grid[1], dof)` with m_zip_grid[1] = blk_list.size() (= #elements for the 1-element-per-block GPU layout) on gridDim.y (hardware limit 65535) -> cudaErrorInvalidValue printed by check_last_error as "!! invalid arguments" and an abort. Three runs died at the first step after a remesh to >65535 elements (p13L 65717, p14L 66032, q4 levels (15,12) 70442); all runs below 62.4k completed. Fix (patches/dendrolib/zip_grid_over65535.patch): grid = (#blocks, max elems per block, dof), kernels read blk from block_id_x and ele_id from block_id_y. Binary built; the verification run (a mesh > 65535 elements) is pending a node. This limit affects every post-merger production run at the resolutions that matter; other y/z-limited launches (unzip grids, write_sb, ghost pack) use small y.
- Zip-grid fix VERIFIED: run bigE (q1, cap 15, MINDEPTH 5, AMR_R 1.15) reached 72864 elements (> 65535) and completed (T=1) without the "invalid argument" crash (before the fix every run died at the first step after a remesh above 65535). A mesh of ~81k+ elements hit the next limit: DVector 32-bit index ("5.6e9 entries do not fit the vector's index type"), i.e. 24 vars x 2197 padded points x elements > 2^32 at ~81.4k elements per rank.
## 64-bit index migration (user request 2026-11-11)
Widened to 64 bit: GPU `DVec` / `ts::Ctx` index type (unsigned int -> unsigned long, bssnCtxGPU.cuh), kernel/launcher unzip sizes (`szpdof_uz`, `sz_per_dof`, `unzip_sz`, `cg_sz`, strides: size_t in bssn_kernels.cuh, mesh_gpu.cuh, bssnCtxGPU.cu) and the device helper sizes (device_malloc/host_malloc/memcpy n: size_t in dendrolib device.h). Per-variable products such as var*unzip_sz are now 64-bit. Build + regression pending.
