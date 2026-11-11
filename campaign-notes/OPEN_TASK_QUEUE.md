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
