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

## INVERSION ROADMAP (added 2026-11-10; solver work finishes first, then this)
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
