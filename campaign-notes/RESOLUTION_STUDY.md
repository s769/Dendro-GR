# Accuracy-vs-cost study for the inverse problem (started 2026-11-10)

## Batch 1: smoke-file settings (wavelet-tol function 3, ramp r=20..220, AMR 0.1, GW radii 10..25), T=26, production physics, np=1 per GPU
Reference = cap 13.  cost = stepping-phase ETS.   (run_resstudy.sh, analyze_res.py)
| run | cap | wavelet tol | cost | r=25 max dphase | r=25 max dA/A | r=18 max dphase |
|---|---|---|---|---|---|---|
| c11 | 11 | 1e-5 | 747 s | 1.8e-3 rad | 2.5e-3 | 1.29 rad |
| c12 | 12 | 1e-5 | 2306 s | 1.9e-3 | 2.7e-3 | 0.19 |
| c12t4 | 12 | 1e-4 | 995 s | 2.4e-3 | 2.6e-3 | 0.18 |
| c13 | 13 | 1e-5 | 2881 s | ref | ref | ref |
Findings: loosening the tolerance 10x at cap 12 costs 2.3x less with no visible accuracy loss at either radius; at r=25 all runs agree to ~2e-3
regardless of cap (error floor, NOT resolution-limited: the T=26 window at r=25 is dominated by the initial junk radiation arriving at the sphere);
at r=18 the cap matters (c11 far off, c12 0.19 rad from c13).  T=26 cannot say anything about phase accumulation over the inspiral.

IMPORTANT: the smoke files do NOT mimic the production adaptivity.  Production (q1/q8.par): BSSN_USE_WAVELET_TOL_FUNCTION=6, ramp R0=3..R1=12
(tolerance 1e-5 only within r<3, 1e-3 beyond r=12), AMR_FAC 0.20, GW radii 50..100.  The smoke files (function 3, R0=20, R1=220, AMR 0.1) resolve the
wave zone and the outgoing junk shell at 1e-5, so the mesh grows steadily (6.7k -> 51k active elements by t=36 at cap 11) and single-rank remeshes take
50-140 s: that is the "mesh blow-up at T>27" seen earlier.  Cost/accuracy conclusions must come from production-like adaptivity (batch 2).
The FD smoothness conclusions (TIER1_FD.md) are unaffected (they are about the derivative), but they were taken with the smoke adaptivity.

## Batch 2: production-like adaptivity (q1.par.toml: tol function 6, RESTORE off, GW radii 50..100), T=110, np=1 per GPU
(run_resstudy2.sh; one shared TwoPunctures solve (49 s, grid 65x78x10); sample interval 0.5208 M for every cap; CAHD 0.06, SSL, BH tracking on)
| run | cap | notes | result |
|---|---|---|---|
| p11 | 11 | tol 1e-5 | FAILED to be useful: mesh grew 26k -> 68.6k active elements (max level only 9), remesh 236 s, stalled at t=45; killed |
| p12 | 12 | tol 1e-5 | complete: stepping 2812 s for 3370 steps (T=110), mesh ~9.8k elements throughout |
| p12t4 | 12 | BSSN_WAVELET_TOL 1e-4 | complete: 2903 s; waveform identical to p12 to 2.6e-11 rad / 2.4e-11 amplitude |
| p13 | 13 | tol 1e-5 | reached t~80 when the node expired (mesh ~10.4k elements, ~96 steps/min); rerun in batch 3 |
Findings:
* A depth cap that is too low for the adaptivity (cap 11) breaks the refinement strategy (broad refinement at coarse levels, cost explodes); cap >= 12 gives ~10k-element meshes.
* Tolerance function 6 (grUtils.cpp): BSSN_WAVELET_TOL only applies for r <= 8 (near the orbit); r in (8, R_min=50) log-ramps to BSSN_GW_REFINE_WTOL (1e-4);
  plateau at GW_REFINE_WTOL out to R_max; beyond that and for t < t_lim=max(r,(r+120)/sqrt2) the tolerance is BSSN_WAVELET_TOL_MAX (1e-3, "WAMR disabled").
  => BSSN_WAVELET_TOL is NOT an effective accuracy/cost knob under function 6 (1e-5 -> 1e-4 changes the waveform by 1e-11). Real knobs: depth cap
  (BH-region resolution), BSSN_GW_REFINE_WTOL (wave zone), AMR_FAC, BH AMR radii/ratio.
* Cost at cap 12: ~3370 steps for T=110 = 2812 s on ONE GPU (np=1) => a q1 inspiral to ~190 M is O(1.5-2 GPU-hours) at this level IF its accuracy suffices (unknown: needs the cap-13/14 comparison).
* The partial cap-13 series (t<=74) overlaps the arrival of the r=70/100 signal only marginally, so no cross-resolution phase number yet.

## Batch 3 (queued: whichever of the 5h/2h allocations starts first; watch_resstudy3.sh): p13 to T=110, p14 to T=80 (convergence rate on the shared window),
## g3c12 / g5c12 = cap 12 with BSSN_GW_REFINE_WTOL 1e-3 / 1e-5 (wave-zone tolerance sensitivity)

## RESULTS (2026-11-10 evening): convergence with the depth cap, production-like adaptivity (q1.par: tol function 6, AMR 0.2, radii 50..100), ID type 0 (TwoPunctures), np=1 per GPU
Runs: p12 (cap 12, T=110), p13 (cap 13, T=110), p14 (cap 14, T=80); analyze_res.py (name@T syntax), common sample times, phase offset removed at the first sample above 10% of the reference amplitude.
Cost (stepping-phase ETS on one GPU; p12/p13 ran with 4 concurrent processes on the node, p14 alone): p12 2812 s (3370 steps), p13 3064 s (6752 steps), p14 1728 s (9792 steps, T=80).
Waveform error of psi4_22 vs the cap-14 reference, window t<=80 (signal at r arrives at t~r, so only the first ~30 M of signal at r=50 and less beyond):
| radius | cap 13: max dphase / max dA/A | cap 12: max dphase / max dA/A |
|---|---|---|
| r=50 | 0.15 rad / 21% | 1.44 rad / 407% |
| r=60 | 0.021 rad / 7.9% | 1.31 rad / 387% |
| r=70 | 3.3e-3 rad / 0.6% | 0.58 rad / 89% |
Cap 12 vs cap 13 over the full T=110 window: max dphase 2.77 / 1.96 / 1.14 / 0.55 rad at r=50/70/90/100, amplitude errors 370% / 280% / 107% / 68%  => CAP 12 IS NOT USABLE (not converged).
Observed convergence order (R = diff(12,14)/diff(13,14) = 2^p + 1): p ~ 3 (r=50), ~6 (r=60), ~7.4 (r=70): consistent with the 6th-order scheme where the signal is resolved.
Implied error of cap 13 ~ diff(13,14) (0.15 / 0.02 / 3e-3 rad at r=50/60/70); implied error of cap 14 ~ diff(13,14)/(2^p-1) ~ 0.018 / 3e-4 / 2e-5 rad.
Conclusions: (1) cap >= 13 is the minimum for this setup, cap 14 for ~0.02 rad at r=50; (2) a q1 run to T=110 costs ~1-1.5 GPU-hours at cap 13-14 on ONE GPU, i.e. the forward solve is cheap enough for ensemble work
(far below the q8 depth-17 case); (3) caveats: window <= 80 M dominated by the junk-radiation transient at r=50-70, one configuration, the phase accumulation over the full inspiral and a real error target
(needs the noise model) are still open; r>=80 not usable in the t<=80 window.
Batch 3 status: p14 done; cap-12 variants (BSSN_GW_REFINE_WTOL 1e-3 / 1e-5, AMR_FAC 0.4) running on c116-001 (the first launch failed: node c111-022 has a different module tree, the env script aborted at `module load`).
Infrastructure lessons: (a) launching via `ssh node "nohup ... &"` sometimes loses the job -- use `(setsid nohup ... &)` and verify with ps; (b) a run directory is deleted when a run with the same name starts (batch-2 p13 survived only because the batch-3 launch failed).

## Batch 3 (cap 12 variants) + production-ID FD at cap 12 (2026-11-10)
- g3c12 (GW_REFINE_WTOL 1e-3), g5c12 (1e-5), a4c12 (AMR_FAC 0.4) vs p13 reference: g3/g5 are *identical* to p12 in psi4_22 (final dphase -0.549 rad at r=25): at cap 12 the depth cap, not the tolerance, limits the mesh. AMR_FAC 0.4 only reduces it to -0.43 rad. Cap 12 stays unusable; tolerance knobs are not the lever (~9% cheaper for g3/g5, 2.56e3 s vs 2.81e3 s ETS).
- Production-ID FD harness (run_fd_tp.sh, TwoPunctures with per-member warm-start solves, cap 12, T=110) works end to end: fid 1195 s wall, 4 concurrent replays ~4750 s wall each.
- Central-FD plateau (eps 0.001 vs 0.003, |dz|/|z| of psi4_22, r idx 5): 1e-6 (t<35, junk+early inspiral), 7e-5 (t=42), 5e-3 (t=52), 0.14 (t=62), 0.25 at merger. Linear in eps early. At cap 12 the merger happens at t~83 (unconverged), so the late-time plateau failure is not a trustworthy statement about FD itself; rerun at cap 13 (queued job fdtp13).
- NOTE: run through t<~60 only is smooth; analyze_wave_fd.py's "last 25%" window is post-merger garbage here (use analyze_wave_fd_dgr.py with a pre-merger window or ex-time-resolved table).

## Cap-13 production-ID FD + CPU-pinning correction (2026-11-11)
- With mpirun --bind-to none (see OPEN_TASK_QUEUE.md pinning note): cap-13 replays of 6752 steps take ~14 min each with 4 concurrent on one node (~9 steps/s); the earlier pinned runs were ~7x slower. Earlier ETS wall-times in this file for concurrent runs are inflated; accuracy numbers unaffected.
- Cap-13 (T=110, merger at t~84) central-FD d psi4_22/d m1 (bare mass of BH1, TwoPunctures per-member): eps 0.001 vs 0.003 plateau mismatch 6e-7 (t<42), 2e-5 (t=52), 2e-4 (t=62), 6e-4 at t=110 (ringdown). Excellent: first derivatives via FD are reliable at cap 13 with production initial data.
- Second derivative d2 psi4/dm1^2 from the same members, (z+ + z- - 2 z0)/eps^2: plateau 3e-3..5e-4 for t in 30..42, 2e-2..5e-2 for t in 52..62, 0.5 in the ringdown (eps 0.003 too large; the O(eps^2) truncation of the second difference is visible). Use eps ~0.001 or smaller for Hessians; verify with a third eps (0.0003).

## Convergence at the production configuration, with correct CPU binding (2026-11-11)
- psi4_22 (r=25 shell, T=110 incl. merger t~84 and ringdown), reference cap 14 (p14u): cap 13 max|dphase| 7.3e-3 rad, final -1.8e-3 rad, max|dA/A| 6.3e-3 (rms 2.1e-3); cap 12: 0.55 rad / 0.68. Cap 13 is converged to ~0.007 rad (<< the 0.03 rad needed for SNR~30), error ratio 13 vs 12 about 1/75 (>6 orders of 2: dominated by the cap-12 AMR failure, not a clean power law). r=18: cap 13 0.046 rad / 0.16 (extraction-radius effects at small r; use r>=50 shells for the product).
- Cost per run with --bind-to none (single GPU, concurrent runs on the other GPUs): cap 13 T=110 live-remeshing 851 s wall (ETS 784 s); cap 14 1958 s; cap 15 >3 h (still running when the allocation ends). FD members replayed on the recorded mesh: ~14 min at cap 13 (4-6 concurrent on a 4-GPU node). => a p=2 Jacobian (4 members + fiducial) is ~15-30 min of one node, and the p=2 Hessian stencil (9 members + fiducial) ~45 min.

## p=2 Hessian of psi4_22 w.r.t. (m1, m2) bare masses, cap 13, eps=0.001 (analyze_hess_p2.py)
Members: fid, +-e on BH1 (existing), +-e on BH2 (b2p/b2m), four mixed (pp, mm, pm, mp); each with its own warm-started TwoPunctures solve; replay of the fiducial mesh record.
- Symmetry check (q=1 aligned: pi rotation exchanges the BHs, psi4_22 invariant): J1 = J2 to 1e-7 (t<30) ... 1e-10 (t>70); H11 = H22 to 4e-3 (junk phase, signal ~1e-7) and 1e-5 or better for t>50. This validates the whole chain (ID per member, replayed mesh, extraction) at the level the Hessian needs.
- Relative magnitudes |H|/|z| (t=110): H11=H22=25.9, H12=35.6, |J|/|z|=8.0, i.e. strongly nonlinear in the late phase: second-order terms matter for the posterior shape at the merger/ringdown. By the q=1 symmetry only (J1, H11, H12) are independent, so 4 members suffice at q=1; the general q needs all 9.
- Not yet done: eps-plateau of the mixed derivative (needs a second eps), comparison against a third eps for H11 (0.0003), and conversion to physical parameters (q, total mass, spins).

### Hessian eps-plateau (cap 13, p=2, analyze_hess_plateau.py; eps 3e-4 / 1e-3 / 3e-3, 15 members + fiducial)
Relative mismatch |X(1e-3)-X(3e-4)|/|X(3e-4)| for first derivative J1, H11, mixed H12 (psi4_22 r=25 shell):
- J1: 1e-6 .. 3e-6 for t<42, 2e-5 at t=52, 2e-4 at t=73, <=6e-3 in the ringdown. First derivatives are fine at any of these eps.
- H11: 6e-1 (t=10, signal ~1e-7: junk phase, irrelevant) -> 1.6e-1 (21) -> 6e-2 (31) -> 4e-2 (37) -> 2.5e-2 (42) -> 1.1e-2 (52) -> 2.8e-2 (73) -> 0.6 (99, ringdown) ; H12: 7e-2 .. 5e-3 for t<73, 0.5 (99).
- 3e-3 vs 1e-3 for H11: 6e-4 (t=42), 2e-2 (52), 7e-2 (73): truncation O(eps^2) visible at the larger step. 3e-4 vs 1e-3: noise-dominated: second differences amplify a solver noise floor delta/eps^2 with delta ~ 1e-9 |z| in the inspiral, growing ~1000x in the ringdown (phase sensitivity). 
- Verdict: with FD at cap 13 the Hessian is good to ~1-3% for the inspiral+merger (t<~75) at eps ~1e-3 (optimal between truncation 1e-3-scale and noise), degrades to tens of percent in the ringdown. Gauss-Newton/Fisher-metric derivatives (dG = dJ^T J + J^T dJ) need H only contracted with J, so these errors enter the sampler as a deterministic, mildly noisy proposal metric (correctness preserved by the MH correction; efficiency affected). If a clean ringdown Hessian is needed: (a) richer stencils (4th-order, eps 3e-3, error ~eps^4) -- 2x members, or (b) a tangent-linear second-order code (exact, no noise floor), which is the real argument for investing in the TL model (Tier 3.1) rather than a full adjoint.

## Cap-15 reference (2026-11-11, p15v, T=110, 8841 s wall on one GPU, CPU binding fixed)
psi4_22 (r=25 shell, |z|>10% of max): cap 14 vs cap 15: max|dphase| 1e-4 rad, max|dz|/max|z| 1e-4 -> cap 14 is converged to ~1e-4 rad; cap 13 vs cap 15: 7.2e-3 rad / 4.8e-3 (same as vs cap 14), cap 12: 0.43 rad. The phase/amplitude error therefore drops ~60x from cap 13 to 14 and ~70x from 14 to 15 (cap 12 -> 13 ~60x): clean convergence once cap >= 13 (error ~ 10^(-1.8 per level)). Costs (ETS, 1 GPU, T=110): cap 13 784 s, cap 14 1863 s, cap 15 8606 s; with CFL 0.5/rm50 cap 13 is ~415 s at 2e-3 rad.
Decision: production runs at cap 13 (error ~0.007 rad, i.e. ~4x below the 0.03 rad dephasing allowed at SNR~30); use cap 14 as the verification run for any posterior mode that matters (error 1e-4 rad).


## !! CORRECTION 2026-11-11 08:45 -- see STATE_2026-11-11.md top: the T=110 comparisons above test the junk-radiation transient (shell r=100, retarded time <= 10 M; BHs have not merged: q=1 r=3.6 at T=110), not inspiral phase accuracy. Redo with T=260 and retarded-time windows is running (p1{2,3,4}L_T260).

## REAL inspiral convergence, q=1 (T=260, merger at u~145 at r=50; r_sep=0 after ~t=130-140 in the BH tracker) -- 2026-11-11 10:55
Comparison on retarded time u=t-r_ex, window u>=60 (junk gone), psi4_22, new binary, CFL 0.25/remesh 50, both BHs at the stated level (analyze_inspiral.py):
- cap 12: unphysical (BH separation -> 0 before t=100; psi4 off by factors of ~1).
- cap 13 vs cap 14 (r=50 shell): dphase = -0.60 rad at u=100, **-6.5 rad at the merger peak (u=140)**, -7.2 rad after merger; amplitude errors 85% near u=100 -> cap 13 is NOT adequate: its inspiral is much too fast (numerical dissipation / puncture drift). (r=100 shell gives the same picture.)
- Therefore the earlier "cap 13: 0.007 rad" statements measured the junk transient only (see CORRECTION in STATE_2026-11-11.md). The production par (q1: levels 15/15) was presumably chosen for this reason. Whether cap 14 is itself converged is unknown: cap 15 and 16 at T=260 are needed (queued: allocation long20h; each cap-15 run is multi-hour on one GPU).
- p13L crashed at t=238 (post-merger) the moment the mesh exceeded 65535 elements (64730 -> 65717) with "!! invalid argument" (CUDA error printed by check_last_error, device.h:214); the earlier q=4 (15,12) crash at ~70k elements is the same signature. p14L's mesh is at 62.4k elements and growing at t=254. => likely a hard limit at 65535 (a launch dimension in y/z, or a 16-bit index) hit by post-merger meshes; must be found and fixed before any production run through ringdown. Not yet located (grid dims of unzip/zip use y = number of blocks m_zip_grid[1], m2b etc.).
