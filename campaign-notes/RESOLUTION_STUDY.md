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
