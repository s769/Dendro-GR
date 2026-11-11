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
