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

## Batch 2: production-like (q1.par.toml adaptivity, RESTORE off, caps 11/12/13 and cap-12 tol 1e-4), T=110, radii 50..100 -- IN PROGRESS
(run_resstudy2.sh; sample interval 0.5208 M for every cap)
