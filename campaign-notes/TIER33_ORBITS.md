# Tier 3.3 — Hybridization length (orbits before merger)

## Goal

Expose “number of orbits before merger” as a controlled ID parameter; measure cost vs orbits at q=8.

## Existing hooks (no new physics yet)

| knob | where | role |
|------|-------|------|
| `TPID_PAR_B` | par / `TPID::par_b` | puncture half-separation scale |
| `BSSN_BH{1,2}.{X,V_*}` | par | initial locations / momenta (sets orbit count for fixed q) |
| `BSSN_BH_MERGE_TIME` | used in `dataUtils.cpp` ORBIT refine | disables ORBIT refine after merge+ringdown |
| ORBIT refine | `dataUtils.cpp` | Nyquist-based AMR along inspiral |

## Scout (c106, job 86880) — sep scale with **reused** q8 TPID sol

Script: `run_tier33_sep_cost_q8.sh`. Scales `TPID_PAR_B` + BH `X` by fac; **does not re-solve TPID** (sol is for fac=1). Short `TIME_END=0.25`, 1 GPU.

| fac | wall (s) | ETS (s) | lmax |
|----:|---------:|--------:|-----:|
| 1.00 | 33 | 2.422 | 13 |
| 1.25 | 30 | 0.368 | 10 |

Caveat: fac≠1 with frozen sol is **not** a faithful hybridization-length sample (ID inconsistent → shallower mesh). Treat as cost-vs-refinement sensitivity only.

## Next coding

1. Re-TPID (or analytic ID) per separation / target-orbit; then re-run cost table.
2. Add explicit `BSSN_TARGET_ORBITS` → BH coords via PN table.
3. q8 4-GPU cost sweep once ID is consistent.

## TARGET_ORBITS scaffold (2026-11-09)

Knob: `BSSN_TARGET_ORBITS` (TOML float / JSON number). Leading-order PN:

`a/M = (32 π ν N)^{2/5}`, `par_b = a/2`, BH X = ±`par_b` about `TPID_CENTER_OFFSET`.

Wired in `parameters.cpp` (TOML) and `grUtils.cpp` (JSON). Smoke (`run_tier33_target_orbits_smoke.sh`):

| N | expect a/M | expect par_b | printed |
|--:|-----------:|-------------:|---------|
| 5.0 | 6.91328 | 3.33496 | **match** (`T33_TARGET_ORBITS_SMOKE_PASS`) |

**Caveat:** smoke reused existing TPID sol — ID inconsistent until re-TPID. Next: re-TPID per N, then q8 cost table.

## re-TPID table (q1 equal-mass, 2026-11-09)

| N | a/M | par_b | wall (s) | sol |
|--:|----:|------:|---------:|-----|
| 3.0 | 5.63566 | 2.71864 | 478 | `tpid_tier33_orbits/N3/` |
| 5.0 | 6.91328 | 3.33496 | 362 | `tpid_tier33_orbits/N5/` |
| 8.0 | 8.34319 | 4.02475 | 427 | `tpid_tier33_orbits/N8/` |

Script: `run_tier33_retpid_orbits.sh`. Cost ETS with consistent ID next (`tier33_cost_N*`).

## Cost ETS (consistent ID, TIME_END=0.15, 1 GPU)

Source: `tier33_cost_orbits.log` on c110-028.

| N | ETS (s) | wall (s) | lmax |
|--:|--------:|---------:|-----:|
| 3.0 | 6.784 | 24 | 9 |
| 5.0 | 6.746 | 23 | 9 |
| 8.0 | 7.151 | 26 | 9 |

Short window — lmax identical. Longer TIME_END needed to expose orbit-length cost.
