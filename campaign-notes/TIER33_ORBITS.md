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
