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

## Next coding

1. Add explicit `BSSN_TARGET_ORBITS` (or document mapping from sep/p) in par + resolve to BH coords via PN/2PN guess or table.
2. Short q8 cost sweep: 2–3 orbit targets × frozen-mesh ETS (1 GPU then 4 GPU).
3. Script: `run_tier33_orbits_q8.sh` (GPU) once the knob resolves.
