# Full-step optimization log (2026-11-09 onward)

Goal: speed up the whole step (not only RHS) in the regime the inverse problem needs; every change env-selectable; report bit-id.

## Diagnosis (q8 meshprobe, 4 GPU, CUDA-aware, PT_ASYNC; nsys `nsys_bdy1`, dev0)
GPU busy 64-71% of the 1.84 s step. Kernel time (s/rank): PT RHS 0.55-0.61, unzip 0.41-0.51, stock-boundary RHS 0.36-0.39, ghost pack/unpack 0.14-0.18.
Idle gaps total 0.6 s/rank: **0.42 s = `__write_sb_cg__` -> `__read_rb_cg__` (ghost exchange, 845 us per stage)**; 0.08 s enforce_evar->enforce_evar (122 gaps, 634 us); rest small.
The boundary kernel (37-111 CTAs < 148 SMs) is latency-bound: ~0.63 ms/launch even though it holds 3-10% of blocks.

## Results
| # | change | knob | q8 4GPU ETS (s) | bit-id | verdict |
|---|--------|------|----------------|--------|---------|
| 1 | launch stock-boundary kernel FIRST on a side stream so it overlaps the interior PT kernel (disjoint blocks) | `DENDRO_BDY_STREAM` (**default ON**, `=0` off) | 1.92 -> 1.70 (**-11%**) x2 repeats | GW psi4 output byte-identical A/B | **KEEP, default on** |
| 2 | unzip ghost-independent elements while exchange in flight (`unzip_cg_part`, indep/dep lists in MeshGPU) | `DENDRO_UNZIP_OVERLAP=1` (default off) | 1.70 -> 1.70 (A 1.701/1.739, B 1.699/1.719) | per-element same kernel | **NO-GO** (only 54-68% of elements are ghost-independent at these partition sizes; exchange wait not hidden) |
| 3 | UCX settings: TLS=cuda_ipc,cuda_copy,sm,self ; RNDV_SCHEME=put_zcopy ; RNDV_THRESH=inf | env | 1.74->1.79 ; 1.71->1.88 ; 1.73->9.30 | n/a | **NO-GO** (default best) |

Noise note: Constraints (and ~1e-18 abs in psi4) differ run-to-run even for identical serial runs; unrelated to these changes (A vs A2 vs A3 all differ).
Data: `tier0/ab_bdystream.log`, `tier0/ab_unzipov.log`, `tier0/ab_ucx*.log`.

## Extraction-path cost (found while explaining slow waveform replays)
Waveform/constraint extraction (`BSSNCtxGPU::write_vtu`) runs on the HOST: CPU unzip of all 24 vars, single-core `physical_constraints` over every block (shared global
scratch `BSSN_DERIV_WORKSPACE` => not thread-safe), zip, ghost, psi4 interpolation. Timers (`DENDRO_TIME_EXTRACT=1`), coarse q1, 3.4k elements, 15 extractions in T=4:
total 23.4 s of ETS 34.5 s (**68%**): unzip+D2H 8.4 s (0.56 s/call), constraints loop 11.1 s (0.74 s/call), zip+ghost 0.4, extractConstraints 1.0, psi4 2.5.
Cost grows with mesh size (replays at 40k elements ran ~10 s/step with extraction every 4 steps). Production pars extract every 50-100 steps (q4: 50, q8: 100), so at q8's ~4M steps
extraction is a plausible 10-30% of wall time — NOT measured at production scale; needs one profiled q8 run. Cheapest lever: `BSSN_GW_EXTRACT_FREQ` (parameter).

Experimental `DENDRO_GW_SHELL_ONLY=1` (default OFF): device unzip + per-block D2H of only blocks near the GW spheres, constraints/psi4 on those only. Coarse q1: ETS 34.5 -> 23.0 s (-33%),
unzip+D2H 8.4 -> 0.04 s. **NOT VALIDATED**: psi4 differs from the host-unzip path by up to 1% (l2_m2), 4.5% (l3_m2), already at t=0, independent of margin
(`DENDRO_GW_SHELL_MARGIN` 2 vs 1000 identical diff) => the difference is host-unzip vs device-unzip INPUT, not the block filter. A one-shot comparison (`DENDRO_UNZIP_CMP=1`) was inconclusive
(host unzip array looked stale/zero at the step-0 call, so states weren't synchronized). Constraint norms are invalid in this mode. Do not use for production until the discrepancy is explained.
