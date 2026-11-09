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
