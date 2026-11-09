# Tier 2.5 — strong scaling (one GB200 node)

Short evolve `BSSN_RK_TIME_END=0.5`, `DENDRO_RHS_PT_ASYNC=1`, honest multi-GPU (no `CUDA_VISIBLE_DEVICES` pin).
Host: c127-032 (job 86503).

## q=8 (`N≈4678`, ℓ=3–13, GTS/LTS ceiling ≈14.3×)

| np | wall (s) | ETS (s) | speedup | efficiency |
|---:|---------:|--------:|--------:|-----------:|
| 1 | — | 4.709 | 1.00× | 1.00 |
| 2 | 23 | 4.052 | 1.16× | **0.58** |
| 4 | 18 | 3.034 | 1.55× | **0.39** |

## q=4 (`N≈4390`, ℓ=3–12, ceiling ≈10.7×) — host-staged

| np | wall (s) | ETS (s) | speedup | efficiency |
|---:|---------:|--------:|--------:|-----------:|
| 1 | 32 | 2.275 | 1.00× | 1.00 |
| 2 | 20 | 1.944 | 1.17× | **0.59** |
| 4 | 15 | 1.453 | 1.57× | **0.39** |

q4 @4GPU CUDA-aware: **1.194s** (−18% vs 1.453). Raw: `tier0/tier25_q4_cudaaware_np4.log`.

Raw: `tier0/tier25_scale_q{4,8}.log`, `tier0/tier25_q{4,8}_np*.log`.

## q=8 with CUDA-aware MPI (`-DMPIX_CUDA_AWARE_SUPPORT=ON`)

| np | wall (s) | ETS (s) | speedup | efficiency |
|---:|---------:|--------:|--------:|-----------:|
| 1 | 37 | 4.739 | 1.00× | 1.00 |
| 2 | 23 | 3.195 | 1.48× | **0.74** |
| 4 | 17 | 2.129 | 2.23× | **0.56** |

vs host-staged @4GPU: 3.03s → **2.13s (−30%)**. Eff 0.39 → **0.56**.
Raw: `tier0/tier25_scale_q8_cudaaware.log`. Now default in `dendro_cmake`.

## q=8 CUDA-aware — 2 nodes (8 GPU)

Nodes c125-022 + c128-001 (separate idev jobs, hostfile MPI). `TIME_END=0.5`.

| np | ETS (s) | speedup vs np4 | efficiency vs np4 |
|---:|--------:|---------------:|------------------:|
| 4 | 2.145 | 1.00× | 1.00 |
| 8 | **1.676** | **1.28×** | **0.64** |

vs 1GPU CUDA-aware (4.739s): 8GPU → **2.83×**, eff **0.35**. Cross-node helps but communication still dominates beyond one node.
Raw: `tier0/tier25_2node_q8.log`. Script: `run_tier25_2node_q8.sh` (`-x LD_LIBRARY_PATH`, `--host node:4`).

## Notes

- Host-staged ghosts were the main limiter; CUDA-aware is the production build flag.
- Remaining gap: unzip (~28% kernel) + stock-boundary + load imbalance.
- Early “4GPU” Tier 2.4 numbers also suffered `CUDA_VISIBLE_DEVICES=0` pin — prefer CUDA-aware table above.
- Host-staged nsys: RHS 40% / unzip 28% / stock-bdy 19%; H2D+D2H dominated memops (`TIER2_PROFILE.md`).
- Fresh CUDA-aware nsys on c128-001: RHS 40.3% / unzip 27.9% / stock-bdy 18.9% (unchanged mix).
