# Tier 2 — unzip cost (next lever after CUDA-aware)

Under CUDA-aware q8/4GPU nsys, `__unzip_cg1__2` is **~28%** of GPU kernel time (second only to PT_ASYNC RHS at 40%). Zip is 1.4% — ignore for now.

## Where it lives

- Kernel: `_deps/dendrolib-src/GPU/include/mesh_gpu.cuh` `__unzip_cg1__2`
- Launch: `MeshGPU::unzip_cg` → grid over elements × vars
- Env: `DENDRO_UNZIP_SPLIT` — `0` stock, `1` active-only (skip `nmap==0`), `2` hang/nh dual

## Measurements (q8/4GPU, CUDA-aware, PT_ASYNC)

### Short `TIME_END=0.5` on c125-022 (86807)

| mode | ETS (s) | vs baseline |
|------|---------|-------------|
| baseline (`SPLIT=0`) | **2.133** | — |
| active (`SPLIT=1`, skip0) | 2.114 | **−0.9%** (noise) |
| dual hang/nh (`SPLIT=2`) | 2.180 | +2.2% |

Hang fraction of **active** elements: **70–99%**. NH-majority assumption is **wrong**.

Zero-map skip removes ~27% of element CTAs (`skipped0≈600` of `~2260`), but those CTAs were near-empty — not the ETS driver.

### Earlier short runs

- v1 dual same-kernel, no skip (`TIME_END=0.25`, c127): 1.251 → 1.275 (+1.9%)
- v2 dual+skip0: 2.128 → 2.159 (+1.4%)

## Verdict

**List-infra / hang–nh split: no-go for ETS.** Keep `DENDRO_UNZIP_SPLIT` default off; code stays for experiments.

Real unzip cost is **hang prolong + scatter** on the hang-majority path, not empty CTAs or NH occupancy.

### Hang vs nh kernel time (`DENDRO_UNZIP_TIMING=1`, SPLIT=2)

Per-unzip launch on q8/4GPU (~1.3–1.5 ms total):

| path | elements | time | share |
|------|----------|------|-------|
| nh | ~110–140 | ~0.2 ms | ~15% |
| hang | ~1500–1670 | ~1.1–1.5 ms | **~85%** |

Confirms hang path dominates wall inside unzip. Optimizing NH-only kernels will not move ETS. Raw: `tier0/tier2_unzip_hang_time.log`.

### Batch sync deferral (`DENDRO_UNZIP_BATCH_SYNC`)

Default now: one `device_synchronize` after all `async_k` unzip batches (GPU can run while next batch’s MPI ghost wait runs). `=1` restores old per-batch sync.

| mode | `TIME_END=0.5` ETS | `TIME_END=1.0` ETS |
|------|-------------------:|-------------------:|
| oldsync (`BATCH_SYNC=1`) | 2.151 | 3.966 |
| deferred (`BATCH_SYNC=0`) | 2.146 (−0.2%) | 3.948 (−0.4%) |

**ETS-neutral.** Keep deferred as default (hygiene). Nsys mix unchanged (~40/28/19). Raw: `tier0/tier2_unzip_batchsync_{ab,long}.log`.

### `BSSN_ASYNC_COMM_K` sweep (q8/4GPU, deferred sync)

Production pars use **K=1**. Raising K (must divide 24) **hurts** ETS:

| K | ETS (s) |
|--:|--------:|
| 1 | **2.133** |
| 2 | 2.216 |
| 3 | 2.304 |
| 4 | 2.385 |
| 6 | 2.531 |
| 8 | 2.735 |

Longer confirm: K=1 → 3.924 vs K=4 → 4.409 (`TIME_END=1.0`). **Keep K=1.** Raw: `tier0/tier2_async_k_sweep.log`.

## Next levers (ordered)

1. Hang-path / occupancy — **NCU done** (c125-022, `tier0/tier2_ncu_hang.log`, `unzip_hang_occ.ncu-rep`)
2. Frozen-mesh ensemble: unzip geometry once (Tier 1.3 synergy) — only real multi-member win left on unzip
3. Tier 2.1 staged RHS algebra (RHS is still 40%) — after NCU
4. ~~async_k / hang-nh split / batch-sync~~ **closed no-go / neutral**

### NCU on `__unzip_cg1__2` (q8, 1GPU, SPLIT=2, short)

Launch: `(N_ele,1,24)×(7,7,1)`, regs/thread=64, static smem≈9.0 KB.

| metric | value | note |
|--------|------:|------|
| Theoretical occupancy | 50% | block-limit by regs (16 blocks/SM) |
| Achieved occupancy | ~38% | |
| Active / eligible warps/sched | 6.3 / **0.75** | latency-bound |
| Est. speedup: uncoalesced global loads | **~50–64%** | gather via `e2n_cg` |
| Est. speedup: uncoalesced global stores | ~28–36% | scatter to unzip |
| Est. speedup: 49-thread block (partial warps) | ~23% | try `DENDRO_UNZIP_TB88=1` → 8×8 |
| Long-scoreboard stall | ~29% | follows uncoalesced gathers |

**Verdict:** occupancy is secondary; **uncoalesced CG gather/scatter** is the hang-path wall. 8×8 CTA A/B is a cheap check; real win likely needs coalesced staging or frozen-mesh unzip reuse (#5).

### `DENDRO_UNZIP_TB88` A/B (q8/4GPU, `TIME_END=0.5`, c125-022)

| mode | ETS (s) | vs 7×7 |
|------|--------:|-------:|
| tb77 (default) | **2.135** | — |
| tb88 | 2.230 | **+4.4%** |

**NO-GO.** Idle threads in 8×8 outweigh warp-mask theory. Keep 7×7. Patch stays behind env for experiments (`patches/dendrolib/unzip-tb88.patch`). Raw: `tier0/tier2_unzip_tb88_ab.log`.

Hang-path single-kernel micro-opts exhausted for now → next: **frozen-mesh unzip geometry reuse** (queue #5).

### Longer confirm `TIME_END=1.0` (c125-022)

| mode | ETS (s) |
|------|---------|
| baseline | 3.931 |
| active (`SPLIT=1`) | 3.910 (−0.55%) |

Confirms noise-level only. Raw: `tier0/tier2_unzip_active_long.log`.
