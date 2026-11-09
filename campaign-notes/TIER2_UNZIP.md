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

### Batch sync deferral (`DENDRO_UNZIP_BATCH_SYNC`)

Default now: one `device_synchronize` after all `async_k` unzip batches (GPU can run while next batch’s MPI ghost wait runs). `=1` restores old per-batch sync.

| mode | `TIME_END=0.5` ETS | `TIME_END=1.0` ETS |
|------|-------------------:|-------------------:|
| oldsync (`BATCH_SYNC=1`) | 2.151 | 3.966 |
| deferred (`BATCH_SYNC=0`) | 2.146 (−0.2%) | 3.948 (−0.4%) |

**ETS-neutral.** Keep deferred as default (hygiene). Nsys mix unchanged (~40/28/19). Raw: `tier0/tier2_unzip_batchsync_{ab,long}.log`.

## Next levers (ordered)

1. Hang-path / occupancy (ncu on `__unzip_cg1__2` hang branch; tile/smem)
2. Deeper ghost/unzip stream overlap (per-batch CUDA streams, not just deferred sync)
3. Frozen-mesh ensemble: unzip geometry once (Tier 1.3 synergy)
4. Tier 2.1 staged RHS algebra (RHS is still 40%)

### Longer confirm `TIME_END=1.0` (c125-022)

| mode | ETS (s) |
|------|---------|
| baseline | 3.931 |
| active (`SPLIT=1`) | 3.910 (−0.55%) |

Confirms noise-level only. Raw: `tier0/tier2_unzip_active_long.log`.
