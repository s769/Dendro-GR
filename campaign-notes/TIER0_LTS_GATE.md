# Tier 0 — GPU LTS gate (go / no-go)

**Verdict: NO-GO on GPU ENUTS as the high-q critical-path lever for this campaign** (unless `BSSN_LTS_TS_OFFSET` is reduced — a separate physics decision).

Re-cost the SVN forward solve around **hybridization length + resolution ladder + frozen-mesh ensemble (Tier 1)**. Keep the ENUTS audit and small-nblocks baseline as inputs to Tier 1.3.

Measured on **c101-014** (job 86116, 2026-11-07): see §0.2 / §0.3 below.

---

## 0.1 ENUTS × `BSSNCtxGPU` (complete)

Full writeup: [`ENUTS_GPU_AUDIT.md`](ENUTS_GPU_AUDIT.md).

Short version:

| ENUTS need | GPU status |
|------------|------------|
| `rhs_blk` / `pre_stage_blk` / `post_*_blk` | **Missing** (CPU has them) |
| `compute_lts_ts_offset` | **Missing** |
| `getBlkTimestepFac` | **Implemented** (matches CPU; was declared-only) |
| Block-async stage vectors + `ENUTSOp` time corrections | **Host-only** |
| Partial / active-set RHS launch | **Missing** (GPU RHS = all blocks) |
| Driver | `gr_cuda.cu` is **ETS-only** (same as CPU production) |

Effort to a *useful* GPU LTS: hybrid host-ENUTS + device batched RHS is **weeks**; device-resident ENUTS is **months**. Not the issue — payoff is.

---

## 0.2 Payoff: block-level work ratio

Formula (offset = 0):  
`R = Σ_l N_l 2^l / (N_tot 2^{L_max})`  
Ceiling on GTS/LTS step-work ≈ `1/R`.

### Production level span (from `pars/q{4,8}.par.toml`)

| q | MAXDEPTH | BH1_MAX_LEV (small) | BH2_MAX_LEV (large) | Δℓ |
|---|--------:|--------------------:|--------------------:|----:|
| 4 | 16 | 16 | 14 | 2 |
| 8 | 17 | 17 | 14 | 3 |

### Default `BSSN_LTS_TS_OFFSET = 4` (parameters.cpp)

`getBlkTimestepFac` returns **1** whenever `(L_max − blev) ≤ 4`. So for production q4/q8, **every block from the large-BH tip through the small-BH tip shares the global fine dt**. LTS only coarsens *below* that band.

### Measured init-mesh histograms (HAD ID, probe depth; `DENDRO_DUMP_BLK_LEV=1`)

| q | N_tot | ℓ span | offset=0 ceiling | **offset=4 (default)** |
|---|------:|--------|-----------------:|-----------------------:|
| 4 | 4390 | 3–12 | **10.7×** | **1.69×** |
| 8 | 4670 | 3–13 | **14.3×** | **1.95×** |

Raw (offset=0) is a ~15×-class win; **shipped offset collapses it below 2×**. Gate stays NO-GO until offset policy changes.

Artifacts: `tier0/mesh_q{4,8}.log`, `tier0/mesh_hist_summary.txt`.

---

## 0.3 Small-nblocks regime (baseline for Tier 1.3)

LTS active sets at the finest levels are **tens–low hundreds** of blocks. That is exactly where the current GPU RHS is launch / latency bound.

Measured PT_ASYNC wall (`rhsgpu_test N 20 0`, c101-014):

| nblocks | time (s) | vs 2048 / (N/2048) |
|--------:|---------:|-------------------:|
| 32 | 0.0029 | ~3.3× slower than linear |
| 64 | 0.0031 | ~3.1× |
| 128 | 0.0048 | ~2.4× |
| 256 | 0.0092 | ~1.8× |
| 512 | 0.0181 | ~1.4× |
| 2048 | 0.0638 | 1.0× |

nsys: at 32 blocks, `eval_rhs3_pt_async` avg ~134 µs vs stock face kernel ~350 µs; `cudaLaunchKernel` still a large API share at small N. Finest LTS active sets live here → **launch-bound**.

**Implication for Tier 1.3:** **K members per CTA** (shared geometry) amortizes this; “K CTAs” alone does not.

---

## Gate decision

| Option | Decision | Why |
|--------|----------|-----|
| Full device ENUTS | **NO-GO** | Months of work; offset kills payoff; small-nblocks tax |
| Hybrid host-ENUTS + GPU subset RHS | **NO-GO for this campaign** | Ceiling ~1.6× under default offset; engineering cost ≫ gain |
| Reduce `BSSN_LTS_TS_OFFSET` toward 0 and revisit | **Parked physics question** | Might restore ~5× ceiling; needs stability/eta study — not a free lunch |
| Proceed without GPU LTS | **GO** | Tier 1 frozen-mesh ensemble + Tier 3.2/3.3 hybridization/resolution |

**Inversion plan without GPU LTS:** single-sim time-to-solution at q≈8–16 rests on (1) FP64 structural opts (Tier 2), (2) node-efficient strong scaling (2.5), (3) fewer orbits / hybridization knob (3.3), (4) resolution ladder (3.2), (5) ensemble throughput via K-in-CTA (1.3).

---

## Code / doc changes this turn

- `ENUTS_GPU_AUDIT.md` — gap list + effort
- `bssnCtxGPU.cu` — `DENDRO_DUMP_BLK_LEV` histogram; `getBlkTimestepFac` body
- `run_tier0.sh` — rebuild + 0.2 mesh probes + 0.3 nsys/ncu when nodes live
- `q{4,8}.meshprobe.par.toml` — short init-only probes
- Build: login-node BLAS → NVIDIA NVPL (`libblas`/`liblapack` under `.../26.9/compilers/lib`) after flexiblas vanished from login image

## Negatives

- SLURM accounting down → no measured hist / nsys this session (queued).
- Ring model is not a substitute for the dump; used only to show order of magnitude and offset sensitivity.
- GPU init path still always uses WAMR for init converge (ignores `USE_SET_REF_MODE`); probe pars set BH_LOC but f2o + WAMR drive the mesh — fine for a ceiling estimate once runnable.
