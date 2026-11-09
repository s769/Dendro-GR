# Tier 2 — next levers (after 2.4/2.5)

## Measured baseline (honest multi-GPU)

| case | 1 GPU ETS | 4 GPU ETS | eff@4 |
|------|----------:|----------:|------:|
| q4 | 2.275s | 1.453s | **0.39** |
| q8 | 4.709s | 3.034s | **0.39** |

nsys q8/4GPU kernel share: PT_ASYNC RHS 40%, unzip 28%, stock-boundary RHS 19%, ghost pack/unpack ~10%, **zip 1.4%**.
Memops: H2D 60% / D2H 34% (host-staged ghosts).

## 2.2 Zip-fuse — low priority

Zip is only **1.4%** of GPU kernel time. Fusing zip into RHS will not move ETS until unzip/ghosts are fixed. Park behind unzip work.

## 2.3 Launch hygiene — compact boundary (prototype)

`bssnrhs3_stock_boundary` launched `eval_rhs3<bflag_sel=2>` over **all** blocks (interior CTAs early-return). Prototype: host-filter `m_bflag!=0`, launch only those (`DENDRO_COMPACT_BDY=1`).

| rank local blocks | boundary | fraction |
|------------------:|---------:|---------:|
| 1209 / 1213 | 37 | 3.1% |
| 1168 / 1080 | 111 | ~10% |

A/B q8 `np=4` TIME_END=0.5:

| mode | ETS |
|------|----:|
| baseline (`COMPACT_BDY=0`) | **2.988s** |
| compact (`COMPACT_BDY=1`) | 3.085s |

**Verdict:** compaction is correct but **ETS-neutral** with host D2H rebuild. Stock’s 19% nsys share is real work on the ~3–10% boundary blocks (plus launch tax), not empty CTAs alone. Next: persist compact list in `MeshGPU` on remesh (no per-RHS D2H), then re-nsys stock kernel time.

Default remains **off** (`DENDRO_COMPACT_BDY=1` to enable).

## CUDA-aware MPI — **done (keep ON)**

`MeshGPU::read_from_ghost_cg_*` already has `#ifdef MPIX_CUDA_AWARE_SUPPORT`. Enabling `-DMPIX_CUDA_AWARE_SUPPORT=ON` (now default in `dendro_cmake`):

| q8 np=4 | ETS | eff@4 (1→4) |
|---------|----:|------------:|
| host-staged | 3.03s | 0.39 |
| **CUDA-aware** | **2.13s** | **0.56** |

~30% ETS win @4GPU; 1GPU unchanged (~4.74s).

## Compact boundary — device-side (measured)

`DENDRO_COMPACT_BDY=1` now uses on-device atomic compact (`compact_bdy.cu`), no host D2H.

| mode (q8/4GPU, CUDA-aware) | ETS |
|----------------------------|----:|
| baseline | 2.121s |
| device compact | 2.118s |

**ETS-neutral.** Stock’s ~19% kernel share is real boundary-block algebra (3–10% of blocks), not empty-CTA tax. Leave compact off by default; optional for launch-count hygiene only.

## Higher-impact order (remaining)

1. ~~CUDA-aware ghosts~~ **done**
2. ~~compact bdy for ETS~~ **no-go** (neutral)
3. **Unzip cost** (28% kernel) — next real lever; `__unzip_cg1__2` hanging-interp path.
4. Zip-fuse (2.2) and 19³ (2.6) after unzip.
5. Tier 3.1 TL — dual-res FD failed; separate track.
6. Re-nsys under CUDA-aware — **done** (P2P 59% of memops).

## Ghost path code map (dendroLib)

`MeshGPU::read_from_ghost_cg_begin/end` in `_deps/dendrolib-src/GPU/include/mesh_gpu.cuh` (~L2649+):
- `__write_sb_cg__` packs device send buffer
- host `AsyncExchangeContex` + MPI on **host** buffers → implies D2H before Isend / H2D after Irecv (matches nsys memop mix)
- `__read_rb_cg__` unpacks into device vec

Next experiment: confirm with nsys MPI+CUDA trace whether send/recv touch host-pinned buffers; then trial CUDA-aware MPI (`MPI_Isend` on device pointers) if OpenMPI build supports it on Horizon.

Artifacts: `tier0/tier23_compact_ab.log`, `tier0/tier25_scale_q{4,8}.log`, `TIER2_PROFILE.md`.
