# Dendro-GR GPU optimization brief

Portable handoff for a new machine/agent. Repo clone: local only — **no upstream push access**.

## Goal

Max single-GPU performance of Dendro-GR BSSN evolution. Explicit timestepping; high mass-ratio runs need many steps on fine AMR, so outer-loop optimization is impractical until the per-step GPU cost drops.

## Repo / build

- Upstream: https://github.com/paralab/Dendro-GR (no push access)
- GPU driver: `bssnSolverCUDA` / `BSSN_GR/cuda/`
- dendrolib (Dendro-5.01) comes from CMake `FetchContent` or `-DDENDRO_dendrolib_DIR=...`
- CUDA-only. Use **A100** for baseline work. AMD MI250X requires a HIP port first.

## Production hot path

```
zip CG (device)
  → read_from_ghost + unzip_cg   [MPI; CUDA-aware if MPIX_CUDA_AWARE_SUPPORT]
  → eval_rhs3 <<<nblocks, (7,7,7), ~52KB>>>
  → zip_cg
  → ETS RK update + enforce_evar_cons
```

| Piece | Location |
|-------|----------|
| RHS orchestration | `BSSN_GR/cuda/src/bssnCtxGPU.cu` (`BSSNCtxGPU::rhs`) |
| Kernel | `BSSN_GR/cuda/include/bssn_kernels.cuh` (`eval_rhs3` / `bssnrhs3`) |
| Generated body | `BSSN_GR/cuda/scripts/bssneqs_eta_const_standard_gauge6.cpp` |
| Codegen | `CodeGen/` (SymPy CSE, staging, DAG variants) |

Kernel shape: one CUDA block per Dendro unzip block; interior `7³` threads; pad `pw=3` → `13³`; shared `su`/`Du`/`DDu`; `__launch_bounds__(343, 3)`.

## Bottleneck (still true)

- **Register pressure** (~56 regs) caps occupancy at ~3 blk/SM once smem is PreferShared
- PreferL1 on GB200 carves shared too small → often **1 blk/SM** (avoid for float-tile paths)
- gauge6: 24 loads, ~210 derivs, many `__syncthreads`
- No tensor cores; `DEVICE_RHS_NSTREAMS=1`

## Horizon GB200 path (local)

**Default:** `DENDRO_RHS_PT_ASYNC=1` — FP64 pointwise fused FD (~1.7× stock, bit-id interior).  
**Hedge:** `DENDRO_RHS_PT_F32SU=1` — float tile (~2.1×); not production until high-q A3.  
**Next:** FP64 staged algebra, zip fusion, 19³ blocks, ENUTS-on-GPU, 8th-order study.  
**Not:** tensor cores for RHS (K=7/13). See `HANDOFF_NEXT.md`, `GB200_RESULTS.md`.

## 1-GPU time split (in-repo LS6 `q1-ss` log)

- RHS ~71%
- unzip ~21%
- zip ~2%

Comm matters but is not the first 2×. At higher GPU counts, “other” evolve overhead grows; unzip fraction often falls.

## Priority list

1. Nsight on `eval_rhs3` (spill, barrier, occupancy, memory)
2. Codegen fission by equation groups (cut live ranges)
3. Fewer barriers in deriv pencils
4. Selective mixed precision (FP32 scratch/algebra, FP64 state); validate constraints + ψ₄
5. Sync removal / streams / CUDA-aware MPI check
6. Comm overlap polish (not NCCL-first)
7. Tensor cores — skip unless a real GEMM formulation appears

## Mixed precision ladder

1. FP64 state + FP32 deriv scratch  
2. FP32 RHS algebra, FP64 RK/state  
3. FP32 evolve + periodic FP64 constraint clean  
4. Full FP32 (acceptance only)

## How to load this on another system

```bash
# copy clone + notes (example)
rsync -a Dendro-GR/ other:/path/Dendro-GR/
rsync -a Dendro-GR-notes/ other:/path/Dendro-GR-notes/
```

Then in a new Cursor agent chat:

> Read `/path/Dendro-GR-notes/GPU_OPT_BRIEF.md` and follow `.cursor/rules/gpu-opt-context.mdc` before proposing GPU changes.

Or paste a Cursor **User Rule** that says: when working on Dendro-GR GPU, always read that brief first.

Local Cursor rule (this clone): `Dendro-GR/.cursor/rules/gpu-opt-context.mdc`  
(Always-apply in this project; not pushed upstream.)

## Perlmutter paths (NERSC)

- Host: `perlmutter.nersc.gov` (user `srvenkat`)
- Home: `/global/homes/s/srvenkat`
- Scratch: `/pscratch/sd/s/srvenkat`
- This brief on Perlmutter: `/global/homes/s/srvenkat/Dendro-GR-notes/GPU_OPT_BRIEF.md`
- Local Cursor rule (if cloned): `/global/homes/s/srvenkat/Dendro-GR/.cursor/rules/gpu-opt-context.mdc`

On Perlmutter agent chat: read the brief first before proposing GPU changes.

