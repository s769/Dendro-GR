# GPU opt experiments log (Perlmutter A100-80GB)

**GB200 transfer:** see [`GB200_HANDOFF.md`](GB200_HANDOFF.md) and tarball  
`/pscratch/sd/s/srvenkat/Dendro-GR-gpuopt/transfer/dendro-gr-gpuopt-pt.tgz`  
(repack: `../scripts/pack_gb200_transfer.sh`). Solver flag: `DENDRO_RHS_PT=1`.

Baseline: `rhsgpu_test 2048 20 0` → **gpu_time = 0.1710 s**  
`eval_rhs3` + `launch_bounds(343,3)`: 56 regs, spill st/ld 9560/17128 B

## Results

| Experiment | gpu_time | vs baseline | Notes |
|------------|----------|-------------|-------|
| baseline lb(343,3) PreferShared | **0.1710** | — | keep as reference |
| **point-wise fused + PreferL1** (`DENDRO_TIME_PT`) | **0.1371** | **+20%** | bit-identical; items 1–2 |
| point-wise PreferShared | 0.1660 | +3% | PreferL1 wins once smem is only `su` |
| point-wise lb(343,4) PreferL1 | 0.1818 | −6% | more spills (10748/18924) |
| point-wise + staged seconds (item 3) | 0.1571 | +8% | bit-identical; 11-var reload loses to unstaged |
| point-wise + cp.async issue+wait (`DENDRO_TIME_PT_ASYNC`) | 0.1577 | +8% | correct; no overlap yet → slower than sync ld |
| point-wise + KO in FP32 (`DENDRO_TIME_PT_KOF32`) | 0.1356 | +20% | ~1e-4 rel err on α; tiny extra win |
| F32 Du/DDu PreferShared (`DENDRO_TIME_F32`) | 0.1419 | +17% | rel err ~1e-3; evolve C_HAM×4 |
| **F32 + block-center cast** (`DENDRO_TIME_F32C`) | 0.1438 | +16% | rel err ~100–500× better than raw F32 |
| lb(343,1) fused | 0.1844 | −8% | fewer spills, lower occupancy loses |
| flat grad_buf | 0.1711 | ~0 | no effect |
| elide 68 syncs | 0.1711 | ~0 | not barrier-bound |
| PreferL1 on baseline | 0.2440 | −43% | hurts pencil smem |
| rhs4 / WS / HBM deriv workspace | slower | — | adds traffic to slow tier |

### Point-wise path (items 1–2)

- Device: `derivs_point.cuh` (`__pt_fused_{x,y,z}3`, `__pt_dxy/xz/yz`)
- Codegen: `scripts/gen_gauge6_point.py` → `gauge6_point.cpp`
- Kernel: `eval_rhs3_pt` / `bssnrhs3_pt` — **su only** (17.6 KB), no Du/DDu block passes
- Must `__syncthreads()` after each var’s point stencils before the next `__ld_blk_var1__` (β/gt/At batches race otherwise)
- ptxas: spill st/ld **7040/12748** (vs baseline 9560/17128)
- nsys @512×5: `eval_rhs3` 2.43 ms/call vs `eval_rhs3_pt` **1.81 ms/call** (1.34×)

### Mixed precision (item 6)

- Raw float field cast breaks near punctures (FD error ~ε₃₂·|u|/h)
- **Block-center cast** `(float)(u − u_c)` before smem stencils: same speed as F32, much smaller RHS error
- **KO-only FP32** on the point path: ~1% over PT, small RHS drift — viable for dissipation

### Block size (item 5)

- `rhsgpu_test` hard-codes **13³** (one octant + 3-pt pad). Interior fraction 7³/13³ ≈ 16%.
- Production histogram still needs a real q≥10 mesh dump from `bssnSolverCUDA` (not in this microbench).

## Tensor cores

- Dense stencil-as-`cublasDgemm` + `CUBLAS_TENSOR_OP_MATH`: **1.69×** vs naive pencil loop (N=13, large batch); FP64 OK.
- Dense GEMM does N² work vs 6-pt sparse stencil — need **custom MMA / img2col-sparse**, not raw DGEMM, to beat Dendro shared-mem pencils.
- Skip FP64 TC on Blackwell for this kernel (vector FP64 is the right target).

## Next actions

1. **True cp.async double-buffer**: prefetch var v+1 into `su_B` while computing v on `su_A` (issue+wait alone is a loss)
2. Evolve with PT (default candidate) / `DENDRO_TIME_F32C` / PT_KOF32 and watch C_HAM
3. Block-size histogram from a production mesh; prototype 19³ on Hopper/Blackwell smem
4. Full-step `nsys` (unzip/zip + ghosts) — kernel-only wins may shrink on real steps
5. Revisit staging only if combined with keeping multiple `su` buffers resident (needs ≥Hopper smem)
