# Tier 2.4 — q=8 4-GPU nsys (honest multi-GPU)

c127-032, `mpirun -np 4`, `DENDRO_RHS_PT_ASYNC=1`, `TIME_END=0.5`, no `CUDA_VISIBLE_DEVICES` pin.
ETS ≈ 3.06s (matches Tier 2.5 np=4). Report: `tier0/tier24_q8_4gpu.nsys-rep`.

## CUDA kernel time share

| % | kernel | note |
|--:|--------|------|
| 40.3 | `eval_rhs3_pt_async` | fused interior RHS |
| 27.8 | `__unzip_cg1__2` | unzip to block grid |
| 19.0 | `eval_rhs3` (stock) | domain-boundary / bflag≠0 path |
| 5.1 | `__read_rb_cg__` | ghost recv unpack |
| 4.4 | `__write_sb_cg__` | ghost send pack |
| 1.4 | `__zip_cg__` | zip back to nodal |
| 1.5 | `axpy_cu_2d` | RK update |

## Memcpy (by time)

| % | op |
|--:|----|
| 60.1 | H2D |
| 34.4 | D2H |
| 5.5 | D2D |

## Implications (host-staged)

- RHS is only ~40% of GPU kernel time; unzip (28%) + stock boundary RHS (19%) are first-class.
- H2D/D2H dominate memops → host-staged ghosts explained eff 0.39 @4 GPU.

## CUDA-aware re-nsys (same case, `MPIX_CUDA_AWARE_SUPPORT`)

ETS ≈ 2.25s. Kernel mix unchanged (PT 40% / unzip 28% / stock 19%). Memops:

| % | op |
|--:|----|
| **58.7** | **Peer-to-Peer** |
| 15.7 | D2D |
| 13.0 | H2D |
| 12.5 | D2H |

Host H2D/D2H collapsed; P2P/GDR carries ghosts. Remaining ETS gap is compute (unzip + stock-bdy), not staging.

Raw: `tier0/tier24_q8_4gpu_cudaaware.nsys-rep`.
