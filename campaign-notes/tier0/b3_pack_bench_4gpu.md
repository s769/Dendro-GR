# B3 ghost pack A/B — 4 GPU (q8 meshprobe)

Source: `b3_pack_bench_4gpu.log` on c126-026, 2026-11-09.  
Harness: `DENDRO_B3_PACK_BENCH=K` one-shot in `Ctx::unzip` (same zip buffer ×K; msg-size test).

## Results (npes=4, CUDA-aware)

| K | serial (s) | packed (s) | speedup |
|--:|-----------:|-----------:|--------:|
| 2 | 0.00142 | 0.00169 | **0.84×** |
| 4 | 0.00255 | 0.00425 | **0.60×** |
| 8 | 0.00468 | 0.00887 | **0.53×** |

## Read

- Packing **hurts** on q8 (large block count / fat messages). Matches `B3_PACK_PLAN`: pack wins at **fine LTS** (few blocks), not mid-mesh.
- RHS ensK already pays at 32bl (2.89×); ghost pack needs a fine-level / few-block mesh to re-measure.
- API ready: `read_from_ghost_cg_ensK_{begin,end}` in dendrolib. Ctx production wire still pending (bench is one-shot only).

## Note

ETS in these runs is **polluted** by the 20-rep bench inside first unzip — ignore ETS here.

## q1 smoke_cons (4 GPU, same day)

Source: `b3_pack_q1_4gpu.log` on c126-026.

| K | serial (s) | packed (s) | speedup |
|--:|-----------:|-----------:|--------:|
| 2 | 0.00070 | 0.00082 | **0.86×** |
| 4 | 0.00127 | 0.00193 | **0.66×** |
| 8 | 0.00240 | 0.00447 | **0.54×** |

Same NO-GO as q8 — q1 still ~1k active elements/rank, not fine-LTS few-block. Park production pack wire until a true few-block LTS mesh exists; RHS ensK remains the bankable fine-nblocks win.
