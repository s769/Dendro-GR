# B3 fine-nblocks ensK (RHS fused; ghosts not yet packed)

Source: `b3_fine_ensk.log` on c126-026 (GB200), lb1 prod binary, 2026-11-09.

ensK / serialK speedup (v1 `grid.y=K`):

| nblocks | K=2 | K=4 | K=8 |
|--------:|----:|----:|----:|
| 32 | 1.93× | 2.63× | **2.89×** |
| 64 | 1.48× | 1.49× | 1.69× |
| 128 | 1.06× | 1.09× | 1.20× |

## Read

- Fine LTS (32 bl) is where ensemble RHS fusion pays — approaching the 2–3× bankable target from `BATCHING.md`.
- 128 bl already near flat; matches prior 256/2048 baseline.
- **Still missing for B3:** packed K ghost exchange (1-rank idev cannot measure MPI pack). Next: `__write_sb_cg_ensK` in dendrolib + 2–4 GPU measure.

## Status

RHS half of B3 **measured**. Ghost-pack half **scaffolded** (`B3_PACK_PLAN.md`); needs multi-GPU.
