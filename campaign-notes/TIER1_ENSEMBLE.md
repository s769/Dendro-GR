# Tier 1.3 — Ensemble K on PT_ASYNC (first cut)

## API

`bssnrhs3_pt_async(..., DEVICE_UINT K = 1)`

- Shared block list (`grid.x = nblocks`)
- Ensemble members on `grid.y = K`
- Unzipped layout: leading member stride `BSSN_NUM_VARS * szpdof_uz` per member
- `K = 1` is the production single-member path

## Bench (`rhsgpu_test`)

```bash
DENDRO_TIME_PT_ASYNC=1 DENDRO_ENSEMBLE_K=8 ./rhsgpu_test <nblocks> <nReps> 0
```

## Measured (c124-009, GB200, nReps=20, K=8)

| nblocks | single (s) | serial-K (s) | fused-K (s) | fused/single | serial/fused |
|--------:|----------:|-------------:|------------:|-------------:|-------------:|
| 64 | 0.0032 | 0.0276 | 0.0183 | **5.7×** | 1.51× |
| 256 | 0.0095 | 0.0675 | 0.0559 | 5.9× | 1.21× |
| 2048 | 0.0615 | 0.3990 | 0.3798 | 6.2× | 1.05× |

Repeat (c104-018): 64→5.6× single / 1.55× vs serial; 256→1.20× vs serial; `mismatched_slots=0`. 4-GPU smoke ETS≈4.0s.

Handoff target at 64 blocks: K=8 well under 8× single → **met** (5.7×). Gain vs serial launches is mostly launch amortization and shrinks as nblocks grows (compute-bound). Second version (K members/CTA sharing block geometry/smem) still open for better small-nblocks efficiency.

## Bit-id across slots (Tier 1.4 kernel slice)

At 64 blocks, K=8, identical inputs in every slot: `mismatched_slots=0` vs K=1 reference. Full-solver 1.4 (atomics / reductions) still open.

Raw: `tier0/ensemble_k_c124-009.txt`
