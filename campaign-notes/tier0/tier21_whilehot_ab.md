# B1 while-hot RIJ — A/B vs PT_ASYNC (lb1)

Source: `tier21_whilehot_ab.log`, `tier21_whilehot_confirm.log` on c126-026, 2026-11-09.

## What changed

`gauge6_point_async_whilehot.cpp` (from CSE via `gen_gauge6_whilehot_rij.py`):

- Single christoffel (CSE).
- gt/chi **seconds** scoped until early Ricci (no restencil).
- RIJ right after Gt2 load; At_rhs reuses stored `DENDRO_RIJ*`.

## Results

| metric | ASYNC | WHILEHOT | delta |
|--------|------:|---------:|------:|
| rhsgpu 256bl | 0.0071 | 0.0068 | −4% |
| rhsgpu 2048bl | 0.0508 | 0.0346 | **−32%** |
| max_rel | — | 6.2e-16 | ulp |

### q8 ETS (3 passes)

| pass | ASYNC | WHILEHOT | delta |
|-----:|------:|---------:|------:|
| 1 | 2.0640 | 2.0373 | −1.3% |
| 2 | 2.0521 | 2.0423 | −0.5% |
| 3 | 2.0463 | 2.0403 | −0.3% |
| **mean** | **2.054** | **2.040** | **−0.7%** |

## Verdict

**Small keep** — `DENDRO_RHS_PT_ASYNC_WHILEHOT=1`. First B1 algebra change that moves ETS the right way (CSE was flat/NO-GO). Not enough alone to replace lb1 prod default. Env opt-in for now.

## Fine nblocks + q1 (c105-005, 2026-11-09)

Source: `whilehot_fine_ensk.log`.

### ensK under ASYNC @ fine nblocks (whilehot binary same RHS)

| nblocks | K | ensK/serialK |
|--------:|--:|-------------:|
| 32 | 4 | 2.72× |
| 32 | 8 | 2.84× |
| 64 | 4 | 1.57× |
| 64 | 8 | 1.59× |

whilehot single-RHS bit-id: max_rel 6.2e-16 (same as before).

### q1 smoke_cons ETS

| mode | ETS (s) | delta |
|------|--------:|------:|
| ASYNC | 12.4584 | — |
| WHILEHOT | 12.4295 | **−0.23%** |

Consistent small keep; fine-nblocks ensK headroom unchanged vs lb1 baseline.
