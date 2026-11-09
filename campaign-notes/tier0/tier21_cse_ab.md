# B1 CSE christoffel — A/B vs PT_ASYNC (lb1)

Source: `tier21_cse_ab.log`, `cse_maxerr.log` on c126-026, 2026-11-09.

## What changed

`gauge6_point_async_cse.cpp` (via `gen_gauge6_cse_christoffel.py`): identical igt/C1/C2/C3 body was computed 3× (K/Gt/At); now once, decls live across K→At. No restencil. Gtk + RIJ still once each.

## Results

| metric | ASYNC | CSE | delta |
|--------|------:|----:|------:|
| rhsgpu 2048bl | 0.0655 | 0.0524 | **−20%** |
| rhsgpu 256bl | 0.0058 | 0.0056 | −3% |
| rhsgpu 64bl | 0.0040 | 0.0039 | −2.5% |
| q8 meshprobe ETS | 2.0585 | 2.0622 | **+0.2% (flat)** |
| max_rel vs ASYNC | — | 6.2e-16 | ulp-level |
| bit-id | — | no (sched noise) | ok numerically |

## Verdict

**NO-GO for production ETS.** FLOP win shows in large-nblocks microbench; q8 ETS unchanged. Longer C* live ranges likely cancel any spill win (B0 was spill-bound). Keep generator + `DENDRO_RHS_PT_ASYNC_CSE` for experiments; do not flip default.

## Next B1 angle

While-hot: compute RIJ right after Gt load (gt/chi seconds still hot), then end-scope those seconds before B/At loads — shortens the long gt-second live range without restencil. Separate from this CSE hoist.
