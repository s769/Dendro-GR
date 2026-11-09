# Tier 1.3 v2 — K members per CTA

v1 launches `grid=(nblocks,K)` with independent smem tiles (launch amortization only).

## Implementation (first cut)

`bssnrhs3_pt_async(..., K, kshare=true)` → `grid=(nblocks,1)`, kernel arg `K_share=K`.
One CTA per mesh block; block geometry / dx / η loaded once; loop `k=0..K-1` over member state into the same `su` tile (still K algebra + FD passes).

Bit-id: identical inputs in every slot → `mismatched_slots=0` vs K=1 reference (same as v1).

## Measured (c127-032, GB200, nReps=20, K=8)

| nblocks | single (s) | serial-K | ensK v1 | kshare | kshare/single | ensK/kshare |
|--------:|----------:|---------:|--------:|-------:|--------------:|------------:|
| 64 | 0.0031 | 0.0266 | **0.0177** | 0.0249 | 8.12× | 0.71× (v1 faster) |
| 256 | 0.0088 | 0.0644 | 0.0544 | 0.0537 | 6.11× | ~1.01× |
| 2048 | 0.0622 | 0.4003 | 0.3851 | 0.3797 | 6.10× | ~1.01× |

Raw: `tier0/ensemble_kshare_c127-032.txt`.

## Verdict

**No-go for production path.** Serializing K algebra inside one CTA underfills the GPU at small `nblocks` (the regime we care about). v1 `grid.y=K` remains the ensemble launch. Further gains need true shared-FD / multi-member-in-registers work, not a CTA-serial loop.

Status: first cut measured; keep API behind `kshare` flag for experiments only.
