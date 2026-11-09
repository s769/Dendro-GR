# B2 baseline — ensK v1 under `launch_bounds(343,1)`

Source: `tier21_ensK_lb1.log` (lb1) and `tier21_ensK_baseline.log` (lb3, earlier same day).

## lb1 (prod)

| nblocks | K | ensK / serialK |
|--------:|--:|---------------:|
| 64 | 2 | 1.462× |
| 64 | 4 | 1.461× |
| 64 | 8 | 1.516× |
| 256 | 2 | 1.016× |
| 256 | 4 | 1.119× |
| 256 | 8 | 1.169× |
| 2048 | 2 | 1.040× |
| 2048 | 4 | 1.043× |
| 2048 | 8 | 1.044× |

## Notes

- Headroom still at small-nblocks; 2048 nearly flat.
- F32SU under lb1 is **slower** than ASYNC (2.36 vs 2.06 on q8 meshprobe) — park F32SU.
- kshare still not preferred over ensK at 64bl.
