# Tier 2 / queue #7 — coalesced hang gather

## Evidence

NCU on `__unzip_cg1__2` (hang-majority): est. **50–64%** speedup from uncoalesced global **loads**, **28–36%** from stores. Occupancy/TB88 closed as secondary.

Hang gather today (`mesh_gpu.cuh`):

```text
v3[0][local] = dptr_in[var*cg_sz + e2n_cg[ele*nPe + local]]
```

Neighboring threads in a 7×7 CTA hit unrelated CG indices → sector waste.

## Approaches (ordered)

1. **Frozen-mesh gather plan (preferred)**  
   At mesh build / first unzip under replay: for each hang element, emit a compact list of unique parent CG ids + local scatter map. Runtime: coalesced load of unique parents into smem, then local remap. Extra memory O(nPe) per hang ele; rebuild only on remesh.

2. **CG-driven scatter**  
   Invert map: CTA over CG nodes writes into element DG slots that reference them. Better load coalescing; harder write conflicts.

3. **Warp-shared parent octant**  
   When hang child shares one parent, one warp loads parent nPe coalesced, then p2c. Only helps single-parent hang (common).

## Success metric

Same q8/4GPU short: unzip kernel time −≥15% ETS, or NCU excessive sectors −≥30% on `__unzip_cg1__2`.

## Status

### Unique-CG + jump stats (q8 hang list, `DENDRO_UNZIP_GATHER_STATS=1`)

`n=4337` hang elements: **unique_cg = 343 = nPe** → compress **1.00×** (dedupe dead).

Adjacent CG-id jumps (c106-029):

| order | mean Σ\|Δid\| |
|-------|-------------:|
| lex (current gather) | ~3.41e6 |
| sorted by cg_id | ~1.58e5 |
| **sort_jump_ratio** | **0.046** (~22× tighter) |

**Sort-by-cg_id gather A/B (q8/4GPU, `TIME_END=0.5`, c106-029):**

| mode | ETS (s) |
|------|--------:|
| baseline | **2.180** |
| `DENDRO_UNZIP_SORT_GATHER=1` | 3.889 (**+78%**) |

**NO-GO.** Odd-even sort in-CTA dominates; clustering signal does not pay off at nPe=343. Keep behind env; do not enable. Next: CG-driven scatter or park unzip micro-opts.

Raw: `tier0/tier2_gather_jump_c106.out`, `tier0/tier2_sort_gather_ab.log`.
