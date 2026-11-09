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

Design. KSHARE (~10–15%) already showed geometry share is not enough — this is the real hang lever.
