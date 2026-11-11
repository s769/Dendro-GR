# dendrolib local patches

`current_local_full.patch` = complete `git diff` of the working dendrolib checkout used for all
campaign builds, against dendrolib commit `326ffc3a580d303a0b668a11060d18191daccdea`.
Reproduce: clone dendrolib at that commit, `git apply patches/dendrolib/current_local_full.patch`,
then configure Dendro-GR with `-DDENDRO_dendrolib_DIR=<that checkout>`.
The other files here are earlier per-feature patches/snippets (historical).

## 2026-11-11 additions (apply on top of current_local_full.patch)
- `pinned_pool.patch` (include/device.h): size-bucketed pinned host memory pool (DENDRO_PINNED_POOL=0 disables) -6.5% ETS at cap 13.
- `igt_omp.patch` (include/mesh.tcc, include/dvec.h): OpenMP single-rank INJECTION intergrid transfer (DENDRO_IGT_OMP=0 disables) + [igt-phases] timers (DENDRO_TIME_REMESH=1). Needs OpenMP in the CUDA compile (now set in the top-level CMakeLists.txt).
- `zip_grid_over65535.patch` (GPU/include/mesh_gpu.cuh): the zip kernels (`__zip_cg__`, `__zip_dg__`, `__zip_dg_enforce_c0__`) launched with gridDim.y = #blocks (= #elements here), which fails ("!! invalid argument") above 65535 elements (reproduced in 3 runs: p13L 65717, p14L 66032 elements, q4 (15,12) 70442). Axes swapped: grid = (#blocks, max elems/block, dof). VERIFICATION PENDING (needs a >65535-element run; first attempt: big5b_T1).
