#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
# 1) frozen-mesh CPU vs GPU (identical mesh) physics check
true
# 2) OpenMP thread-count sweep at production depth (init_grid, remesh, extraction scale)
# 3) TwoPunctures cold vs warm, thread scaling
echo BATCH_DONE > tier0/batch.done
