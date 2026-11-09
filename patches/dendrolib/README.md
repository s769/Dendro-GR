# dendrolib local patches

`current_local_full.patch` = complete `git diff` of the working dendrolib checkout used for all
campaign builds, against dendrolib commit `326ffc3a580d303a0b668a11060d18191daccdea`.
Reproduce: clone dendrolib at that commit, `git apply patches/dendrolib/current_local_full.patch`,
then configure Dendro-GR with `-DDENDRO_dendrolib_DIR=<that checkout>`.
The other files here are earlier per-feature patches/snippets (historical).
