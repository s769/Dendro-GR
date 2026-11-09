/**
 * @file meshReplay.h
 * @brief Frozen-mesh remesh history record / replay for ensemble + FD Jacobian.
 *
 * Record refinement flags immediately before each remesh (and a post-init
 * octant snapshot for diagnostics). Replay applies those flags and calls
 * Mesh::ReMesh via remesh_and_gridtransfer — required because building a
 * mesh from orphan octants breaks Dendro inter-grid transfer.
 *
 * Env:
 *   DENDRO_MESH_RECORD=<dir>
 *   DENDRO_MESH_REPLAY=<dir>
 */
#pragma once

#include <string>
#include <vector>

#include "TreeNode.h"
#include "mesh.h"

namespace bssn {
namespace mesh_replay {

enum class Mode { Off, Record, Replay };

Mode mode();
const std::string& dir();
void init(MPI_Comm comm);

bool due(DendroIntL step);

/** Record local refinement flags (+ optional post-remesh oct snapshot). */
int record_flags(ot::Mesh* mesh, DendroIntL step, DendroScalar time);

/** Load flags for the due event onto @p mesh; advances replay cursor. */
int apply_due_flags(ot::Mesh* mesh);

/** Optional: record local octants after a remesh (diagnostics / Tier 3.2). */
int record_oct_snapshot(const ot::Mesh* mesh, DendroIntL step,
                        DendroScalar time);

std::size_t remaining();

}  // namespace mesh_replay
}  // namespace bssn
