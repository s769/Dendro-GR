#ifndef PHYSCON_H
#define PHYSCON_H

#include <iostream>

#include "derivs.h"
#include "grUtils.h"
#include "parameters.h"

void physical_constraints(double **uZipConVars, const double **uZipVars,
                          const unsigned int &offset, const double *pmin,
                          const double *pmax, const unsigned int *sz,
                          const unsigned int &bflag);


#include <vector>

struct PhysConBlock {
    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
};

/** physical_constraints over many blocks, OpenMP-parallel (OMP_NUM_THREADS). */
void physical_constraints_blocks(double **uZipConVars, const double **uZipVars,
                                 const std::vector<PhysConBlock> &blks);

#endif
