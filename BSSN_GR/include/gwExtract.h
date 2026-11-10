/**
 *@author: Milinda Fernando.
 *School of Computing University of Utah
 *@brief: Contains the COG/SymPyGR (symbolic) code generation to perform
 *gravitational wave (GW) extraction.
 *
 * The extraction method is based on "Extraction of Gravitational Waves in
 *Numerical Relativity" https://arxiv.org/abs/1606.02532 : Chapter 2 , 3, 4.
 *
 * For Spin Weighted Spherical Harmonics (SWSH) we use the SphericalFunction
 *module based on https://arxiv.org/pdf/1604.08140.pdf
 *
 *
 * */

#ifndef DENDRO_5_0_GWEXTRACT_H
#define DENDRO_5_0_GWEXTRACT_H

#include <cmath>
#include <complex>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "daUtils.h"
#include "dendro.h"
#include "grDef.h"
#include "mesh.h"
#include "parameters.h"
#include "point.h"

extern "C" void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
                       double* b, int* ldb, int* info);

namespace GW {

// Definitions related to SWHS and Lebedev quad. to perform GW extraction.
#include "nrswsh.h"

static_assert(sizeof(LEBEDEV_SWSH) / sizeof(LEBEDEV_SWSH[0]) ==
                  (GW::BSSN_GW_MAX_L + 1) * (GW::BSSN_GW_MAX_L + 1) - 4,
              "BSSN_GW_MAX_L does not match the harmonic table");


/**
 * Cached psi4 sphere sampler (DENDRO_GW_CACHE_SAMPLER, default ON).
 * The Lebedev sphere points are fixed in space, so the octree search and the
 * 1D Lagrange weights only change when the mesh does. Points are grouped by
 * element; each extraction gathers each unique element's nodal values once per
 * field via Mesh::getElementNodalValues (which handles hanging nodes exactly as
 * the generic interpolateToCoords does) and applies the cached weights in the
 * same x->y->z contraction order, so results agree to rounding.
 */
struct PsiSamplerEle {
    unsigned int ele;
    std::vector<unsigned int> pts;  // index into the Lebedev point list
    std::vector<double> w;          // 3*(order+1) 1D weights per point (x,y,z)
};
struct PsiSampler {
    unsigned long long sig = 0;
    bool valid             = false;
    std::vector<std::vector<PsiSamplerEle>> perRadius;
    std::vector<std::vector<unsigned int>> validIdx;  // sorted, per radius
};

inline unsigned long long psiMeshSignature(const ot::Mesh* mesh) {
    const ot::TreeNode* el = mesh->getAllElements().data();
    unsigned long long h   = 1469598103934665603ull;
    auto mix               = [&](unsigned long long v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(mesh->getNumLocalMeshElements());
    mix(mesh->getMPICommSize());
    // Element ids index the full (pre-ghost + local + post-ghost) array, so the
    // local-begin offset and total size must be part of the signature: a
    // neighbour-rank remesh can shift them while the local elements stay put.
    mix(mesh->getElementLocalBegin());
    mix(mesh->getElementLocalEnd());
    mix(mesh->getAllElements().size());
    for (unsigned int e = mesh->getElementLocalBegin();
         e < mesh->getElementLocalEnd(); e++) {
        mix(el[e].minX());
        mix(el[e].minY());
        mix(el[e].minZ());
        mix(el[e].getLevel());
    }
    for (unsigned int k = 0; k < BSSN_GW_NUM_RADAII; k++) {
        unsigned long long b;
        std::memcpy(&b, &BSSN_GW_RADAII[k], sizeof(b));
        mix(b);
    }
    return h;
}

inline void buildPsiSampler(const ot::Mesh* mesh, PsiSampler& S,
                            const Point* grid_limit, const Point* domain_limit,
                            unsigned int numPts) {
    S.perRadius.assign(BSSN_GW_NUM_RADAII, {});
    S.validIdx.assign(BSSN_GW_NUM_RADAII, {});
    if (!mesh->isActive()) return;
    const unsigned int rankActive = mesh->getMPIRank();
    const unsigned int npesActive = mesh->getMPICommSize();
    const unsigned int eleOrder   = mesh->getElementOrder();
    const double gR[3] = {grid_limit[1].x() - grid_limit[0].x(),
                          grid_limit[1].y() - grid_limit[0].y(),
                          grid_limit[1].z() - grid_limit[0].z()};
    const double dR[3] = {domain_limit[1].x() - domain_limit[0].x(),
                          domain_limit[1].y() - domain_limit[0].y(),
                          domain_limit[1].z() - domain_limit[0].z()};
    const ot::TreeNode* allElements = mesh->getAllElements().data();
    std::vector<ot::TreeNode> meshOctree(allElements,
                                         allElements +
                                             mesh->getAllElements().size());
    const unsigned int lB = mesh->getElementLocalBegin();
    const unsigned int lE = mesh->getElementLocalEnd();

    for (unsigned int k = 0; k < BSSN_GW_NUM_RADAII; k++) {
        std::vector<double> dc(3 * numPts);
        for (unsigned int pts = 0; pts < numPts; pts++) {
            dc[3 * pts + 0] = (BSSN_GW_RADAII[k] * sin(LEBEDEV_THETA[pts]) *
                               cos(LEBEDEV_PHI[pts]));
            dc[3 * pts + 1] = (BSSN_GW_RADAII[k] * sin(LEBEDEV_THETA[pts]) *
                               sin(LEBEDEV_PHI[pts]));
            dc[3 * pts + 2] = (BSSN_GW_RADAII[k] * cos(LEBEDEV_THETA[pts]));
        }
        std::vector<ot::SearchKey> skey;
        for (unsigned int i = 0; i < numPts; i++) {
            const double c[3] = {dc[3 * i], dc[3 * i + 1], dc[3 * i + 2]};
            if (c[0] < domain_limit[0].x() || c[0] > domain_limit[1].x() ||
                c[1] < domain_limit[0].y() || c[1] > domain_limit[1].y() ||
                c[2] < domain_limit[0].z() || c[2] > domain_limit[1].z())
                continue;
            unsigned int oc[3];
            oc[0] = (unsigned int)(grid_limit[0].x() +
                                   (c[0] - domain_limit[0].x()) * (gR[0] / dR[0]));
            oc[1] = (unsigned int)(grid_limit[0].y() +
                                   (c[1] - domain_limit[0].y()) * (gR[1] / dR[1]));
            oc[2] = (unsigned int)(grid_limit[0].z() +
                                   (c[2] - domain_limit[0].z()) * (gR[2] / dR[2]));
            skey.push_back(
                ot::SearchKey(oc[0], oc[1], oc[2], m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
            skey.back().addOwner(i);
        }
        const std::vector<ot::TreeNode>& spl = mesh->getSplitterElements();
        std::vector<ot::Key> splitterKeys;
        for (unsigned int i = 0; i < npesActive; i++) {
            skey.push_back(ot::SearchKey(spl[2 * i]));
            splitterKeys.push_back(ot::Key(spl[2 * i]));
        }
        std::vector<ot::Key> keys;
        mergeKeys(skey, keys);
        SFC::seqSearch::SFC_treeSearch(splitterKeys.data(), keys.data(), 0,
                                       splitterKeys.size(), 0, keys.size(),
                                       m_uiMaxDepth, m_uiMaxDepth,
                                       ROOT_ROTATION);
        const unsigned int myBegin = splitterKeys[rankActive].getSearchResult();
        const unsigned int myEnd =
            (rankActive + 1 < npesActive)
                ? splitterKeys[rankActive + 1].getSearchResult()
                : keys.size();
        SFC::seqSearch::SFC_treeSearch(keys.data(), meshOctree.data(), myBegin,
                                       myEnd, lB, lE, m_uiMaxDepth,
                                       m_uiMaxDepth, ROOT_ROTATION);
        std::map<unsigned int, PsiSamplerEle> byEle;
        for (unsigned int i = myBegin; i < myEnd; i++) {
            if (!(keys[i].getFlag() & OCT_FOUND)) continue;
            const unsigned int ele = keys[i].getSearchResult();
            std::vector<unsigned int>* owners = keys[i].getOwnerList();
            const ot::TreeNode& tn            = meshOctree[ele];
            const double pmin[3] = {
                domain_limit[0].x() + (tn.minX() - grid_limit[0].x()) * (dR[0] / gR[0]),
                domain_limit[0].y() + (tn.minY() - grid_limit[0].y()) * (dR[1] / gR[1]),
                domain_limit[0].z() + (tn.minZ() - grid_limit[0].z()) * (dR[2] / gR[2])};
            const double pmax[3] = {
                domain_limit[0].x() + (tn.maxX() - grid_limit[0].x()) * (dR[0] / gR[0]),
                domain_limit[0].y() + (tn.maxY() - grid_limit[0].y()) * (dR[1] / gR[1]),
                domain_limit[0].z() + (tn.maxZ() - grid_limit[0].z()) * (dR[2] / gR[2])};
            PsiSamplerEle& E = byEle[ele];
            E.ele            = ele;
            for (unsigned int w = 0; w < owners->size(); w++) {
                const unsigned int pi = (*owners)[w];
                double c[3]           = {dc[3 * pi], dc[3 * pi + 1], dc[3 * pi + 2]};
                for (int a = 0; a < 3; a++) {  // same snapping as lagrangeInterp...
                    if (fabs(c[a] - pmin[a]) < 1e-6) c[a] = pmin[a];
                    if (fabs(c[a] - pmax[a]) < 1e-6) c[a] = pmax[a];
                }
                E.pts.push_back(pi);
                for (int a = 0; a < 3; a++) {
                    const double h = (pmax[a] - pmin[a]) / eleOrder;
                    for (unsigned int li = 0; li <= eleOrder; li++) {
                        double L = 1.0;
                        for (unsigned int lj = 0; lj <= eleOrder; lj++) {
                            if (lj == li) continue;
                            L *= ((c[a] - (pmin[a] + h * lj))) /
                                 ((pmin[a] + h * li) - (pmin[a] + h * lj));
                        }
                        E.w.push_back(L);
                    }
                }
                S.validIdx[k].push_back(pi);
            }
        }
        std::sort(S.validIdx[k].begin(), S.validIdx[k].end());
        for (auto& kv : byEle) S.perRadius[k].push_back(std::move(kv.second));
    }
    S.valid = true;
}

/** out_real/out_imag[pt] for radius k using the cached sampler. */
inline void evalPsiSampler(const ot::Mesh* mesh, const PsiSampler& S,
                           unsigned int k, const double* inR, const double* inI,
                           double* outR, double* outI) {
    const unsigned int eo  = mesh->getElementOrder();
    const unsigned int n1  = eo + 1;
    const unsigned int nPe = mesh->getNumNodesPerElement();
    const std::vector<PsiSamplerEle>& els = S.perRadius[k];
#pragma omp parallel
    {
        std::vector<double> nvR(nPe), nvI(nPe), yzR(n1 * n1), yzI(n1 * n1),
            zR(n1), zI(n1);
        // Per-thread scratch for hanging-node interpolation: the default
        // getElementNodalValues overload shares RefElement::im_vec1/2 and is
        // not thread-safe.
        std::vector<double> im1(nPe), im2(nPe);
#pragma omp for schedule(dynamic, 4)
        for (long q = 0; q < (long)els.size(); q++) {
            const PsiSamplerEle& E = els[q];
            mesh->getElementNodalValues(inR, nvR.data(), E.ele, false, im1.data(),
                                        im2.data());
            mesh->getElementNodalValues(inI, nvI.data(), E.ele, false, im1.data(),
                                        im2.data());
            for (size_t p = 0; p < E.pts.size(); p++) {
                const double* wx = &E.w[p * 3 * n1];
                const double* wy = wx + n1;
                const double* wz = wy + n1;
                for (unsigned int kk = 0; kk < n1; kk++)
                    for (unsigned int j = 0; j < n1; j++) {
                        double sr = 0.0, si = 0.0;
                        for (unsigned int li = 0; li < n1; li++) {
                            sr += nvR[kk * n1 * n1 + j * n1 + li] * wx[li];
                            si += nvI[kk * n1 * n1 + j * n1 + li] * wx[li];
                        }
                        yzR[kk * n1 + j] = sr;
                        yzI[kk * n1 + j] = si;
                    }
                for (unsigned int kk = 0; kk < n1; kk++) {
                    double sr = 0.0, si = 0.0;
                    for (unsigned int li = 0; li < n1; li++) {
                        sr += yzR[kk * n1 + li] * wy[li];
                        si += yzI[kk * n1 + li] * wy[li];
                    }
                    zR[kk] = sr;
                    zI[kk] = si;
                }
                double vr = 0.0, vi = 0.0;
                for (unsigned int li = 0; li < n1; li++) {
                    vr += zR[li] * wz[li];
                    vi += zI[li] * wz[li];
                }
                outR[E.pts[p]] = vr;
                outI[E.pts[p]] = vi;
            }
        }
    }
}

/**
 * @brief extract the GW waves using poly fit.
 * @param [in] mesh: input mesh
 * @param [in] cVar: constraint variables.
 * @param [in] timestep : timestep value.
 * @param [in] time: simulation time.
 *
 * */

template <typename T>
void extractFarFieldPsi4(const ot::Mesh* mesh, const T** cVar,
                         unsigned int timestep, double time) {
    // get the active comm and information about it
    unsigned int rankActive = mesh->getMPIRank();
    unsigned int npesActive = mesh->getMPICommSize();
    MPI_Comm commActive     = mesh->getMPICommunicator();
    unsigned int globalRank = mesh->getMPIRankGlobal();

    unsigned int totalModes = 0;
    for (unsigned int l = 0; l < BSSN_GW_NUM_LMODES; l++)
        totalModes += 2 * BSSN_GW_L_MODES[l] + 1;

    const unsigned int TOTAL_MODES = totalModes;

    DendroComplex* swsh_coeff =
        new DendroComplex[BSSN_GW_NUM_RADAII * TOTAL_MODES];
    DendroComplex* swsh_coeff_g =
        new DendroComplex[BSSN_GW_NUM_RADAII * TOTAL_MODES];

    std::vector<unsigned int> lmCounts;
    std::vector<unsigned int> lmOffset;

    lmCounts.resize(BSSN_GW_NUM_LMODES);
    lmOffset.resize(BSSN_GW_NUM_LMODES);

    for (unsigned int l = 0; l < BSSN_GW_NUM_LMODES; l++)
        lmCounts[l] = 2 * BSSN_GW_L_MODES[l] + 1;

    lmOffset[0] = 0;
    omp_par::scan(&(*(lmCounts.begin())), &(*(lmOffset.begin())),
                  BSSN_GW_NUM_LMODES);

    std::vector<double> psi4L2R;
    std::vector<double> psi4L2I;

    std::vector<double> psi4L2R_g;
    std::vector<double> psi4L2I_g;

    psi4L2R.resize(BSSN_GW_NUM_RADAII, 0);
    psi4L2I.resize(BSSN_GW_NUM_RADAII, 0);
    psi4L2R_g.resize(BSSN_GW_NUM_RADAII, 0);
    psi4L2I_g.resize(BSSN_GW_NUM_RADAII, 0);

    if (mesh->isActive()) {
        const unsigned int rankActive = mesh->getMPIRank();
        const unsigned int npesActive = mesh->getMPICommSize();

        const unsigned int numPts     = LEBEDEV_NUM_PTS;

        std::vector<double> domain_coords;
        domain_coords.resize(3 * LEBEDEV_NUM_PTS);

        std::vector<double> psi4_real;
        psi4_real.resize(LEBEDEV_NUM_PTS);

        std::vector<double> psi4_imag;
        psi4_imag.resize(LEBEDEV_NUM_PTS);

        std::vector<unsigned int> validIndex;

        Point grid_limits[2];
        Point domain_limits[2];

        grid_limits[0] =
            Point(bssn::BSSN_OCTREE_MIN[0], bssn::BSSN_OCTREE_MIN[1],
                  bssn::BSSN_OCTREE_MIN[2]);
        grid_limits[1] =
            Point(bssn::BSSN_OCTREE_MAX[0], bssn::BSSN_OCTREE_MAX[1],
                  bssn::BSSN_OCTREE_MAX[2]);

        domain_limits[0] =
            Point(bssn::BSSN_COMPD_MIN[0], bssn::BSSN_COMPD_MIN[1],
                  bssn::BSSN_COMPD_MIN[2]);
        domain_limits[1] =
            Point(bssn::BSSN_COMPD_MAX[0], bssn::BSSN_COMPD_MAX[1],
                  bssn::BSSN_COMPD_MAX[2]);

        static PsiSampler sampler;
        static const bool use_cache = []() {
            const char* e = std::getenv("DENDRO_GW_CACHE_SAMPLER");
            return !(e && e[0] == '0');  // default ON
        }();
        if (use_cache) {
            const unsigned long long sig = psiMeshSignature(mesh);
            if (!sampler.valid || sampler.sig != sig) {
                buildPsiSampler(mesh, sampler, grid_limits, domain_limits,
                                numPts);
                sampler.sig = sig;
            }
        }

        for (unsigned int k = 0; k < BSSN_GW_NUM_RADAII; k++) {
            for (unsigned int pts = 0; pts < numPts; pts++) {
                domain_coords[3 * pts + 0] =
                    (BSSN_GW_RADAII[k] * sin(LEBEDEV_THETA[pts]) *
                     cos(LEBEDEV_PHI[pts]));
                domain_coords[3 * pts + 1] =
                    (BSSN_GW_RADAII[k] * sin(LEBEDEV_THETA[pts]) *
                     sin(LEBEDEV_PHI[pts]));
                domain_coords[3 * pts + 2] =
                    (BSSN_GW_RADAII[k] * cos(LEBEDEV_THETA[pts]));
            }

            if (use_cache) {
                evalPsiSampler(mesh, sampler, k,
                               cVar[bssn::VAR_CONSTRAINT::C_PSI4_REAL],
                               cVar[bssn::VAR_CONSTRAINT::C_PSI4_IMG],
                               psi4_real.data(), psi4_imag.data());
                validIndex = sampler.validIdx[k];
                static const bool verify =
                    std::getenv("DENDRO_GW_CACHE_VERIFY") != nullptr;
                if (verify) {
                    std::vector<double> chkR(numPts, 0.0), chkI(numPts, 0.0);
                    std::vector<unsigned int> vi0, vi1;
                    ot::da::interpolateToCoords(
                        mesh, cVar[bssn::VAR_CONSTRAINT::C_PSI4_REAL],
                        domain_coords.data(), domain_coords.size(),
                        grid_limits, domain_limits, chkR.data(), vi0);
                    ot::da::interpolateToCoords(
                        mesh, cVar[bssn::VAR_CONSTRAINT::C_PSI4_IMG],
                        domain_coords.data(), domain_coords.size(),
                        grid_limits, domain_limits, chkI.data(), vi1);
                    double md = 0.0;
                    for (unsigned int q = 0; q < vi0.size(); q++)
                        md = std::max(md, fabs(chkR[vi0[q]] -
                                               psi4_real[vi0[q]]));
                    printf("[gw-cache-verify] rank %u ts %u radius %u: n_orig "
                           "%zu n_cache %zu same_idx %d max|dR| %.3e\n",
                           rankActive, timestep, k, vi0.size(),
                           validIndex.size(), (int)(vi0 == validIndex), md);
                }
            } else {
            validIndex.clear();
            ot::da::interpolateToCoords(
                mesh, cVar[bssn::VAR_CONSTRAINT::C_PSI4_REAL],
                domain_coords.data(), domain_coords.size(), grid_limits,
                domain_limits, &(*(psi4_real.begin())), validIndex);

            validIndex.clear();
            ot::da::interpolateToCoords(
                mesh, cVar[bssn::VAR_CONSTRAINT::C_PSI4_IMG],
                domain_coords.data(), domain_coords.size(), grid_limits,
                domain_limits, &(*(psi4_imag.begin())), validIndex);
            }

            for (unsigned int index = 0; index < validIndex.size(); index++) {
                psi4L2R[k] += (psi4_real[validIndex[index]] *
                               psi4_real[validIndex[index]]);
                psi4L2I[k] += (psi4_imag[validIndex[index]] *
                               psi4_imag[validIndex[index]]);
            }

            for (unsigned int l = 0; l < BSSN_GW_NUM_LMODES; l++) {
                // the table holds every l from 2 up, in order
                const unsigned int swshOffset =
                    BSSN_GW_L_MODES[l] * BSSN_GW_L_MODES[l] - 4;
                for (unsigned int m = 0; m < 2 * BSSN_GW_L_MODES[l] + 1; m++) {
                    swsh_coeff[k * TOTAL_MODES + lmOffset[l] + m] =
                        DendroComplex(0.0, 0.0);
                    for (unsigned int index = 0; index < validIndex.size();
                         index++) {
                        DendroComplex psi4 =
                            DendroComplex(psi4_real[validIndex[index]],
                                          psi4_imag[validIndex[index]]);
                        swsh_coeff[k * TOTAL_MODES + lmOffset[l] + m] +=
                            (psi4 *
                             std::conj(LEBEDEV_SWSH[swshOffset + m]
                                                   [validIndex[index]]) *
                             LEBEDEV_W[validIndex[index]]);
                    }

                    // Eric and I both agreed that r^2 here is not necessary but
                    // the 4*M_PI should be there.
                    //  swsh_coeff[k*TOTAL_MODES+lmOffset[l]+m]*=(4*M_PI*BSSN_GW_RADAII[k]*BSSN_GW_RADAII[k]);
                    swsh_coeff[k * TOTAL_MODES + lmOffset[l] + m] *= (4 * M_PI);
                }
            }
        }
    }

    // par::Mpi_Reduce(swsh_coeff,swsh_coeff_g,(BSSN_GW_NUM_RADAII*TOTAL_MODES),MPI_SUM,0,commActive);
    // for(unsigned int k=0;k<BSSN_GW_NUM_RADAII;k++)
    MPI_Reduce(swsh_coeff, swsh_coeff_g, (BSSN_GW_NUM_RADAII * TOTAL_MODES),
               MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, commActive);
    MPI_Reduce(&(*(psi4L2R.begin())), &(*(psi4L2R_g.begin())),
               BSSN_GW_NUM_RADAII, MPI_DOUBLE, MPI_SUM, 0, commActive);
    MPI_Reduce(&(*(psi4L2I.begin())), &(*(psi4L2I_g.begin())),
               BSSN_GW_NUM_RADAII, MPI_DOUBLE, MPI_SUM, 0, commActive);

    if (!globalRank) {
        int n = BSSN_GW_NUM_RADAII;

        std::ofstream fileGW;
        char fName[256];
        sprintf(fName, "%s_GW_L2.dat", bssn::BSSN_PROFILE_FILE_PREFIX.c_str());
        fileGW.open(fName, std::ofstream::app);

        if (timestep == 0) {
            fileGW << "TimeStep\t"
                   << " t\t";
            for (unsigned int r = 0; r < BSSN_GW_NUM_RADAII; r++)
                fileGW << "r=" << BSSN_GW_RADAII[r] << "\t";
            fileGW << std::endl;
        }

        fileGW << timestep << "\t" << time << "\t";
        for (unsigned int r = 0; r < BSSN_GW_NUM_RADAII; r++) {
            psi4L2R_g[r] = sqrt(psi4L2R_g[r]);
            psi4L2I_g[r] = sqrt(psi4L2I_g[r]);
            fileGW << "(" << psi4L2R_g[r] << "," << psi4L2I_g[r] << ")\t";
        }
        fileGW << std::endl;
        fileGW.close();

        for (unsigned int l = 0; l < BSSN_GW_NUM_LMODES; l++) {
            for (unsigned int m = 0; m < 2 * BSSN_GW_L_MODES[l] + 1; m++) {
                // write data to file
                std::ofstream fileGW;
                char fName[256];

                // std::cout<<"l : "<<BSSN_GW_L_MODES[l]<< " m :
                // "<<(int)(m-BSSN_GW_L_MODES[l])<<std::endl;
                sprintf(fName, "%s_GW_l%d_m%d.dat",
                        bssn::BSSN_PROFILE_FILE_PREFIX.c_str(),
                        BSSN_GW_L_MODES[l], (int)(m - BSSN_GW_L_MODES[l]));
                fileGW.open(fName, std::ofstream::app);

                if (timestep == 0) {
                    fileGW << "TimeStep\t"
                           << " t\t";
                    for (unsigned int r = 0; r < BSSN_GW_NUM_RADAII; r++)
                        fileGW << "r" << r << "\t";

                    fileGW << std::endl;
                }

                fileGW.precision(GW::BSSN_GW_OUTPUT_PRECISION);
                fileGW << std::scientific;
                fileGW << timestep << "\t" << time << "\t";
                for (unsigned int r = 0; r < BSSN_GW_NUM_RADAII; r++)
                    fileGW << swsh_coeff_g[r * TOTAL_MODES + lmOffset[l] + m]
                           << "\t";

                fileGW << std::endl;

                fileGW.close();
            }
        }
    }

    delete[] swsh_coeff;
    delete[] swsh_coeff_g;

    return;
}

}  // end of namespace GW

#endif
