/**
 * @file bssnCtxGPU.cu
 * @brief BSSN GPU ctx file
 * @version 0.1
 * @date 2022-02-08
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <vector>

#include "bssnChkptSchema.h"
#include "bssnCtxGPU.cuh"
#include <chrono>
#include "meshReplay.h"
// CONST_MEM DEVICE_REAL device::refel_1d[2 * REFEL_CONST_MEM_MAX];

namespace bssn {
BSSNCtxGPU::BSSNCtxGPU(ot::Mesh* pMesh) : Ctx() {
    m_uiMesh    = pMesh;

    // if(m_uiMesh)
    m_mesh_cpu  = device::MeshGPU();
    m_dptr_mesh = m_mesh_cpu.alloc_mesh_on_device(m_uiMesh);

    // variable allocation for evolution variables
    m_var[VL::CPU_EV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, BSSN_NUM_VARS, true);
    m_var[VL::CPU_CV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST,
                                    BSSN_CONSTRAINT_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        BSSN_NUM_VARS, true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        BSSN_CONSTRAINT_NUM_VARS, true);

    m_var[VL::GPU_EV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::DEVICE, BSSN_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        BSSN_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_OUT].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        BSSN_NUM_VARS, true);

    m_uiTinfo._m_uiStep = 0;
    m_uiTinfo._m_uiT    = 0;
    m_uiTinfo._m_uiTb   = BSSN_RK_TIME_BEGIN;
    m_uiTinfo._m_uiTe   = BSSN_RK_TIME_END;
    m_uiTinfo._m_uiTh   = BSSN_RK45_TIME_STEP_SIZE;

    m_uiElementOrder    = BSSN_ELE_ORDER;

    m_uiMinPt    = Point(BSSN_GRID_MIN_X, BSSN_GRID_MIN_Y, BSSN_GRID_MIN_Z);
    m_uiMaxPt    = Point(BSSN_GRID_MAX_X, BSSN_GRID_MAX_Y, BSSN_GRID_MAX_Z);

    // deriv vars,
    // m_deriv_evars      = new BSSN_EVAR_DERIVS[DEVICE_RHS_NSTREAMS];
    // m_dptr_deriv_evars =
    // (BSSN_EVAR_DERIVS*)GPUDevice::device_malloc<BSSN_EVAR_DERIVS>(DEVICE_RHS_NSTREAMS);

    // const unsigned int ws_szp_stream = DEVICE_MAX_DERIVS * DEVICE_RHS_BLK_SZ
    // * DEVICE_RHS_BATCHED_GRAIN_SZ; m_deriv_base       =
    // GPUDevice::device_malloc<DEVICE_REAL>(DEVICE_RHS_NSTREAMS *
    // ws_szp_stream); for(unsigned int i=0; i < DEVICE_RHS_NSTREAMS; i++)
    // {
    //     BSSN_EVAR_DERIVS * const deriv_evars=m_deriv_evars + i;
    //     DEVICE_REAL* const  deriv_base = m_deriv_base + i* ws_szp_stream;
    //     const unsigned int BATCHED_BLOCKS_SZ = DEVICE_RHS_BATCHED_GRAIN_SZ;
    //     const unsigned int BLK_SZ = DEVICE_RHS_BLK_SZ;
    //     #include "../scripts/bssnrhs_memalloc.cuh"
    // }
    // GPUDevice::host_to_device<BSSN_EVAR_DERIVS>(m_deriv_evars,m_dptr_deriv_evars,DEVICE_RHS_NSTREAMS);
    int maxbytes = 13 * 13 * 13 * 8 * 3;  // 3 blocks KB
    cudaFuncSetAttribute(device::eval_rhs3<3, 13, 13>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, BSSN_NUM_VARS,
                                      BSSN_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, BSSN_NUM_VARS,
                                    BSSN_ASYNC_COMM_K);

    device::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                          BSSN_NUM_VARS, BSSN_ASYNC_COMM_K);
    device::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                        BSSN_NUM_VARS, BSSN_ASYNC_COMM_K);

    m_dptr_min_det = GPUDevice::device_malloc<double>(1);
    GPUDevice::host_to_device<double>(&m_dMinDetGt, m_dptr_min_det, 1);
    // Cap sized for large meshes; grown in post_timestep if needed.
    m_uiBlockMinsCap  = 65536;
    m_dptr_block_mins = GPUDevice::device_malloc<double>(m_uiBlockMinsCap);

    return;
}

BSSNCtxGPU::~BSSNCtxGPU() {
    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();

    // GPUDevice::device_free<DEVICE_REAL>(m_deriv_base);
    // GPUDevice::device_free<BSSN_EVAR_DERIVS>(m_dptr_deriv_evars);
    // delete m_deriv_evars;

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, BSSN_NUM_VARS,
                                      BSSN_ASYNC_COMM_K);
    device::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                          BSSN_NUM_VARS, BSSN_ASYNC_COMM_K);
    m_mesh_cpu.dealloc_mesh_on_device(m_dptr_mesh);
    GPUDevice::device_free<double>(m_dptr_min_det);
    if (m_dptr_block_mins) GPUDevice::device_free<double>(m_dptr_block_mins);
    return;
}

int BSSNCtxGPU::host_to_device_sync() {
    if (!m_uiMesh->isActive()) return 0;

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::H2D].start();
#endif
    DVec& m_evar      = m_var[VL::CPU_EV];
    DVec& m_dptr_evar = m_var[VL::GPU_EV];

    GPUDevice::host_to_device(m_evar.get_vec_ptr(), m_dptr_evar.get_vec_ptr(),
                              m_evar.get_size());

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::H2D].stop();
#endif

    return 0;
}

int BSSNCtxGPU::host_to_device_async(cudaStream_t s) {
    if (!m_uiMesh->isActive()) return 0;

    DVec& m_evar      = m_var[VL::CPU_EV];
    DVec& m_dptr_evar = m_var[VL::GPU_EV];

    GPUDevice::host_to_device_async(
        m_evar.get_vec_ptr(), m_dptr_evar.get_vec_ptr(), m_evar.get_size(), s);

    return 0;
}

int BSSNCtxGPU::device_to_host_sync() {
    if (!m_uiMesh->isActive()) return 0;

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::D2H].start();
#endif

    DVec& m_evar      = m_var[VL::CPU_EV];
    DVec& m_dptr_evar = m_var[VL::GPU_EV];

    GPUDevice::device_to_host(m_evar.get_vec_ptr(), m_dptr_evar.get_vec_ptr(),
                              m_evar.get_size());

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::D2H].stop();
#endif

    return 0;
}

int BSSNCtxGPU::device_to_host_async(cudaStream_t s) {
    if (!m_uiMesh->isActive()) return 0;

    DVec& m_evar      = m_var[VL::CPU_EV];
    DVec& m_dptr_evar = m_var[VL::GPU_EV];

    GPUDevice::device_to_host_async<DEVICE_REAL, cudaStream_t>(
        m_evar.get_vec_ptr(), m_dptr_evar.get_vec_ptr(), m_evar.get_size(), s);

    return 0;
}

int BSSNCtxGPU::initialize() {
    if (bssn::BSSN_RESTORE_SOLVER) {
        this->restore_checkpt();
        this->host_to_device_sync();
        return 0;
    }

    this->init_grid();

    bool isRefine = false;
    DendroIntL oldElements, oldElements_g;
    DendroIntL newElements, newElements_g;

    DendroIntL oldGridPoints, oldGridPoints_g;
    DendroIntL newGridPoints, newGridPoints_g;

    unsigned int iterCount         = 1;
    const unsigned int max_iter    = bssn::BSSN_INIT_GRID_ITER;
    const unsigned int rank_global = m_uiMesh->getMPIRankGlobal();
    MPI_Comm gcomm                 = m_uiMesh->getMPIGlobalCommunicator();

    DendroScalar* unzipVar[bssn::BSSN_NUM_VARS];
    unsigned int refineVarIds[bssn::BSSN_NUM_REFINE_VARS];

    for (unsigned int vIndex = 0; vIndex < bssn::BSSN_NUM_REFINE_VARS; vIndex++)
        refineVarIds[vIndex] = bssn::BSSN_REFINE_VARIABLE_INDICES[vIndex];

    double wTol = bssn::BSSN_WAVELET_TOL;
    std::function<double(double, double, double, double* hx)> waveletTolFunc =
        [](double x, double y, double z, double* hx) {
            return bssn::computeWTolDCoords(x, y, z, hx);
        };

    DVec& m_evar     = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ_IN];

    do {
        this->unzip(m_evar, m_evar_unz, bssn::BSSN_ASYNC_COMM_K);
        m_evar_unz.to_2d(unzipVar);
        // isRefine=this->is_remesh();
        //  enforce WMAR refinement based refinement initially.
        // Under DENDRO_MESH_REPLAY, the init-converge refinement iterations
        // recorded by the fiducial are replayed (no WAMR decisions), so all
        // ensemble members start on the identical mesh.
        if (max_iter == 0)
            isRefine = false;
        else if (bssn::mesh_replay::mode() ==
                 bssn::mesh_replay::Mode::Replay)
            isRefine = (bssn::mesh_replay::apply_init_flags(m_uiMesh,
                                                            iterCount) == 0);
        else {
            isRefine =
                bssn::isReMeshWAMR(m_uiMesh, (const double**)unzipVar,
                                   refineVarIds, bssn::BSSN_NUM_REFINE_VARS,
                                   waveletTolFunc, bssn::BSSN_DENDRO_AMR_FAC);
            if (isRefine) bssn::mesh_replay::record_init_flags(m_uiMesh, iterCount);
        }

        if (isRefine) {
            ot::Mesh* newMesh =
                this->remesh(bssn::BSSN_DENDRO_GRAIN_SZ,
                             bssn::BSSN_LOAD_IMB_TOL, bssn::BSSN_SPLIT_FIX);

            oldElements   = m_uiMesh->getNumLocalMeshElements();
            newElements   = newMesh->getNumLocalMeshElements();

            oldGridPoints = m_uiMesh->getNumLocalMeshNodes();
            newGridPoints = newMesh->getNumLocalMeshNodes();

            par::Mpi_Allreduce(&oldElements, &oldElements_g, 1, MPI_SUM, gcomm);
            par::Mpi_Allreduce(&newElements, &newElements_g, 1, MPI_SUM, gcomm);

            par::Mpi_Allreduce(&oldGridPoints, &oldGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());
            par::Mpi_Allreduce(&newGridPoints, &newGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());

            if (!rank_global) {
                std::cout << "[bssnCtx] iter : " << iterCount
                          << " (Remesh triggered) ->  old mesh : "
                          << oldElements_g << " new mesh : " << newElements_g
                          << std::endl;
                std::cout << "[bssnCtx] iter : " << iterCount
                          << " (Remesh triggered) ->  old mesh (zip nodes) : "
                          << oldGridPoints_g
                          << " new mesh (zip nodes) : " << newGridPoints_g
                          << std::endl;
            }

            // grid_transfer now rebuilds the device mesh itself
            this->grid_transfer(newMesh);

            std::swap(m_uiMesh, newMesh);
            delete newMesh;
        }

        iterCount += 1;

    } while (isRefine &&
             (newElements_g != oldElements_g ||
              newGridPoints_g != oldGridPoints_g) &&
             (iterCount < max_iter));

    this->init_grid();

    // // realloc bssn deriv space
    deallocate_bssn_deriv_workspace();
    allocate_bssn_deriv_workspace(m_uiMesh, 1);

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);

    // calculate the minimum dx
    bssn::BSSN_CURRENT_MIN_DX =
        (bssn::min_domain_width() *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)bssn::BSSN_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));

    bssn::BSSN_RK45_TIME_STEP_SIZE =
        bssn::BSSN_CFL_FACTOR *
        (bssn::min_domain_width() *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)bssn::BSSN_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    m_uiTinfo._m_uiTh = bssn::BSSN_RK45_TIME_STEP_SIZE;

    if (!m_uiMesh->getMPIRankGlobal()) {
        const DendroScalar dx_finest =
            (bssn::min_domain_width() *
             ((1u << (m_uiMaxDepth - lmax)) / ((double)bssn::BSSN_ELE_ORDER)) /
             ((double)(1u << (m_uiMaxDepth))));
        const DendroScalar dt_finest = bssn::BSSN_CFL_FACTOR * dx_finest;

        std::cout << "================= Grid Info (After init grid "
                     "converge):==============================================="
                     "========"
                  << std::endl;
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dx: " << dx_finest << std::endl;
        std::cout << "dt: " << dt_finest << std::endl;
        std::cout << "========================================================="
                     "======================================================"
                  << std::endl;
    }

    // Tier 0.2: block-count-per-level histogram + theoretical LTS work ratio.
    // Enable with DENDRO_DUMP_BLK_LEV=1. Uses element level of each block's
    // first local element (same as ENUTS getBlkTimestepFac path).
    if (std::getenv("DENDRO_DUMP_BLK_LEV")) {
        const unsigned int depth = m_uiMaxDepth + 1;
        std::vector<unsigned long long> local_hist(depth, 0), hist(depth, 0);
        const std::vector<ot::Block>& blkList = m_uiMesh->getLocalBlockList();
        const ot::TreeNode* pNodes = m_uiMesh->getAllElements().data();
        for (unsigned int b = 0; b < blkList.size(); b++) {
            const unsigned int lev =
                pNodes[blkList[b].getLocalElementBegin()].getLevel();
            if (lev < depth) local_hist[lev]++;
        }
        MPI_Allreduce(local_hist.data(), hist.data(), (int)depth, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, m_uiMesh->getMPIGlobalCommunicator());
        unsigned long long ntot = 0;
        double work_lts = 0.0;
        for (unsigned int l = 0; l < depth; l++) {
            ntot += hist[l];
            work_lts += (double)hist[l] * (double)(1ull << l);
        }
        const double work_gts =
            (ntot > 0 && lmax < 63)
                ? (double)ntot * (double)(1ull << lmax)
                : 0.0;
        const double ratio = (work_gts > 0.0) ? (work_lts / work_gts) : 0.0;
        if (!m_uiMesh->getMPIRankGlobal()) {
            std::cout << "[BLK_LEV_HIST] nblocks_total=" << ntot
                      << " lmin=" << lmin << " lmax=" << lmax << std::endl;
            for (unsigned int l = 0; l < depth; l++) {
                if (hist[l] == 0) continue;
                std::cout << "[BLK_LEV_HIST] level=" << l
                          << " nblocks=" << hist[l] << std::endl;
            }
            std::cout << "[BLK_LEV_HIST] LTS_work_ratio="
                      << "sum_l N_l*2^l / (N_tot*2^Lmax) = " << ratio
                      << "  (ceiling on GTS/LTS step-work; smaller is better "
                         "for LTS)"
                      << std::endl;
            std::cout << "[BLK_LEV_HIST] GTS_over_LTS_ceiling="
                      << ((ratio > 0.0) ? (1.0 / ratio) : 0.0) << std::endl;
        }
    }

    set_appropriate_derivs(bssn::BSSN_PADDING_WIDTH);

    this->host_to_device_sync();
    return 0;
}

int BSSNCtxGPU::init_grid() {
    DVec& m_evar                = m_var[VL::CPU_EV];
    DVec& m_dptr_evar           = m_var[VL::GPU_EV];

    const ot::TreeNode* pNodes  = &(*(m_uiMesh->getAllElements().begin()));
    const unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int* e2n_cg  = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg  = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe      = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd   = m_uiMesh->getNodeLocalEnd();

    DendroScalar* zipIn[bssn::BSSN_NUM_VARS];
    m_evar.to_2d(zipIn);

    DendroScalar var1[bssn::BSSN_NUM_VARS];

    DendroScalar mp, mm, mp_adm, mm_adm, E, J1, J2, J3;
    // set the TP communicator.
    if (bssn::BSSN_ID_TYPE == 0) {
        TP_MPI_COMM = m_uiMesh->getMPIGlobalCommunicator();
        TwoPunctures((double)0, (double)0, (double)0, var1, &mp, &mm, &mp_adm,
                     &mm_adm, &E, &J1, &J2, &J3);
    }

    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        DendroScalar var[bssn::BSSN_NUM_VARS];
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    const unsigned int nodeLookUp_CG =
                        e2n_cg[elem * nPe +
                               k * (eleOrder + 1) * (eleOrder + 1) +
                               j * (eleOrder + 1) + i];
                    if (nodeLookUp_CG >= nodeLocalBegin &&
                        nodeLookUp_CG < nodeLocalEnd) {
                        const unsigned int nodeLookUp_DG =
                            e2n_dg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        unsigned int ownerID, ii_x, jj_y, kk_z;
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);
                        const DendroScalar len =
                            (double)(1u << (m_uiMaxDepth -
                                            pNodes[ownerID].getLevel()));

                        const DendroScalar x =
                            pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        const DendroScalar y =
                            pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        const DendroScalar z =
                            pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        if (bssn::BSSN_ID_TYPE == 0) {
                            const DendroScalar xx = GRIDX_TO_X(x);
                            const DendroScalar yy = GRIDY_TO_Y(y);
                            const DendroScalar zz = GRIDZ_TO_Z(z);

                            TwoPunctures((double)xx, (double)yy, (double)zz,
                                         var, &mp, &mm, &mp_adm, &mm_adm, &E,
                                         &J1, &J2, &J3);
                        } else {
                            // all other values are handled in the initial data
                            // wrapper including an error message
                            initialDataFunctionWrapper((double)x, (double)y,
                                                       (double)z, var);
                        }
                        for (unsigned int v = 0; v < bssn::BSSN_NUM_VARS; v++)
                            zipIn[v][nodeLookUp_CG] = var[v];
                    }
                }
    }

    for (unsigned int node = m_uiMesh->getNodeLocalBegin();
         node < m_uiMesh->getNodeLocalEnd(); node++)
        m_dMinDetGt =
            std::min(m_dMinDetGt, enforce_bssn_constraints(zipIn, node));

#ifdef BSSN_EXTRACT_BH_LOCATIONS
    m_uiBHLoc[0] = Point(bssn::BH1.getBHCoordX(), bssn::BH1.getBHCoordY(),
                         bssn::BH1.getBHCoordZ());
    m_uiBHLoc[1] = Point(bssn::BH2.getBHCoordX(), bssn::BH2.getBHCoordY(),
                         bssn::BH2.getBHCoordZ());
#endif

    return 0;
}

int BSSNCtxGPU::finalize() { return 0; }

int BSSNCtxGPU::write_vtu(bool extract) {
    if (!m_uiMesh->isActive()) return 0;
    static const bool tm_on = std::getenv("DENDRO_TIME_EXTRACT") != nullptr;
    static double tm_acc[6] = {0, 0, 0, 0, 0, 0};
    static unsigned tm_n    = 0;
    auto tm_now = []() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    double tm0 = tm_on ? tm_now() : 0.0;
    // DENDRO_GW_SHELL_ONLY=1: evaluate constraints/psi4 only on blocks near the
    // GW extraction spheres, using a DEVICE unzip + per-block D2H of just those
    // blocks (no host unzip of the whole mesh). Constraint norms are then
    // INVALID; use only when the waveform is the observable.
    static const bool gw_shell_only = []() {
        const char* e = std::getenv("DENDRO_GW_SHELL_ONLY");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    static const double gw_shell_margin = []() {
        const char* e = std::getenv("DENDRO_GW_SHELL_MARGIN");
        return e ? atof(e) : 2.0;  // in units of the block width
    }();

    DVec& m_evar     = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ_IN];
    DVec& m_cvar     = m_var[VL::CPU_CV];
    DVec& m_cvar_unz = m_var[VL::CPU_CV_UZ_IN];

    if (gw_shell_only)
    {
        this->unzip(m_var[VL::GPU_EV], m_var[VL::GPU_EV_UZ_IN],
                    BSSN_ASYNC_COMM_K);  // device unzip (fast)
        GPUDevice::device_synchronize();
    }
    else
        this->unzip(m_evar, m_evar_unz, BSSN_ASYNC_COMM_K);

    // DENDRO_UNZIP_CMP=1: one-shot comparison of host unzip vs device unzip of
    // the same state (debug; run with DENDRO_GW_SHELL_ONLY unset).
    {
        static bool cmp_done = false;
        if (!cmp_done && std::getenv("DENDRO_UNZIP_CMP") && !gw_shell_only &&
            m_uiMesh->getMPICommSizeGlobal() == 1) {
            cmp_done = true;
            const size_t unz_sz = m_uiMesh->getDegOfFreedomUnZip();
            const size_t tot    = unz_sz * bssn::BSSN_NUM_VARS;
            this->unzip(m_var[VL::GPU_EV], m_var[VL::GPU_EV_UZ_IN],
                        BSSN_ASYNC_COMM_K);
            GPUDevice::device_synchronize();
            std::vector<double> hd(tot);
            cudaMemcpy(hd.data(), m_var[VL::GPU_EV_UZ_IN].get_vec_ptr(),
                       tot * sizeof(double), cudaMemcpyDeviceToHost);
            const double* hh = m_evar_unz.get_vec_ptr();
            const std::vector<ot::Block>& bl = m_uiMesh->getLocalBlockList();
            std::vector<double> wmax(bssn::BSSN_NUM_VARS, 0.0),
                vmax(bssn::BSSN_NUM_VARS, 0.0);
            double worst = 0.0; size_t wb = 0; unsigned wv = 0;
            double byflag[2] = {0, 0};
            for (size_t b = 0; b < bl.size(); b++) {
                const size_t off = bl[b].getOffset();
                const size_t n   = (size_t)bl[b].getAllocationSzX() *
                                 bl[b].getAllocationSzY() *
                                 bl[b].getAllocationSzZ();
                for (unsigned v = 0; v < bssn::BSSN_NUM_VARS; v++)
                    for (size_t i = 0; i < n; i++) {
                        const double a = hh[v * unz_sz + off + i];
                        const double c = hd[v * unz_sz + off + i];
                        const double d = fabs(a - c);
                        if (fabs(a) > vmax[v]) vmax[v] = fabs(a);
                        if (d > wmax[v]) wmax[v] = d;
                        if (d > worst) { worst = d; wb = b; wv = v; }
                        if (d > byflag[bl[b].getBlkNodeFlag() != 0])
                            byflag[bl[b].getBlkNodeFlag() != 0] = d;
                    }
            }
            printf("[unzip-cmp] blocks=%zu worst|diff|=%.3e var=%u block=%zu "
                   "bflag=%u  max diff interior-blocks=%.3e boundary-blocks=%.3e\n",
                   bl.size(), worst, wv, wb, bl[wb].getBlkNodeFlag(),
                   byflag[0], byflag[1]);
            for (unsigned v = 0; v < bssn::BSSN_NUM_VARS; v++)
                printf("[unzip-cmp] var %2u max|a|=%.3e max|diff|=%.3e\n", v,
                       vmax[v], wmax[v]);
        }
    }

    DendroScalar* consUnzipVar[bssn::BSSN_CONSTRAINT_NUM_VARS];
    DendroScalar* consVar[bssn::BSSN_CONSTRAINT_NUM_VARS];

    DendroScalar* evolUnzipVar[bssn::BSSN_NUM_VARS];
    DendroScalar* evolVar[bssn::BSSN_NUM_VARS];

    m_evar_unz.to_2d(evolUnzipVar);
    m_cvar_unz.to_2d(consUnzipVar);

    m_evar.to_2d(evolVar);
    m_cvar.to_2d(consVar);
    double tm1 = tm_on ? tm_now() : 0.0;

#if BSSN_COMPUTE_CONSTRAINTS

    const std::vector<ot::Block> blkList = m_uiMesh->getLocalBlockList();

    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(bssn::BSSN_COMPD_MIN[0], bssn::BSSN_COMPD_MIN[1],
                       bssn::BSSN_COMPD_MIN[2]);
    const Point pt_max(bssn::BSSN_COMPD_MAX[0], bssn::BSSN_COMPD_MAX[1],
                       bssn::BSSN_COMPD_MAX[2]);
    const unsigned int PW = bssn::BSSN_PADDING_WIDTH;

    std::vector<PhysConBlock> pc_blocks;
    pc_blocks.reserve(blkList.size());
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        offset   = blkList[blk].getOffset();
        sz[0]    = blkList[blk].getAllocationSzX();
        sz[1]    = blkList[blk].getAllocationSzY();
        sz[2]    = blkList[blk].getAllocationSzZ();

        bflag    = blkList[blk].getBlkNodeFlag();

        dx       = blkList[blk].computeDx(pt_min, pt_max);
        dy       = blkList[blk].computeDy(pt_min, pt_max);
        dz       = blkList[blk].computeDz(pt_min, pt_max);

        ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
        ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
        ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

        ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
        ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
        ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

        if (gw_shell_only) {
            // Keep only blocks whose (padded) box, widened by one block width,
            // intersects some extraction sphere centred at the origin.
            double dmin2 = 0.0, dmax2 = 0.0;
            double mw    = 0.0;
            for (int a = 0; a < 3; a++) {
                const double lo = ptmin[a], hi = ptmax[a];
                const double nearest =
                    (lo > 0.0) ? lo : ((hi < 0.0) ? hi : 0.0);
                const double farthest =
                    (fabs(lo) > fabs(hi)) ? fabs(lo) : fabs(hi);
                dmin2 += nearest * nearest;
                dmax2 += farthest * farthest;
                if (hi - lo > mw) mw = hi - lo;
            }
            mw *= gw_shell_margin;
            const double dmin = sqrt(dmin2) - mw, dmax = sqrt(dmax2) + mw;
            bool keep = false;
            for (unsigned int ir = 0; ir < GW::BSSN_GW_NUM_RADAII && !keep;
                 ir++) {
                const double R = GW::BSSN_GW_RADAII[ir];
                keep           = (dmin <= R) && (R <= dmax);
            }
            if (!keep) continue;
            // D2H of this block's 24 unzipped evolution vars (one 2D copy).
            const size_t unz_sz = m_uiMesh->getDegOfFreedomUnZip();
            cudaMemcpy2D(evolUnzipVar[0] + offset, unz_sz * sizeof(double),
                         m_var[VL::GPU_EV_UZ_IN].get_vec_ptr() + offset,
                         unz_sz * sizeof(double),
                         (size_t)sz[0] * sz[1] * sz[2] * sizeof(double),
                         bssn::BSSN_NUM_VARS, cudaMemcpyDeviceToHost);
        }
        {
            PhysConBlock pb;
            pb.offset = offset;
            for (int a = 0; a < 3; a++) {
                pb.ptmin[a] = ptmin[a];
                pb.ptmax[a] = ptmax[a];
                pb.sz[a]    = sz[a];
            }
            pb.bflag = bflag;
            pc_blocks.push_back(pb);
        }
    }
    physical_constraints_blocks(consUnzipVar,
                                (const DendroScalar**)evolUnzipVar, pc_blocks);

    /*double consVecMin[bssn::BSSN_CONSTRAINT_NUM_VARS];
    double consVecMax[bssn::BSSN_CONSTRAINT_NUM_VARS];*/
    double constraintMaskedL2[bssn::BSSN_CONSTRAINT_NUM_VARS];
    double tm2 = tm_on ? tm_now() : 0.0;
    this->zip(m_cvar_unz, m_cvar);
    m_uiMesh->readFromGhostBegin(m_cvar.get_vec_ptr(), m_cvar.get_dof());
    m_uiMesh->readFromGhostEnd(m_cvar.get_vec_ptr(), m_cvar.get_dof());
    double tm3 = tm_on ? tm_now() : 0.0;

    if (extract) {
        // pre-enforcement min since the last write; NaN if nothing was
        // enforced
        const double noMinDet = std::numeric_limits<double>::max();
        double devMinDet      = noMinDet;
        GPUDevice::device_to_host<double>(&devMinDet, m_dptr_min_det, 1);
        GPUDevice::host_to_device<double>(&noMinDet, m_dptr_min_det, 1);
        const double locMinDet = std::min(m_dMinDetGt, devMinDet);
        m_dMinDetGt            = noMinDet;
        double minDetGt        = noMinDet;
        MPI_Allreduce(&locMinDet, &minDetGt, 1, MPI_DOUBLE, MPI_MIN,
                      m_uiMesh->getMPICommunicator());
        if (minDetGt == noMinDet)
            minDetGt = std::numeric_limits<double>::quiet_NaN();

        // Shell-only mode computes constraints on a subset of blocks only, so
        // the constraint norms would be meaningless: skip them.
        if (!gw_shell_only)
            bssn::extractConstraints(m_uiMesh, (const DendroScalar**)consVar,
                                     evolVar[BHLOC::EXTRACTION_VAR_ID],
                                     BHLOC::EXTRACTION_TOL,
                                     m_uiTinfo._m_uiStep, m_uiTinfo._m_uiT,
                                     minDetGt);
        double tm4 = tm_on ? tm_now() : 0.0;
#ifndef BSSN_KERR_SCHILD_TEST
#ifdef BSSN_EXTRACT_GRAVITATIONAL_WAVES
        GW::extractFarFieldPsi4(m_uiMesh, (const DendroScalar**)consVar,
                                m_uiTinfo._m_uiStep, m_uiTinfo._m_uiT);
#endif
#endif
        if (tm_on) {
            const double tm5 = tm_now();
            tm_acc[0] += tm1 - tm0;  // unzip + D2H + to_2d
            tm_acc[1] += tm2 - tm1;  // (setup)
            tm_acc[2] += tm3 - tm2;  // (physical_constraints loop precedes tm2? see below)
            tm_acc[3] += tm4 - tm3;  // extractConstraints
            tm_acc[4] += tm5 - tm4;  // extractFarFieldPsi4
            tm_acc[5] += tm5 - tm0;
            if (++tm_n % 5 == 0 && !m_uiMesh->getMPIRankGlobal())
                printf("[extract-time] n=%u total=%.2fs unzip+d2h=%.2f "
                       "constraints_loop+pre=%.2f zip+ghost=%.2f "
                       "extractConstraints=%.2f psi4=%.2f\n",
                       tm_n, tm_acc[5], tm_acc[0], tm_acc[1], tm_acc[2],
                       tm_acc[3], tm_acc[4]);
        }
    }

#endif

#ifdef BSSN_ENABLE_VTU_OUTPUT

    if (bssn::BSSN_IO_OUTPUT_FREQ > 0 &&
        (m_uiTinfo._m_uiStep % bssn::BSSN_IO_OUTPUT_FREQ) == 0) {
        std::vector<std::string> pDataNames;
        const unsigned int numConstVars = bssn::BSSN_NUM_CONST_VARS_VTU_OUTPUT;
        const unsigned int numEvolVars  = bssn::BSSN_NUM_EVOL_VARS_VTU_OUTPUT;

        double* pData[(numConstVars + numEvolVars)];

        for (unsigned int i = 0; i < numEvolVars; i++) {
            pDataNames.push_back(std::string(
                bssn::BSSN_VAR_NAMES[BSSN_VTU_OUTPUT_EVOL_INDICES[i]]));
            pData[i] = evolVar[BSSN_VTU_OUTPUT_EVOL_INDICES[i]];
        }

        for (unsigned int i = 0; i < numConstVars; i++) {
            pDataNames.push_back(
                std::string(bssn::BSSN_CONSTRAINT_VAR_NAMES
                                [BSSN_VTU_OUTPUT_CONST_INDICES[i]]));
            pData[numEvolVars + i] = consVar[BSSN_VTU_OUTPUT_CONST_INDICES[i]];
        }

        std::vector<char*> pDataNames_char;
        pDataNames_char.reserve(pDataNames.size());

        for (unsigned int i = 0; i < pDataNames.size(); i++)
            pDataNames_char.push_back(const_cast<char*>(pDataNames[i].c_str()));

        const char* fDataNames[] = {"Time", "Cycle"};
        const double fData[] = {m_uiTinfo._m_uiT, (double)m_uiTinfo._m_uiStep};

        char fPrefix[256];
        sprintf(fPrefix, "%s_%d", bssn::BSSN_VTU_FILE_PREFIX.c_str(),
                m_uiTinfo._m_uiStep);

        if (bssn::BSSN_VTU_Z_SLICE_ONLY) {
            unsigned int s_val[3]  = {1u << (m_uiMaxDepth - 1),
                                      1u << (m_uiMaxDepth - 1),
                                      1u << (m_uiMaxDepth - 1)};
            unsigned int s_norm[3] = {0, 0, 1};
            io::vtk::mesh2vtu_slice(
                m_uiMesh, s_val, s_norm, fPrefix, 2, fDataNames, fData,
                (numEvolVars + numConstVars), (const char**)&pDataNames_char[0],
                (const double**)pData);
        } else
            io::vtk::mesh2vtuFine(m_uiMesh, fPrefix, 2, fDataNames, fData,
                                  (numEvolVars + numConstVars),
                                  (const char**)&pDataNames_char[0],
                                  (const double**)pData);
    }

#endif

#ifdef BSSN_EXTRACT_BH_LOCATIONS
    if (extract)
        bssn::writeBHCoordinates((const ot::Mesh*)m_uiMesh,
                                 (const Point*)m_uiBHLoc, 2,
                                 m_uiTinfo._m_uiStep, m_uiTinfo._m_uiT);
#endif

    return 0;
}

int BSSNCtxGPU::write_checkpt() {
    if (!m_uiMesh->isActive()) return 0;

    // alternate slots; written after evolve(), so step is one past a multiple
    const unsigned int cpIndex =
        (m_uiTinfo._m_uiStep / bssn::BSSN_CHECKPT_FREQ) % 2;

    const bool is_merged =
        ((bssn::BSSN_BH_LOC[0] - bssn::BSSN_BH_LOC[1]).abs() < 0.1);

    // Idle ranks skip the latch, so ranks that rejoin the active comm after
    // the merger would re-write slot 3 alone and desync the collectives.
    // Agree on both flags first.
    int mergeLocal[2]  = {is_merged ? 1 : 0,
                           bssn::BSSN_MERGED_CHKPT_WRITTEN ? 1 : 0};
    int mergeGlobal[2] = {0, 0};
    par::Mpi_Allreduce(mergeLocal, mergeGlobal, 2, MPI_MAX,
                       m_uiMesh->getMPICommunicator());
    bssn::BSSN_MERGED_CHKPT_WRITTEN = (mergeGlobal[1] != 0);

    // slot 3 is written ALONGSIDE the normal slot, not instead of it
    if (mergeGlobal[0] && !bssn::BSSN_MERGED_CHKPT_WRITTEN) {
        bssn::BSSN_MERGED_CHKPT_WRITTEN = true;
        write_checkpt_to_slot(3);
    }

    int ret       = write_checkpt_to_slot(cpIndex);

    // published last, and never for slot 3, so it always names the newest
    // COMPLETE normal checkpoint; the reduction holds rank 0 until every rank
    // has closed its files
    int retGlobal = 0;
    par::Mpi_Allreduce(&ret, &retGlobal, 1, MPI_MAX,
                       m_uiMesh->getMPICommunicator());
    if (retGlobal == 0)
        chkpt_publish_latest(bssn::BSSN_CHKPT_FILE_PREFIX, cpIndex,
                             m_uiTinfo._m_uiStep, m_uiMesh->getMPIRank());
    else if (!m_uiMesh->getMPIRank())
        std::cout << RED << "[BSSNCtx] checkpoint slot " << cpIndex
                  << " failed to write; .latest still names the previous one"
                  << NRM << std::endl;

    return retGlobal;
}

int BSSNCtxGPU::write_checkpt_to_slot(unsigned int cpIndex) {
    unsigned int rank = m_uiMesh->getMPIRank();
    unsigned int npes = m_uiMesh->getMPICommSize();

    DendroScalar* eVar[BSSN_NUM_VARS];
    DVec& m_evar = m_var[VL::CPU_EV];
    m_evar.to_2d(eVar);

    char fName[256];
    const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin() +
                                     m_uiMesh->getElementLocalBegin()));
    chkpt_fname_oct(fName, sizeof(fName), bssn::BSSN_CHKPT_FILE_PREFIX, cpIndex,
                    rank);
    int status = io::checkpoint::writeOctToFile(
        fName, pNodes, m_uiMesh->getNumLocalMeshElements());

    unsigned int numVars  = bssn::BSSN_NUM_VARS;
    const char** varNames = bssn::BSSN_VAR_NAMES;

    /*for(unsigned int i=0;i<numVars;i++)
    {
        sprintf(fName,"%s_%s_%d_%d.var",fNamePrefix,varNames[i],cpIndex,rank);
        io::checkpoint::writeVecToFile(fName,m_uiMesh,m_uiPrevVar[i]);
    }*/

    chkpt_fname_var(fName, sizeof(fName), bssn::BSSN_CHKPT_FILE_PREFIX, cpIndex,
                    rank);
    status |= io::checkpoint::writeVecToFile(
        fName, m_uiMesh, (const double**)eVar, bssn::BSSN_NUM_VARS);

    // rank 0 writes the global min
    double locMinDet = m_dMinDetGt;
    GPUDevice::device_to_host<double>(&locMinDet, m_dptr_min_det, 1);
    locMinDet       = std::min(locMinDet, m_dMinDetGt);
    double minDetGt = locMinDet;
    MPI_Allreduce(&locMinDet, &minDetGt, 1, MPI_DOUBLE, MPI_MIN,
                  m_uiMesh->getMPICommunicator());

    if (!rank) {
        chkpt_fname_step(fName, sizeof(fName), bssn::BSSN_CHKPT_FILE_PREFIX,
                         cpIndex);
        std::cout << "[BSSNCtx] \t writing checkpoint file : " << fName
                  << std::endl;
        std::ofstream outfile(fName);
        if (!outfile) {
            std::cout << fName << " file open failed " << std::endl;
            status = 1;
        }

        // field list lives in bssnChkptSchema.h -- add new fields there
        ChkptMeta meta;
        meta.tb                 = m_uiTinfo._m_uiTb;
        meta.te                 = m_uiTinfo._m_uiTe;
        meta.t                  = m_uiTinfo._m_uiT;
        meta.th                 = m_uiTinfo._m_uiTh;
        meta.step               = m_uiTinfo._m_uiStep;
        meta.elementOrder       = m_uiElementOrder;
        meta.waveletTol         = bssn::BSSN_WAVELET_TOL;
        meta.loadImbTol         = bssn::BSSN_LOAD_IMB_TOL;
        meta.numVars            = numVars;
        meta.activeCommSz       = m_uiMesh->getMPICommSize();
        meta.bh1[0]             = m_uiBHLoc[0].x();
        meta.bh1[1]             = m_uiBHLoc[0].y();
        meta.bh1[2]             = m_uiBHLoc[0].z();
        meta.bh2[0]             = m_uiBHLoc[1].x();
        meta.bh2[1]             = m_uiBHLoc[1].y();
        meta.bh2[2]             = m_uiBHLoc[1].z();
        meta.bhVelValid         = m_bBHVelValid;
        meta.bhv1[0]            = m_uiBHVel[0].x();
        meta.bhv1[1]            = m_uiBHVel[0].y();
        meta.bhv1[2]            = m_uiBHVel[0].z();
        meta.bhv2[0]            = m_uiBHVel[1].x();
        meta.bhv2[1]            = m_uiBHVel[1].y();
        meta.bhv2[2]            = m_uiBHVel[1].z();
        meta.bhMerged           = m_bIsBHMerged;
        meta.mergeTime          = m_dMergeTime;
        meta.mergeStep          = m_uiMergeStep;
        // must round-trip, or a post-merger restart re-writes slot 3
        meta.mergedChkptWritten = bssn::BSSN_MERGED_CHKPT_WRITTEN;
        meta.minDetGt           = minDetGt;

        json checkPoint;
        chkpt_write_meta(checkPoint, meta);

        // same base91 BH history keys as the CPU checkpoint
        auto [bh1_str, bh2_str, time_str] =
            encode_bh_locs(m_uiBHLocHistory, m_uiBHTimeHistory);
        checkPoint["DENDRO_BSSN_BH_LOC_T"]  = time_str;
        checkPoint["DENDRO_BSSN_BH_LOC_B1"] = bh1_str;
        checkPoint["DENDRO_BSSN_BH_LOC_B2"] = bh2_str;

        outfile << std::setw(4) << checkPoint << std::endl;
        outfile.close();
        if (!outfile) {
            std::cout << fName << " file write failed " << std::endl;
            status = 1;
        }
    }

    return status;
}

int BSSNCtxGPU::restore_checkpt() {
    unsigned int numVars = 0;
    std::vector<ot::TreeNode> octree;
    json checkPoint;

    int rank;
    int npes;
    MPI_Comm comm = m_uiMesh->getMPIGlobalCommunicator();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    unsigned int activeCommSz;

    char fName[256];
    unsigned int restoreStatus = 0;
    unsigned int restoreStatusGlobal =
        0;  // 0 indicates successfully restorable.

    ot::Mesh* newMesh;
    unsigned int restoreStep[2];
    restoreStep[0]                = 0;
    restoreStep[1]                = 0;

    unsigned int restoreFileIndex = 0;

    // explicit slot skips the 0/1 scan; slot 3 is otherwise unreachable
    bool slotDecided              = false;
    bool explicitSlot             = false;
    if (bssn::BSSN_RESTORE_CHECKPT_SLOT >= 0) {
        const unsigned int slot = (unsigned int)bssn::BSSN_RESTORE_CHECKPT_SLOT;
        unsigned int slotExists = 0;

        if (!rank) {
            slotExists = chkpt_resolve_step(fName, sizeof(fName),
                                            bssn::BSSN_CHKPT_FILE_PREFIX, slot)
                             ? 1
                             : 0;
            if (!slotExists)
                std::cout << "BSSN_RESTORE_CHECKPT_SLOT=" << slot
                          << " requested but " << fName
                          << " does not exist; falling back to auto-detect."
                          << std::endl;
        }
        par::Mpi_Bcast(&slotExists, 1, 0, comm);

        if (slotExists) {
            slotDecided      = true;
            explicitSlot     = true;
            restoreFileIndex = slot;
        }
    }

    // sentinel beats step comparison; absent for pre-sentinel checkpoint sets,
    // which fall through to the scan below
    if (!slotDecided) {
        unsigned int latestSlot  = 0;
        unsigned int latestValid = 0;

        if (!rank) {
            latestValid =
                (chkpt_read_latest(bssn::BSSN_CHKPT_FILE_PREFIX, latestSlot) &&
                 chkpt_resolve_step(fName, sizeof(fName),
                                    bssn::BSSN_CHKPT_FILE_PREFIX, latestSlot))
                    ? 1
                    : 0;
        }
        par::Mpi_Bcast(&latestValid, 1, 0, comm);
        par::Mpi_Bcast(&latestSlot, 1, 0, comm);

        if (latestValid) {
            slotDecided      = true;
            restoreFileIndex = latestSlot;
        }
    }

    for (unsigned int cpIndex = 0; !slotDecided && cpIndex < 2; cpIndex++) {
        restoreStatus = 0;

        if (!rank) {
            chkpt_resolve_step(fName, sizeof(fName),
                               bssn::BSSN_CHKPT_FILE_PREFIX, cpIndex);
            std::ifstream infile(fName);
            if (!infile) {
                std::cout << fName << " file open failed " << std::endl;
                restoreStatus = 1;
            }

            if (restoreStatus == 0) {
                infile >> checkPoint;
                // only the step is needed to pick a slot; the true restore
                // below reads everything for the winning slot
                restoreStep[cpIndex] = checkPoint["DENDRO_TS_STEP_CURRENT"];
            }
        }
    }

    // must stay guarded, or an overridden slot is reset to 0 and broadcast
    if (!slotDecided) {
        if (!rank) {
            if (restoreStep[0] < restoreStep[1])
                restoreFileIndex = 1;
            else
                restoreFileIndex = 0;
        }

        par::Mpi_Bcast(&restoreFileIndex, 1, 0, comm);
    }

    // a write that died partway leaves short files; restore the other normal
    // slot rather than abort on them. An explicit slot is restored as asked.
    if (!explicitSlot && restoreFileIndex < 2 &&
        !chkpt_slot_complete(bssn::BSSN_CHKPT_FILE_PREFIX, restoreFileIndex,
                             BSSN_NUM_VARS, comm)) {
        const unsigned int other = 1 - restoreFileIndex;
        if (chkpt_slot_complete(bssn::BSSN_CHKPT_FILE_PREFIX, other,
                                BSSN_NUM_VARS, comm)) {
            if (!rank)
                std::cout << "WARNING: checkpoint slot " << restoreFileIndex
                          << " is incomplete; restoring slot " << other
                          << " instead." << std::endl;
            restoreFileIndex = other;
        }
    }

    restoreStatus = 0;
    octree.clear();
    if (!rank)
        std::cout << "[BSSNCtx] :  Trying to restore from checkpoint index : "
                  << restoreFileIndex << std::endl;

    if (!rank) {
        chkpt_resolve_step(fName, sizeof(fName), bssn::BSSN_CHKPT_FILE_PREFIX,
                           restoreFileIndex);
        std::ifstream infile(fName);
        if (!infile) {
            std::cout << fName << " file open failed " << std::endl;
            restoreStatus = 1;
        }

        if (restoreStatus == 0) {
            infile >> checkPoint;

            // seed with current state so absent optional keys are no-ops
            ChkptMeta meta;
            meta.bhMerged           = m_bIsBHMerged;
            meta.mergeTime          = m_dMergeTime;
            meta.mergeStep          = m_uiMergeStep;
            meta.mergedChkptWritten = bssn::BSSN_MERGED_CHKPT_WRITTEN;
            meta.minDetGt           = m_dMinDetGt;
            chkpt_read_meta(checkPoint, meta);
            m_dMinDetGt             = meta.minDetGt;

            m_uiTinfo._m_uiTb       = meta.tb;
            m_uiTinfo._m_uiTe       = meta.te;
            m_uiTinfo._m_uiT        = meta.t;
            m_uiTinfo._m_uiStep     = meta.step;
            m_uiTinfo._m_uiTh       = meta.th;
            if (meta.elementOrder != bssn::BSSN_ELE_ORDER) {
                if (!rank)
                    std::cout << "[BSSNCtxGPU] : Checkpoint element order "
                              << meta.elementOrder
                              << " differs from BSSN_ELE_ORDER "
                              << bssn::BSSN_ELE_ORDER << std::endl;
                MPI_Abort(comm, 1);
            }
            m_uiElementOrder        = meta.elementOrder;

            bssn::BSSN_WAVELET_TOL  = meta.waveletTol;
            bssn::BSSN_LOAD_IMB_TOL = meta.loadImbTol;

            numVars                 = meta.numVars;
            activeCommSz            = meta.activeCommSz;

            m_uiBHLoc[0] = Point(meta.bh1[0], meta.bh1[1], meta.bh1[2]);
            m_uiBHLoc[1] = Point(meta.bh2[0], meta.bh2[1], meta.bh2[2]);
            m_bBHVelValid = meta.bhVelValid;
            m_uiBHVel[0]  = Point(meta.bhv1[0], meta.bhv1[1], meta.bhv1[2]);
            m_uiBHVel[1]  = Point(meta.bhv2[0], meta.bhv2[1], meta.bhv2[2]);

            if (meta.hasBhMerge) {
                m_bIsBHMerged = meta.bhMerged;
                set_bh_merge_time(meta.mergeTime, meta.mergeStep);
            }

            bssn::BSSN_MERGED_CHKPT_WRITTEN =
                meta.hasMergedLatch ? meta.mergedChkptWritten : m_bIsBHMerged;

            if (checkPoint.contains("DENDRO_BSSN_BH_LOC_T")) {
                auto bh_decoded = decode_bh_locs(
                    checkPoint["DENDRO_BSSN_BH_LOC_B1"].get<std::string>(),
                    checkPoint["DENDRO_BSSN_BH_LOC_B2"].get<std::string>(),
                    checkPoint["DENDRO_BSSN_BH_LOC_T"].get<std::string>());
                m_uiBHLocHistory  = std::get<0>(bh_decoded);
                m_uiBHTimeHistory = std::get<1>(bh_decoded);
            }
        }
    }

    // The .cp is parsed on rank 0 only here (the CPU ctx parses on every
    // rank), so everything it sets has to be published.
    {
        unsigned int mergedLatch =
            (unsigned int)bssn::BSSN_MERGED_CHKPT_WRITTEN;
        par::Mpi_Bcast(&mergedLatch, 1, 0, comm);
        bssn::BSSN_MERGED_CHKPT_WRITTEN = (mergedLatch != 0);

        unsigned int isMerged           = (unsigned int)m_bIsBHMerged;
        double mergeTime                = m_dMergeTime;
        unsigned int mergeStep          = m_uiMergeStep;
        par::Mpi_Bcast(&isMerged, 1, 0, comm);
        par::Mpi_Bcast(&mergeTime, 1, 0, comm);
        par::Mpi_Bcast(&mergeStep, 1, 0, comm);
        if (rank) {
            m_bIsBHMerged = (isMerged != 0);
            set_bh_merge_time(mergeTime, mergeStep);
        }

        // every rank evaluates isRemeshBH and evolve_bh_loc from the history
        std::string blobs[3];
        if (!rank)
            std::tie(blobs[0], blobs[1], blobs[2]) =
                encode_bh_locs(m_uiBHLocHistory, m_uiBHTimeHistory);
        for (auto& blob : blobs) {
            unsigned int blobLen = (unsigned int)blob.size();
            par::Mpi_Bcast(&blobLen, 1, 0, comm);
            blob.resize(blobLen);
            if (blobLen) MPI_Bcast(&blob[0], (int)blobLen, MPI_CHAR, 0, comm);
        }
        if (rank) {
            auto bh_decoded   = decode_bh_locs(blobs[0], blobs[1], blobs[2]);
            m_uiBHLocHistory  = std::get<0>(bh_decoded);
            m_uiBHTimeHistory = std::get<1>(bh_decoded);
        }
    }

    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout
                << "[BSSNCtx] : Restore step failed, restore file corrupted. "
                << std::endl;
        MPI_Abort(comm, 1);
    }

    MPI_Bcast(&m_uiTinfo, sizeof(ts::TSInfo), MPI_BYTE, 0, comm);
    par::Mpi_Bcast(&bssn::BSSN_WAVELET_TOL, 1, 0, comm);
    par::Mpi_Bcast(&bssn::BSSN_LOAD_IMB_TOL, 1, 0, comm);

    par::Mpi_Bcast(&numVars, 1, 0, comm);
    par::Mpi_Bcast(&m_uiElementOrder, 1, 0, comm);
    par::Mpi_Bcast(&activeCommSz, 1, 0, comm);

    par::Mpi_Bcast(m_uiBHLoc, 2, 0, comm);
    par::Mpi_Bcast(m_uiBHVel, 2, 0, comm);
    MPI_Bcast(&m_bBHVelValid, sizeof(bool), MPI_BYTE, 0, comm);
    bssn::BSSN_BH_LOC[0] = m_uiBHLoc[0];
    bssn::BSSN_BH_LOC[1] = m_uiBHLoc[1];

    if (activeCommSz > npes) {
        if (!rank)
            std::cout
                << " [BSSNCtx] : checkpoint file written from  a larger "
                   "communicator than the current global comm. (i.e. "
                   "communicator shrinking not allowed in the restore step. )"
                << std::endl;

        MPI_Abort(comm, 1);
    }

    bool isActive = (rank < activeCommSz);

    MPI_Comm newComm;
    par::splitComm2way(isActive, &newComm, comm);

    if (isActive) {
        int activeRank;
        int activeNpes;

        MPI_Comm_rank(newComm, &activeRank);
        MPI_Comm_size(newComm, &activeNpes);
        assert(activeNpes == activeCommSz);

        chkpt_resolve_oct(fName, sizeof(fName), bssn::BSSN_CHKPT_FILE_PREFIX,
                          restoreFileIndex, activeRank);
        restoreStatus = io::checkpoint::readOctFromFile(fName, octree);
        assert(par::test::isUniqueAndSorted(octree, newComm));
    }

    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout << "[BSSNCtx]: octree (*.oct) restore file is corrupted "
                      << std::endl;
        MPI_Abort(comm, 1);
    }

    newMesh = new ot::Mesh(octree, 1, m_uiElementOrder, activeCommSz, comm);
    newMesh->setDomainBounds(Point(bssn::BSSN_GRID_MIN_X, bssn::BSSN_GRID_MIN_Y,
                                   bssn::BSSN_GRID_MIN_Z),
                             Point(bssn::BSSN_GRID_MAX_X, bssn::BSSN_GRID_MAX_Y,
                                   bssn::BSSN_GRID_MAX_Z));
// no need to transfer data only to resize the contex variables.
// this->grid_transfer(newMesh);
#ifdef __CUDACC__
    device::MeshGPU*& dptr_mesh = this->get_meshgpu_device_ptr();
    device::MeshGPU* mesh_gpu   = this->get_meshgpu_host_handle();

    mesh_gpu->dealloc_mesh_on_device(dptr_mesh);
    dptr_mesh = mesh_gpu->alloc_mesh_on_device(newMesh);
#endif

    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, BSSN_NUM_VARS,
                                      BSSN_ASYNC_COMM_K);
    device::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                          BSSN_NUM_VARS, BSSN_ASYNC_COMM_K);

    ot::alloc_mpi_ctx<DendroScalar>(newMesh, m_mpi_ctx, BSSN_NUM_VARS,
                                    BSSN_ASYNC_COMM_K);
    device::alloc_mpi_ctx<DendroScalar>(newMesh, m_mpi_ctx_device,
                                        BSSN_NUM_VARS, BSSN_ASYNC_COMM_K);

    // variable allocation for evolution variables
    m_var[VL::CPU_EV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, BSSN_NUM_VARS, true);
    m_var[VL::CPU_CV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST,
                                    BSSN_CONSTRAINT_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        BSSN_NUM_VARS, true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        BSSN_CONSTRAINT_NUM_VARS, true);

    m_var[VL::GPU_EV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::DEVICE, BSSN_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        BSSN_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_OUT].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        BSSN_NUM_VARS, true);

    // only reads the evolution variables.
    if (isActive) {
        int activeRank;
        int activeNpes;

        DendroScalar* inVec[BSSN_NUM_VARS];
        DVec& m_evar = m_var[VL::CPU_EV];
        m_evar.to_2d(inVec);

        MPI_Comm_rank(newComm, &activeRank);
        MPI_Comm_size(newComm, &activeNpes);
        assert(activeNpes == activeCommSz);

        chkpt_fname_var(fName, sizeof(fName), bssn::BSSN_CHKPT_FILE_PREFIX,
                        restoreFileIndex, activeRank);
        restoreStatus = io::checkpoint::readVecFromFile(fName, newMesh, inVec,
                                                        bssn::BSSN_NUM_VARS);
    }

    MPI_Comm_free(&newComm);
    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout << "[BSSNCtx]: varible (*.var) restore file currupted "
                      << std::endl;
        MPI_Abort(comm, 1);
    }

    std::swap(m_uiMesh, newMesh);
    delete newMesh;

    // realloc bssn deriv space
    deallocate_bssn_deriv_workspace();
    allocate_bssn_deriv_workspace(m_uiMesh, 1);

    // initialize() returns right after the restore, so init_grid() and the
    // converge block never set these on this path
    set_appropriate_derivs(bssn::BSSN_PADDING_WIDTH);

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);
    bssn::BSSN_CURRENT_MIN_DX =
        (bssn::min_domain_width() *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)bssn::BSSN_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));

    unsigned int localSz    = m_uiMesh->getNumLocalMeshElements();
    unsigned int totalElems = 0;
    par::Mpi_Allreduce(&localSz, &totalElems, 1, MPI_SUM, comm);

    if (!rank)
        std::cout << " checkpoint at step : " << m_uiTinfo._m_uiStep
                  << "active Comm. sz: " << activeCommSz
                  << " restore successful: "
                  << " restored mesh size: " << totalElems << std::endl;

    m_uiIsETSSynced = false;
    return 0;
}

int BSSNCtxGPU::rhs(DVec* in, DVec* out, unsigned int sz, DendroScalar time) {
    if (!m_uiMesh->isActive()) return 0;

    const unsigned int pencils      = 13;
    const unsigned int pen_sz       = 13;

    DEVICE_REAL* const out_ptr      = out->get_vec_ptr();
    const DEVICE_REAL* const in_ptr = in->get_vec_ptr();

    DVec& m_dptr_uz_i               = m_var[VL::GPU_EV_UZ_IN];
    DVec& m_dptr_uz_o               = m_var[VL::GPU_EV_UZ_OUT];

    const unsigned int unzip_sz     = m_uiMesh->getDegOfFreedomUnZip();

    this->unzip(*in, m_dptr_uz_i, BSSN_ASYNC_COMM_K);

    unsigned int nblocks = m_uiMesh->getLocalBlockList().size();

    // Tier 3.1 #3: one-shot dump of an interior unzipped gauge block for TL vs FD.
    // Env: DENDRO_DUMP_TL_GAUGE=<path.bin>
    // Note: m_mesh_cpu.m_blk_list is device memory — use host ot::Block list.
    if (const char* dump_path = std::getenv("DENDRO_DUMP_TL_GAUGE")) {
        static bool dumped = false;
        const std::vector<ot::Block>& blkList = m_uiMesh->getLocalBlockList();
        if (!dumped && !blkList.empty()) {
            dumped = true;
            int bid = -1;
            for (unsigned int b = 0; b < blkList.size(); ++b) {
                if (blkList[b].getBlkNodeFlag() == 0) {
                    bid = static_cast<int>(b);
                    break;
                }
            }
            if (bid < 0) bid = 0;
            const ot::Block& blk = blkList[static_cast<unsigned int>(bid)];
            const unsigned int nx = blk.getAllocationSzX();
            const unsigned int ny = blk.getAllocationSzY();
            const unsigned int nz = blk.getAllocationSzZ();
            const unsigned int n  = nx * ny * nz;
            const unsigned int off = blk.getOffset();
            const Point pt_min(bssn::BSSN_COMPD_MIN[0], bssn::BSSN_COMPD_MIN[1],
                               bssn::BSSN_COMPD_MIN[2]);
            const Point pt_max(bssn::BSSN_COMPD_MAX[0], bssn::BSSN_COMPD_MAX[1],
                               bssn::BSSN_COMPD_MAX[2]);
            const double dx = blk.computeDx(pt_min, pt_max);
            std::vector<double> h_uz(5u * n);
            GPUDevice::device_synchronize();
            auto copy_var = [&](unsigned int var, unsigned int slot) {
                GPUDevice::device_to_host(
                    h_uz.data() + slot * n,
                    m_dptr_uz_i.get_vec_ptr() + var * unzip_sz + off, n);
            };
            copy_var(bssn::VAR::U_ALPHA, 0);
            copy_var(bssn::VAR::U_K, 1);
            copy_var(bssn::VAR::U_BETA0, 2);
            copy_var(bssn::VAR::U_BETA1, 3);
            copy_var(bssn::VAR::U_BETA2, 4);
            FILE* f = std::fopen(dump_path, "wb");
            if (f) {
                const char magic[4] = {'T', 'L', 'G', '1'};
                const double ko = KO_DISS_SIGMA;
                const double lam0 = static_cast<double>(BSSN_LAMBDA[0]);
                const unsigned int bflag = blk.getBlkNodeFlag();
                const unsigned int ubid = static_cast<unsigned int>(bid);
                std::fwrite(magic, 1, 4, f);
                std::fwrite(&nx, sizeof(nx), 1, f);
                std::fwrite(&ny, sizeof(ny), 1, f);
                std::fwrite(&nz, sizeof(nz), 1, f);
                std::fwrite(&dx, sizeof(dx), 1, f);
                std::fwrite(&ko, sizeof(ko), 1, f);
                std::fwrite(&lam0, sizeof(lam0), 1, f);
                std::fwrite(&ubid, sizeof(ubid), 1, f);
                std::fwrite(&bflag, sizeof(bflag), 1, f);
                std::fwrite(h_uz.data(), sizeof(double), 5u * n, f);
                std::fclose(f);
                std::printf(
                    "DENDRO_DUMP_TL_GAUGE wrote %s blk=%u bflag=%u nx=%u "
                    "dx=%.6e ko=%.3g lam0=%.3g\n",
                    dump_path, ubid, bflag, nx, dx, ko, lam0);
            } else {
                std::printf("DENDRO_DUMP_TL_GAUGE open failed: %s\n",
                            dump_path);
            }
        }
    }

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::RHS].start();
#endif
    // device::bssnrhs2<3, pen_sz, pen_sz,
    // DEVICE_RHS_BATCHED_GRAIN_SZ,DEVICE_RHS_NSTREAMS>
    // (m_dptr_uz_o.get_vec_ptr(),m_dptr_uz_i.get_vec_ptr(),m_mesh_cpu.m_blk_list,
    // nblocks, m_dptr_deriv_evars, unzip_sz, KO_DISS_SIGMA);
    // DENDRO_RHS_PT_F32SU=1 → PT fused on float shared tile (best candidate).
    // DENDRO_RHS_PT_GMEM=1 → point FD from global (0 smem, higher occ).
    // DENDRO_RHS_PT_DB=1 → PT + cp.async double-buffer.
    // DENDRO_RHS_PT_ASYNC=1 → PT + cp.async load (no DB overlap).
    // DENDRO_RHS_PT_ASYNC_DEFER=1 → PT_ASYNC + deferred DD seconds (B1).
    // DENDRO_RHS_PT_ASYNC_CSE=1 → PT_ASYNC + christoffel CSE (B1 follow-on).
// DENDRO_RHS_PT_ASYNC_WHILEHOT=1 → early RIJ + scoped gt/chi seconds.
    // DENDRO_RHS_PT=1 → point-wise fused FD.
    // Hybrid: interior blocks use PT*; domain-boundary (bflag!=0) use stock
    // one-sided FD so C_HAM matches (PT alone diverges on faces).
    // DENDRO_RHS_F32=1 → FP32 Du/DDu shared scratch (FP64 state/algebra).
    if (std::getenv("DENDRO_RHS_PT_F32SU")) {
        device::bssnrhs3_pt_f32su<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                                  DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT_GMEM")) {
        device::bssnrhs3_pt_gmem<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                                 DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT_DB")) {
        device::bssnrhs3_pt_db<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                               DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT_ASYNC_DEFER")) {
        device::bssnrhs3_pt_async_defer<3, pen_sz, pen_sz,
                                        DEVICE_RHS_BATCHED_GRAIN_SZ,
                                        DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT_ASYNC_CSE")) {
        device::bssnrhs3_pt_async_cse<3, pen_sz, pen_sz,
                                      DEVICE_RHS_BATCHED_GRAIN_SZ,
                                      DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT_ASYNC_WHILEHOT")) {
        device::bssnrhs3_pt_async_whilehot<3, pen_sz, pen_sz,
                                           DEVICE_RHS_BATCHED_GRAIN_SZ,
                                           DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT_ASYNC")) {
        device::bssnrhs3_pt_async<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                                  DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT")) {
        device::bssnrhs3_pt<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                            DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_F32C")) {
        // FP32 centered scratch (u-uc); expect ~rounding-level vs stock.
        device::bssnrhs3_f32c<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                              DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_F32")) {
        device::bssnrhs3_f32<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                             DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else if (std::getenv("DENDRO_RHS_PT_KOF32")) {
        device::bssnrhs3_pt_kof32<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                                  DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    } else {
        device::bssnrhs3<3, pen_sz, pen_sz, DEVICE_RHS_BATCHED_GRAIN_SZ,
                         DEVICE_RHS_NSTREAMS>(
            m_dptr_uz_o.get_vec_ptr(), m_dptr_uz_i.get_vec_ptr(),
            m_mesh_cpu.m_blk_list, nblocks, unzip_sz, KO_DISS_SIGMA);
    }
#ifdef __PROFILE_CTX__
    GPUDevice::device_synchronize();
    m_uiCtxpt[ts::CTXPROFILE::RHS].stop();
#endif

    GPUDevice::check_last_error();

    this->zip(m_dptr_uz_o, *out);

    return 0;
}

int BSSNCtxGPU::post_stage(DVec& sIn) {
    // const unsigned int lb = m_uiMesh->getNodeLocalBegin();
    // const unsigned int le = m_uiMesh->getNodeLocalEnd();
    // const unsigned int szpdof = sIn.get_size()/sIn.get_dof();
    // device::cuda_bssn_enforce_evar_cons<<< (le-lb)/1024 + 1,
    // 1024>>>(sIn.get_vec_ptr(),lb,le,bssn::CHI_FLOOR,szpdof);
    // GPUDevice::device_synchronize();
    // GPUDevice::check_last_error();

    return 0;
}

int BSSNCtxGPU::post_timestep(DVec& sIn) {
    // need to enforce constraints in cuda.
    const unsigned int lb     = m_uiMesh->getNodeLocalBegin();
    const unsigned int le     = m_uiMesh->getNodeLocalEnd();
    const unsigned int szpdof = sIn.get_size() / sIn.get_dof();
    const unsigned int nblk   = (le > lb) ? ((le - lb) / 1024 + 1) : 1;
    if (nblk > m_uiBlockMinsCap) {
        if (m_dptr_block_mins)
            GPUDevice::device_free<double>(m_dptr_block_mins);
        m_uiBlockMinsCap  = nblk;
        m_dptr_block_mins = GPUDevice::device_malloc<double>(m_uiBlockMinsCap);
    }
    device::cuda_bssn_enforce_evar_cons<<<nblk, 1024>>>(
        sIn.get_vec_ptr(), lb, le, bssn::CHI_FLOOR, szpdof, m_dptr_block_mins);
    GPUDevice::device_synchronize();
    GPUDevice::check_last_error();

    // Ordered fold: ascending block index → device running min (bit-stable).
    std::vector<double> hmins(nblk);
    GPUDevice::device_to_host<double>(hmins.data(), m_dptr_block_mins, nblk);
    double step_min = std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < nblk; ++i)
        step_min = std::min(step_min, hmins[i]);
    double cur = std::numeric_limits<double>::max();
    GPUDevice::device_to_host<double>(&cur, m_dptr_min_det, 1);
    cur = std::min(cur, step_min);
    GPUDevice::host_to_device<double>(&cur, m_dptr_min_det, 1);

    return 0;
}

bool BSSNCtxGPU::is_remesh() {
    bool isRefine = false;
    if (bssn::BSSN_ENABLE_BLOCK_ADAPTIVITY) return false;

    MPI_Comm comm    = m_uiMesh->getMPIGlobalCommunicator();

    DVec& m_evar     = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ_IN];

    static const bool tm_rm = std::getenv("DENDRO_TIME_REMESH") != nullptr;
    static double tm_rm_acc[3] = {0, 0, 0};
    static unsigned tm_rm_n    = 0;
    auto tmn = []() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    const double trm0 = tm_rm ? tmn() : 0.0;
    this->unzip(m_evar, m_evar_unz, bssn::BSSN_ASYNC_COMM_K);

    DendroScalar* unzipVar[BSSN_NUM_VARS];
    m_evar_unz.to_2d(unzipVar);
    const double trm1 = tm_rm ? tmn() : 0.0;

    unsigned int refineVarIds[bssn::BSSN_NUM_REFINE_VARS];
    for (unsigned int vIndex = 0; vIndex < bssn::BSSN_NUM_REFINE_VARS; vIndex++)
        refineVarIds[vIndex] = bssn::BSSN_REFINE_VARIABLE_INDICES[vIndex];

    double wTol = bssn::BSSN_WAVELET_TOL;
    std::function<double(double, double, double, double* hx)> waveletTolFunc =
        [](double x, double y, double z, double* hx) {
            return bssn::computeWTolDCoords(x, y, z, hx);
        };

    if (bssn::BSSN_REFINEMENT_MODE == bssn::RefinementMode::WAMR) {
        isRefine =
            bssn::isReMeshWAMR(m_uiMesh, (const double**)unzipVar, refineVarIds,
                               bssn::BSSN_NUM_REFINE_VARS, waveletTolFunc,
                               bssn::BSSN_DENDRO_AMR_FAC);
        if (tm_rm) {
            const double trm2 = tmn();
            tm_rm_acc[0] += trm1 - trm0;
            tm_rm_acc[1] += trm2 - trm1;
            tm_rm_acc[2] += trm2 - trm0;
            if (++tm_rm_n % 3 == 0 && !m_uiMesh->getMPIRankGlobal())
                printf("[remesh-time] n=%u total=%.2fs host_unzip=%.2f wamr=%.2f "
                       "(nVars=%u)\n",
                       tm_rm_n, tm_rm_acc[2], tm_rm_acc[0], tm_rm_acc[1],
                       (unsigned)bssn::BSSN_NUM_REFINE_VARS);
        }

    } else if (bssn::BSSN_REFINEMENT_MODE == bssn::RefinementMode::EH) {
        isRefine = bssn::isRemeshEH(
            m_uiMesh, (const double**)unzipVar, bssn::VAR::U_ALPHA,
            bssn::BSSN_EH_REFINE_VAL, bssn::BSSN_EH_COARSEN_VAL, true);

    } else if (bssn::BSSN_REFINEMENT_MODE == bssn::RefinementMode::EH_WAMR) {
        const bool isR1 =
            bssn::isReMeshWAMR(m_uiMesh, (const double**)unzipVar, refineVarIds,
                               bssn::BSSN_NUM_REFINE_VARS, waveletTolFunc,
                               bssn::BSSN_DENDRO_AMR_FAC);
        const bool isR2 = bssn::isRemeshEH(
            m_uiMesh, (const double**)unzipVar, bssn::VAR::U_ALPHA,
            bssn::BSSN_EH_REFINE_VAL, bssn::BSSN_EH_COARSEN_VAL, false);

        isRefine = (isR1 || isR2);
    } else if (bssn::BSSN_REFINEMENT_MODE == bssn::RefinementMode::BH_LOC) {
        isRefine =
            bssn::isRemeshBH(m_uiMesh, m_uiBHLoc, this->get_bh_loc_history(),
                             this->get_bh_loc_time_history());
    } else if (bssn::BSSN_REFINEMENT_MODE == bssn::RefinementMode::BH_WAMR) {
        // The CPU driver (BSSNCtx::is_remesh) implements BH_WAMR; the GPU
        // context did not, so GPU runs with the production parameter files
        // (BSSN_REFINEMENT_MODE = 4) never remeshed. Mirrored here, gated by
        // DENDRO_GPU_BH_WAMR=1 (default OFF, i.e. historical behaviour).
        static const bool gpu_bh_wamr = []() {
            const char* e = std::getenv("DENDRO_GPU_BH_WAMR");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        if (gpu_bh_wamr) {
            const bool isR1 = bssn::isRemeshBH(m_uiMesh, m_uiBHLoc,
                                               this->get_bh_loc_history(),
                                               this->get_bh_loc_time_history());
            const bool isR2 = bssn::addRemeshWAMR(
                m_uiMesh, (const double**)unzipVar, refineVarIds,
                bssn::BSSN_NUM_REFINE_VARS, waveletTolFunc,
                bssn::BSSN_DENDRO_AMR_FAC);
            isRefine = (isR1 || isR2);
        }
    }
    if (tm_rm && bssn::BSSN_REFINEMENT_MODE != bssn::RefinementMode::WAMR) {
        const double trm2 = tmn();
        tm_rm_acc[0] += trm1 - trm0;
        tm_rm_acc[1] += trm2 - trm1;
        tm_rm_acc[2] += trm2 - trm0;
        if (++tm_rm_n % 3 == 0 && !m_uiMesh->getMPIRankGlobal())
            printf("[remesh-time] n=%u total=%.2fs host_unzip=%.2f test=%.2f\n",
                   tm_rm_n, tm_rm_acc[2], tm_rm_acc[0], tm_rm_acc[1]);
    }

    return isRefine;
}

DVec& BSSNCtxGPU::get_evolution_vars() { return m_var[VL::GPU_EV]; }

DVec& BSSNCtxGPU::get_evolution_vars_cpu() { return m_var[VL::CPU_EV]; }

DVec& BSSNCtxGPU::get_constraint_vars() { return m_var[VL::CPU_CV]; }

// DVec& BSSNCtxGPU::get_primitive_vars()
// {
//     return m_pvar;
// }

int BSSNCtxGPU::terminal_output() {
    if (m_uiMesh->isActive()) {
        DendroScalar min = 0, max = 0;
        DVec& m_evar = m_var[VL::CPU_EV];
        min = vecMin(m_uiMesh, m_evar.get_vec_ptr(), ot::VEC_TYPE::CG_NODAL,
                     true);
        max = vecMax(m_uiMesh, m_evar.get_vec_ptr(), ot::VEC_TYPE::CG_NODAL,
                     true);

        if (!(m_uiMesh->getMPIRank())) {
            std::cout << "[BSSNCtx]:  "
                      << bssn::BSSN_VAR_NAMES[bssn::VAR::U_ALPHA]
                      << " (min,max) : \t ( " << min << ", " << max << " ) "
                      << std::endl;
            if (std::isnan(min) || std::isnan(max)) {
                std::cout << "[Error]: NAN detected " << std::endl;
                MPI_Abort(m_uiMesh->getMPICommunicator(), 1);
            }
        }
    }

    return 0;
}

int BSSNCtxGPU::replace_mesh_and_transfer(ot::Mesh* newMesh) {
    if (!newMesh) return 1;
    DendroIntL oldElements = m_uiMesh->getNumLocalMeshElements();
    DendroIntL newElements = newMesh->getNumLocalMeshElements();
    DendroIntL oldElements_g = 0, newElements_g = 0;
    par::Mpi_Reduce(&oldElements, &oldElements_g, 1, MPI_SUM, 0,
                    m_uiMesh->getMPIGlobalCommunicator());
    par::Mpi_Reduce(&newElements, &newElements_g, 1, MPI_SUM, 0,
                    newMesh->getMPIGlobalCommunicator());
    if (!m_uiMesh->getMPIRankGlobal())
        std::cout << "[mesh_replay] adopt mesh | old=" << oldElements_g
                  << " new=" << newElements_g << std::endl;

    this->grid_transfer(newMesh);
    std::swap(newMesh, m_uiMesh);
    delete newMesh;
    m_uiIsETSSynced = false;
    return 0;
}

int BSSNCtxGPU::grid_transfer(const ot::Mesh* m_new) {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].start();
#endif
    DVec& m_evar = m_var[VL::CPU_EV];
    DVec::grid_transfer(m_uiMesh, m_new, m_evar);
    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, BSSN_NUM_VARS,
                                      BSSN_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_new, m_mpi_ctx, BSSN_NUM_VARS,
                                    BSSN_ASYNC_COMM_K);

    device::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx_device,
                                          BSSN_NUM_VARS, BSSN_ASYNC_COMM_K);
    device::alloc_mpi_ctx<DendroScalar>(m_new, m_mpi_ctx_device, BSSN_NUM_VARS,
                                        BSSN_ASYNC_COMM_K);
    // printf("igt ended\n");

    m_var[VL::CPU_CV].destroy_vector();
    m_var[VL::CPU_CV_UZ_IN].destroy_vector();
    m_var[VL::CPU_EV_UZ_IN].destroy_vector();

    m_var[VL::GPU_EV].destroy_vector();
    m_var[VL::GPU_EV_UZ_IN].destroy_vector();
    m_var[VL::GPU_EV_UZ_OUT].destroy_vector();

    m_var[VL::CPU_CV].create_vector(m_new, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST,
                                    BSSN_CONSTRAINT_NUM_VARS, true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        BSSN_CONSTRAINT_NUM_VARS, true);

    m_var[VL::CPU_EV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        BSSN_NUM_VARS, true);

    m_var[VL::GPU_EV].create_vector(m_new, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::DEVICE, BSSN_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        BSSN_NUM_VARS, true);
    m_var[VL::GPU_EV_UZ_OUT].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::DEVICE,
        BSSN_NUM_VARS, true);

#ifdef __CUDACC__
    // the device keeps its own mesh copy; rebuild it for m_new before pushing
    {
        device::MeshGPU*& dptr_mesh = this->get_meshgpu_device_ptr();
        device::MeshGPU* mesh_gpu   = this->get_meshgpu_host_handle();

        mesh_gpu->dealloc_mesh_on_device(dptr_mesh);
        dptr_mesh = mesh_gpu->alloc_mesh_on_device(m_new);
    }
#endif

    this->host_to_device_sync();
    m_uiIsETSSynced = false;

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].stop();
#endif
    return 0;
}

void BSSNCtxGPU::evolve_bh_loc() {
    dendro::logger::debug("Now evolving BH locations");
#ifdef BSSN_EXTRACT_BH_LOCATIONS
    // early exit if we're at time step zero!
    if (m_bBHEvolved) return;

    // step zero doesn't move the BHs, it only seeds the velocity and history
    // this->compute_constraint_variables();

    // compute how long it's been since the last time we calculatd it, thanks to
    // storing history!
    const double dt = this->m_uiTinfo._m_uiStep == 0
                          ? 0.0
                          : m_uiTinfo._m_uiT - m_uiBHTimeHistory.back();
    DVec sIn        = this->get_evolution_vars_cpu();

    Point bhLoc[2];
    DendroScalar* evar[bssn::BSSN_NUM_VARS];
    sIn.to_2d(evar);

    // interpolateToCoords reads ghosts the device RK update never writes
    if (m_uiMesh->isActive()) {
        m_uiMesh->readFromGhostBegin(evar[VAR::U_BETA0], 3);
        m_uiMesh->readFromGhostEnd(evar[VAR::U_BETA0], 3);
    }

    bssn::computeBHLocations((const ot::Mesh*)m_uiMesh, m_uiBHLoc, bhLoc, evar,
                             dt, m_uiBHVel, m_bBHVelValid);
    // if(!m_uiMesh->getMPIRankGlobal())
    // {
    //     std::cout<<"bh0 "<<bhLoc[0]<<std::endl;
    //     std::cout<<"bh1 "<<bhLoc[1]<<std::endl;

    // }
    m_uiBHLoc[0]         = bhLoc[0];
    m_uiBHLoc[1]         = bhLoc[1];

    bssn::BSSN_BH_LOC[0] = m_uiBHLoc[0];
    bssn::BSSN_BH_LOC[1] = m_uiBHLoc[1];

    // append the bssn location points to our history
    this->store_bh_loc_history();

// old bh location extractor.
#if 0
    DendroScalar *evar[bssn::BSSN_NUM_VARS];
    Point bhLoc[2];
    sIn.to_2d(evar);
    bssn::extractBHCoords((const ot::Mesh *)m_uiMesh,(const DendroScalar*)evar[BHLOC::EXTRACTION_VAR_ID],BHLOC::EXTRACTION_TOL,(const Point *) m_uiBHLoc,2,(Point*)bhLoc);
    
    m_uiBHLoc[0] = bhLoc[0];
    m_uiBHLoc[1] = bhLoc[1];
#endif
#endif

    // make sure we know that it's been evolved this time step if we're wanting
    // to evolve it
    m_bBHEvolved = true;

    dendro::logger::info("BH locations evolved!");

    return;
}

void BSSNCtxGPU::store_bh_loc_history() {
    // simple call that stores bh loc history based on bssn::BSSN_BH_LOC
    m_uiBHLocHistory.push_back(
        std::make_pair(bssn::BSSN_BH_LOC[0], bssn::BSSN_BH_LOC[1]));
    m_uiBHTimeHistory.push_back(m_uiTinfo._m_uiT);
}

unsigned int BSSNCtxGPU::getBlkTimestepFac(unsigned int blev, unsigned int lmin,
                                           unsigned int lmax) {
    // Same policy as CPU BSSNCtx (ENUTS / LTS). Declared for Ctx CRTP; not
    // used by the ETS GPU driver today.
    (void)lmin;
    const unsigned int ldiff = BSSN_LTS_TS_OFFSET;
    if ((lmax - blev) <= ldiff) return 1;
    return 1u << (lmax - blev - ldiff);
}

}  // namespace bssn
