// B1: exact-discrete-derivative reference for the BSSN RHS by truncated Taylor arithmetic.
//
// Strategy: the production RHS on a padded block = (linear stencils: first/second derivatives, KO) o (pointwise algebra).
// The stencils are linear => they act coefficient by coefficient on the Taylor coefficient fields (production routines
// deriv_x/..., ko_deriv_x/... reused unchanged via the production include fragments). The pointwise algebra is evaluated
// with the truncated-Taylor scalar type Tay<K> (generated from the production equation file by gen_taylor_eqs.py).
// Validation: (A) coefficient 0 reproduces the production bssnrhs() on the same data; (B) Taylor remainder test:
// |R(u(eps)) - sum_{j<=m} R_j eps^j| scales like eps^(m+1), with R(u(eps)) evaluated by the PRODUCTION double code.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

#include "derivs.h"
#include "parameters.h"
#include "rhs.h"
#include "gr.h"
#include "taylor.hpp"
#include "taylor_names.inc"
#include "taylor_enforce.hpp"
#include "bssn_constraints.h"
#include <mpi.h>

using namespace bssn;
#define KMAX 3
typedef Tay<KMAX> T;

static const int N = 13, PWD = 3, NP = N * N * N, NV = 24;

// coefficient tables (pointers to the K+1 coefficient arrays of every field / derivative array)
#define DECL(n, v) static const double* n##_c[KMAX + 1];
TAYLOR_FIELD_NAMES(DECL)
#undef DECL
#define DECL(n) static const double* n##_c[KMAX + 1];
TAYLOR_DERIV_NAMES(DECL)
#undef DECL
static std::map<std::string, std::vector<const double*>> ko_c;  // name -> per-coefficient pointer

static const double* bcg[KMAX + 1][24][3];
static std::vector<double> workspace[KMAX + 1], ko_workspace[KMAX + 1];

// fill coefficient k: derive everything from the evolved fields f[VAR] (arrays of NP doubles)
static void setup_coeff(int k, const double* const* f, double hx, double hy, double hz, unsigned bflag) {
    unsigned sz[3] = {N, N, N};
    const unsigned BLK_SZ = NP;
    workspace[k].assign((size_t)138 * NP, 0.0);
    double* deriv_base = workspace[k].data();
#define BIND(n, v) const double* const n = f[VAR::v];
    TAYLOR_FIELD_NAMES(BIND)
#undef BIND
    {
#include "bssnrhs_evar_derivs.h"
#include "bssnrhs_derivs.h"
#define REG(n) n##_c[k] = n;
        TAYLOR_DERIV_NAMES(REG)
#undef REG
#define REGG(n, v) bcg[k][VAR::v][0] = grad_0_##n; bcg[k][VAR::v][1] = grad_1_##n; bcg[k][VAR::v][2] = grad_2_##n;
        TAYLOR_FIELD_NAMES(REGG)
#undef REGG
    }
#define REG(n, v) n##_c[k] = n;
    TAYLOR_FIELD_NAMES(REG)
#undef REG
    // KO derivatives into a separate workspace (same names as in the production fragment)
    ko_workspace[k].assign((size_t)72 * NP, 0.0);
    double* kob = ko_workspace[k].data();
    int idx = 0;
#define KODECL(a, v) double* a = kob + (idx++) * (size_t)NP;
    TAYLOR_KO_NAMES(KODECL)
#undef KODECL
#include "bssnrhs_ko_derivs.h"
#define KOREG(a, v) ko_c[#a].resize(KMAX + 1); ko_c[#a][k] = a;
    TAYLOR_KO_NAMES(KOREG)
#undef KOREG
}

// ----------------------------------------------------------------------------------------------------------
static const double PMIN[3] = {-0.6 + 0.35, -0.6 - 0.2, -0.6 + 0.1};
static const double H = 0.1;
typedef std::vector<double> Vec;
struct JetState { Vec c[KMAX + 1][NV]; };

static double bc_asym(int v) { return (v == VAR::U_ALPHA || v == VAR::U_CHI || v == VAR::U_SYMGT0 || v == VAR::U_SYMGT3 || v == VAR::U_SYMGT5) ? 1.0 : 0.0; }
static double bc_fall(int v) { return ((v >= VAR::U_SYMAT0 && v <= VAR::U_SYMAT5) || (v >= VAR::U_GT0 && v <= VAR::U_GT2)) ? 2.0 : 1.0; }

// RHS jets of a jet state: pointwise Taylor algebra, Sommerfeld boundary overwrite (per coefficient), KO added last
static void jet_rhs(const JetState& S, unsigned bflag, JetState& R) {
    const double* fj[KMAX + 1][NV];
    for (int j = 0; j <= KMAX; j++) for (int v = 0; v < NV; v++) fj[j][v] = S.c[j][v].data();
    for (int j = 0; j <= KMAX; j++) setup_coeff(j, fj[j], H, H, H, bflag);
    for (int j = 0; j <= KMAX; j++) for (int v = 0; v < NV; v++) R.c[j][v].assign(NP, 0.0);
    const double t = 0.7, dx_i = H, dt = BSSN_RK45_TIME_STEP_SIZE, h_ssl = BSSN_SSL_H, sig_ssl = BSSN_SSL_SIGMA;
    const unsigned lambda[4] = {BSSN_LAMBDA[0], BSSN_LAMBDA[1], BSSN_LAMBDA[2], BSSN_LAMBDA[3]};
    const double lambda_f[2] = {BSSN_LAMBDA_F[0], BSSN_LAMBDA_F[1]};
    const double bh1x = 0, bh1y = 0, bh1z = 0, bh2x = 0, bh2y = 0, bh2z = 0; (void)bh1x; (void)bh1y; (void)bh1z; (void)bh2x; (void)bh2y; (void)bh2z;
    (void)dt; (void)dx_i; (void)h_ssl; (void)sig_ssl; (void)t; (void)lambda; (void)lambda_f;
    const double sigma = KO_DISS_SIGMA;
    for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) {
        const int pp = i + N * (j + N * k);
        const double x = PMIN[0] + i * H, y = PMIN[1] + j * H, z = PMIN[2] + k * H;
        const double r_coord = std::sqrt(x * x + y * y + z * z);
#include "eta_RIT.inc.cpp"
#define LD(n, v) T n##_T; for (int q = 0; q <= KMAX; q++) n##_T.c[q] = n##_c[q][pp];
        TAYLOR_FIELD_NAMES(LD)
#undef LD
#define LD(n) T n##_T; for (int q = 0; q <= KMAX; q++) n##_T.c[q] = n##_c[q][pp];
        TAYLOR_DERIV_NAMES(LD)
#undef LD
#define OD(n, v, f) T OUT_##n;
        TAYLOR_OUT_NAMES(OD)
#undef OD
#include "taylor_point.inc"
#define STORE(n, v, f) for (int q = 0; q <= KMAX; q++) R.c[q][VAR::v][pp] = OUT_##n.c[q];
        TAYLOR_OUT_NAMES(STORE)
#undef STORE
    }
    if (bflag) {
        double pmax[3] = {PMIN[0] + H * (N - 1), PMIN[1] + H * (N - 1), PMIN[2] + H * (N - 1)};
        unsigned sz[3] = {N, N, N};
        for (int q = 0; q <= KMAX; q++) for (int v = 0; v < NV; v++)
            bssn_bcs(R.c[q][v].data(), S.c[q][v].data(), bcg[q][v][0], bcg[q][v][1], bcg[q][v][2], PMIN, pmax, bc_fall(v), q == 0 ? bc_asym(v) : 0.0, sz, bflag);
    }
#define KOADD(n, v, f) \
    for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) { const int pp = i + N * (j + N * k); \
        for (int q = 0; q <= KMAX; q++) R.c[q][VAR::v][pp] += sigma * (ko_c["grad_0_" #f][q][pp] + ko_c["grad_1_" #f][q][pp] + ko_c["grad_2_" #f][q][pp]); }
    TAYLOR_OUT_NAMES(KOADD)
#undef KOADD
}

static bool interior(int i, int j, int k) { return i >= PWD && i < N - PWD && j >= PWD && j < N - PWD && k >= PWD && k < N - PWD; }

static void enforce_state_jet(JetState& S, FloorLog* lg) {
    const int gI[6] = {VAR::U_SYMGT0, VAR::U_SYMGT1, VAR::U_SYMGT2, VAR::U_SYMGT3, VAR::U_SYMGT4, VAR::U_SYMGT5};
    const int aI[6] = {VAR::U_SYMAT0, VAR::U_SYMAT1, VAR::U_SYMAT2, VAR::U_SYMAT3, VAR::U_SYMAT4, VAR::U_SYMAT5};
    for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
        if (!interior(i, j, k)) continue;
        int pp = i + N * (j + N * k);
        T gt[6], At[6], chi, al;
        for (int q = 0; q <= KMAX; q++) {
            for (int m = 0; m < 6; m++) { gt[m].c[q] = S.c[q][gI[m]][pp]; At[m].c[q] = S.c[q][aI[m]][pp]; }
            chi.c[q] = S.c[q][VAR::U_CHI][pp]; al.c[q] = S.c[q][VAR::U_ALPHA][pp];
        }
        enforce_jet(gt, At, chi, al, CHI_FLOOR, lg);
        for (int q = 0; q <= KMAX; q++) {
            for (int m = 0; m < 6; m++) { S.c[q][gI[m]][pp] = gt[m].c[q]; S.c[q][aI[m]][pp] = At[m].c[q]; }
            S.c[q][VAR::U_CHI][pp] = chi.c[q]; S.c[q][VAR::U_ALPHA][pp] = al.c[q];
        }
    }
}

// state update on the interior only (padding frozen at its initial value): S_out = S + a * K  (all coefficients)
static void axpy_interior(const JetState& S, const JetState& Kk, double a, JetState& out) {
    out = S;
    for (int q = 0; q <= KMAX; q++) for (int v = 0; v < NV; v++)
        for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k)) { int pp = i + N * (j + N * k); out.c[q][v][pp] = S.c[q][v][pp] + a * Kk.c[q][v][pp]; }
}

static void jet_rk4_step(JetState& S, double dt, unsigned bflag, FloorLog* lg) {
    JetState k1, k2, k3, k4, Sa;
    jet_rhs(S, bflag, k1);
    if (getenv("TSTEP_DEBUG")) { for (int v = 0; v < NV; v++) { double m = 0, m0 = 0; for (int p = 0; p < NP; p++) { m = std::max(m, std::fabs(k1.c[0][v][p])); m0 = std::max(m0, std::fabs(S.c[0][v][p])); } printf("   dbg var %d max|R0|=%.3e max|u0|=%.3e\n", v, m, m0); } }
    axpy_interior(S, k1, 0.5 * dt, Sa); jet_rhs(Sa, bflag, k2);
    axpy_interior(S, k2, 0.5 * dt, Sa); jet_rhs(Sa, bflag, k3);
    axpy_interior(S, k3, dt, Sa);       jet_rhs(Sa, bflag, k4);
    JetState Sn = S;
    for (int q = 0; q <= KMAX; q++) for (int v = 0; v < NV; v++)
        for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k)) { int pp = i + N * (j + N * k);
            Sn.c[q][v][pp] = S.c[q][v][pp] + dt / 6.0 * (k1.c[q][v][pp] + 2 * k2.c[q][v][pp] + 2 * k3.c[q][v][pp] + k4.c[q][v][pp]); }
    if (getenv("TSTEP_DEBUG")) { double mc = 1e9, mx = 0; for (int p = 0; p < NP; p++) { mc = std::min(mc, Sn.c[0][VAR::U_CHI][p]); } for (int v=0;v<NV;v++) for (int p = 0; p < NP; p++) mx = std::max(mx, std::fabs(Sn.c[0][v][p]-S.c[0][v][p])); printf("   dbg after update: min chi=%.4f max|dU|=%.3e\n", mc, mx); }
    S = Sn;
    enforce_state_jet(S, lg);
    if (getenv("TSTEP_DEBUG")) { double mc = 1e9; for (int p = 0; p < NP; p++) mc = std::min(mc, S.c[0][VAR::U_CHI][p]); printf("   dbg after enforce: min chi=%.4f chi_active so far=%ld\n", mc, lg->chi_active); }
}

// ---- production double path with the identical procedure
static void prod_rhs_d(const Vec u[NV], unsigned bflag, Vec out[NV]) {
    std::vector<const double*> up(NV); std::vector<double*> op(NV);
    Vec uc[BSSN_CONSTRAINT_NUM_VARS]; std::vector<const double*> ucp(BSSN_CONSTRAINT_NUM_VARS);
    for (int v = 0; v < NV; v++) { out[v].assign(NP, 0.0); up[v] = u[v].data(); op[v] = out[v].data(); }
    for (int c = 0; c < BSSN_CONSTRAINT_NUM_VARS; c++) { uc[c].assign(NP, 0.0); ucp[c] = uc[c].data(); }
    unsigned sz[3] = {N, N, N}, off = 0;
    double pmin[3] = {PMIN[0], PMIN[1], PMIN[2]}, pmax[3] = {PMIN[0] + H * (N - 1), PMIN[1] + H * (N - 1), PMIN[2] + H * (N - 1)};
    bssnrhs(op.data(), up.data(), off, pmin, pmax, sz, bflag, 0.7, ucp.data());
}
static void axpy_d(const Vec s[NV], const Vec k[NV], double a, Vec o[NV]) {
    for (int v = 0; v < NV; v++) { o[v] = s[v]; for (int k3 = 0; k3 < N; k3++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k3)) { int pp = i + N * (j + N * k3); o[v][pp] = s[v][pp] + a * k[v][pp]; } }
}
static void prod_rk4_step(Vec S[NV], double dt, unsigned bflag) {
    Vec k1[NV], k2[NV], k3[NV], k4[NV], Sa[NV];
    prod_rhs_d(S, bflag, k1);
    axpy_d(S, k1, 0.5 * dt, Sa); prod_rhs_d(Sa, bflag, k2);
    axpy_d(S, k2, 0.5 * dt, Sa); prod_rhs_d(Sa, bflag, k3);
    axpy_d(S, k3, dt, Sa);       prod_rhs_d(Sa, bflag, k4);
    Vec Sn[NV];
    for (int v = 0; v < NV; v++) { Sn[v] = S[v]; for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k)) { int pp = i + N * (j + N * k);
        Sn[v][pp] = S[v][pp] + dt / 6.0 * (k1[v][pp] + 2 * k2[v][pp] + 2 * k3[v][pp] + k4[v][pp]); } }
    for (int v = 0; v < NV; v++) S[v] = Sn[v];
    // production enforcement on interior nodes
    std::vector<double*> arr(NV); for (int v = 0; v < NV; v++) arr[v] = S[v].data();
    for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k)) enforce_bssn_constraints(arr.data(), i + N * (j + N * k));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const double amp0 = argc > 2 ? atof(argv[2]) : 0.01, ampj = argc > 3 ? atof(argv[3]) : 0.01, dt = argc > 4 ? atof(argv[4]) : 2e-4; const int nsteps = argc > 1 ? atoi(argv[1]) : 3;
    BSSN_LAMBDA[0] = 1; BSSN_LAMBDA[1] = 1; BSSN_LAMBDA[2] = 1; BSSN_LAMBDA[3] = 1;
    BSSN_LAMBDA_F[0] = 1.0; BSSN_LAMBDA_F[1] = 0.0;
    KO_DISS_SIGMA = 0.1; BSSN_CAHD_C = getenv("TSTEP_CAHD") ? atof(getenv("TSTEP_CAHD")) : 0.02; BSSN_SSL_H = 0.6; BSSN_SSL_SIGMA = 1.0;
    BSSN_RK45_TIME_STEP_SIZE = dt; BSSN_CURRENT_MIN_DX = 0.1;
    RIT_ETA_OUTER = 0.25; RIT_ETA_CENTRAL = 2.0; RIT_ETA_WIDTH = 40.0;
    BSSN_PADDING_WIDTH = PWD; BSSN_CAKO_ENABLED = false; CHI_FLOOR = 0.1;
    set_appropriate_derivs(PWD);
    BSSN_DERIV_WORKSPACE = new double[(size_t)BSSN_NUM_DERIVS * NP];
    srand(12345);
    JetState S0;
    for (int v = 0; v < NV; v++) {
        double base = (v == VAR::U_ALPHA || v == VAR::U_CHI || v == VAR::U_SYMGT0 || v == VAR::U_SYMGT3 || v == VAR::U_SYMGT5) ? 1.0 : 0.0;
        for (int j = 0; j <= KMAX; j++) {
            S0.c[j][v].assign(NP, 0.0);
            const double a = (j == 0 ? amp0 : ampj);
            double ph = 6.28 * rand() / RAND_MAX, kx = 1 + 4.0 * rand() / RAND_MAX, ky = 1 + 4.0 * rand() / RAND_MAX, kz = 1 + 4.0 * rand() / RAND_MAX, ph2 = 6.28 * rand() / RAND_MAX;
            for (int kk = 0; kk < N; kk++) for (int jj = 0; jj < N; jj++) for (int ii = 0; ii < N; ii++) {
                double x = PMIN[0] + ii * H, y = PMIN[1] + jj * H, z = PMIN[2] + kk * H;
                S0.c[j][v][ii + N * (jj + N * kk)] = (j == 0 ? base : 0.0) + a * (std::sin(kx * x + ph) * std::cos(ky * y + ph2) + 0.5 * std::sin(kz * z + ky * x + ph2));
            }
        }
    }
    // enforce initial constraints (det=1,tr=0) on the whole jet state incl. padding is not needed: only interior is enforced after steps
    struct Case { const char* name; unsigned bflag; };
    const unsigned ALL = (1u << OCT_DIR_LEFT) | (1u << OCT_DIR_RIGHT) | (1u << OCT_DIR_DOWN) | (1u << OCT_DIR_UP) | (1u << OCT_DIR_BACK) | (1u << OCT_DIR_FRONT);
    const unsigned PART = (1u << OCT_DIR_LEFT) | (1u << OCT_DIR_UP) | (1u << OCT_DIR_FRONT);
    Case cases[3] = {{"no boundary (bflag=0)", 0}, {"3 faces (LEFT|UP|FRONT)", PART}, {"all 6 faces", ALL}};
    if (getenv("TSTEP_RHSCMP")) {
        for (auto& cs : cases) {
            JetState R; jet_rhs(S0, cs.bflag, R); Vec pr[NV]; Vec u[NV]; for (int v = 0; v < NV; v++) u[v] = S0.c[0][v]; prod_rhs_d(u, cs.bflag, pr);
            printf("RHS compare %s\n", cs.name);
            for (int v = 0; v < NV; v++) { double md = 0; int bi = 0, bj = 0, bk = 0; for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k)) { double d = std::fabs(pr[v][i + N * (j + N * k)] - R.c[0][v][i + N * (j + N * k)]); if (d > md) { md = d; bi = i; bj = j; bk = k; } }
                if (md > 1e-13) printf("  var %d maxdiff %.3e at (%d,%d,%d)\n", v, md, bi, bj, bk); }
        }
    }
    for (auto& cs : cases) {
        JetState S = S0; FloorLog lg;
        for (int s = 0; s < nsteps; s++) jet_rk4_step(S, dt, cs.bflag, &lg);
        printf("== %s : %d RK4 steps dt=%.3f, jets order %d, floor log: points=%ld chi_active=%ld alpha_active=%ld\n", cs.name, nsteps, dt, KMAX, lg.points, lg.chi_active, lg.alpha_active);
        const double epss[] = {0.2, 0.1, 0.05, 0.025};
        double prev[KMAX + 1] = {0, 0, 0, 0};
        for (int ie = 0; ie < 4; ie++) {
            double eps = epss[ie];
            Vec u[NV];
            for (int v = 0; v < NV; v++) { u[v].assign(NP, 0.0); for (int p = 0; p < NP; p++) { double e = 1, s = 0; for (int j = 0; j <= KMAX; j++) { s += S0.c[j][v][p] * e; e *= eps; } u[v][p] = s; } }
            for (int s = 0; s < nsteps; s++) prod_rk4_step(u, dt, cs.bflag);
            double rem[KMAX + 1] = {0, 0, 0, 0};
            for (int m = 0; m <= KMAX; m++) for (int v = 0; v < NV; v++) for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k)) {
                int pp = i + N * (j + N * k); double s = 0, e = 1; for (int q = 0; q <= m; q++) { s += S.c[q][v][pp] * e; e *= eps; }
                rem[m] = std::max(rem[m], std::fabs(u[v][pp] - s)); }
            printf("  eps=%.3f  %.3e %.3e %.3e %.3e", eps, rem[0], rem[1], rem[2], rem[3]);
            if (ie) { printf("  | slopes"); for (int m = 0; m <= KMAX; m++) printf(" %.2f", std::log2(prev[m] / rem[m])); }
            printf("\n");
            for (int m = 0; m <= KMAX; m++) prev[m] = rem[m];
        }
        // order-0 fidelity: eps = 0
        Vec u[NV]; for (int v = 0; v < NV; v++) u[v] = S0.c[0][v];
        for (int s = 0; s < nsteps; s++) prod_rk4_step(u, dt, cs.bflag);
        double md = 0; for (int v = 0; v < NV; v++) for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) if (interior(i, j, k)) { int pp = i + N * (j + N * k); md = std::max(md, std::fabs(u[v][pp] - S.c[0][v][pp])); }
        printf("  coefficient-0 fidelity vs production RK4 path: max|diff| = %.3e\n", md);
    }
    MPI_Finalize();
    return 0;
}
