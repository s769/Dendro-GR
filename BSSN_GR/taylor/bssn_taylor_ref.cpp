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
#include <omp.h>

#include "derivs.h"
#include "parameters.h"
#include "rhs.h"
#include "gr.h"
#include "taylor.hpp"
#include "taylor_names.inc"

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

// production-equivalent double fields (polynomial in eps) -> RHS via production bssnrhs
struct Setup {
    double pmin[3], pmax[3], h;
};


// ---- timing of the pointwise Taylor algebra for truncation order K (cost model input for B4): ns per grid point
template <int K>
static double time_pointwise(int reps, const double pmin[3], double h, double* sink) {
    typedef Tay<K> T;
    const double dx_i = h, dt = BSSN_RK45_TIME_STEP_SIZE, h_ssl = BSSN_SSL_H, sig_ssl = BSSN_SSL_SIGMA, t = 0.7;
    const unsigned lambda[4] = {BSSN_LAMBDA[0], BSSN_LAMBDA[1], BSSN_LAMBDA[2], BSSN_LAMBDA[3]};
    const double lambda_f[2] = {BSSN_LAMBDA_F[0], BSSN_LAMBDA_F[1]};
    const double bh1x = 0, bh1y = 0, bh1z = 0, bh2x = 0, bh2y = 0, bh2z = 0; (void)bh1x; (void)bh1y; (void)bh1z; (void)bh2x; (void)bh2y; (void)bh2z;
    (void)dt; (void)dx_i; (void)h_ssl; (void)sig_ssl; (void)t; (void)lambda; (void)lambda_f;
    double acc = 0; long npts = 0;
    double t0 = omp_get_wtime();
    for (int rep = 0; rep < reps; rep++)
    for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) {
        const int pp = i + N * (j + N * k);
        const double x = pmin[0] + i * h, y = pmin[1] + j * h, z = pmin[2] + k * h;
        const double r_coord = std::sqrt(x * x + y * y + z * z);
#include "eta_RIT.inc.cpp"
#define LD(n, v) T n##_T; for (int q = 0; q <= K; q++) n##_T.c[q] = n##_c[q][pp];
        TAYLOR_FIELD_NAMES(LD)
#undef LD
#define LD(n) T n##_T; for (int q = 0; q <= K; q++) n##_T.c[q] = n##_c[q][pp];
        TAYLOR_DERIV_NAMES(LD)
#undef LD
#define OD(n, v, f) T OUT_##n;
        TAYLOR_OUT_NAMES(OD)
#undef OD
#include "taylor_point.inc"
#define ACC(n, v, f) acc += OUT_##n.c[K];
        TAYLOR_OUT_NAMES(ACC)
#undef ACC
        npts++;
    }
    double dtw = omp_get_wtime() - t0; *sink += acc;
    return dtw / npts * 1e9;
}

int main(int argc, char** argv) {
    const double amp0 = argc > 1 ? atof(argv[1]) : 0.05, ampj = argc > 2 ? atof(argv[2]) : 0.02;
    // ---- parameters (production globals), chosen to exercise every term
    BSSN_LAMBDA[0] = 1; BSSN_LAMBDA[1] = 1; BSSN_LAMBDA[2] = 1; BSSN_LAMBDA[3] = 1;
    BSSN_LAMBDA_F[0] = 1.0; BSSN_LAMBDA_F[1] = 0.0;
    KO_DISS_SIGMA = 0.1; BSSN_CAHD_C = 0.5; BSSN_SSL_H = 0.6; BSSN_SSL_SIGMA = 1.0;
    BSSN_RK45_TIME_STEP_SIZE = 0.01; BSSN_CURRENT_MIN_DX = 0.1;
    RIT_ETA_OUTER = 0.25; RIT_ETA_CENTRAL = 2.0; RIT_ETA_WIDTH = 40.0;
    BSSN_PADDING_WIDTH = PWD; BSSN_CAKO_ENABLED = false;
    set_appropriate_derivs(PWD);  // bind the production stencil function pointers (6th order, pw=3)
    const double tcur = 0.7;
    const double h = 0.1;
    const double pmin[3] = {-0.6 + 0.35, -0.6 - 0.2, -0.6 + 0.1}, pmax[3] = {pmin[0] + h * (N - 1), pmin[1] + h * (N - 1), pmin[2] + h * (N - 1)};
    const double hx = h, hy = h, hz = h;

    // ---- smooth Taylor-coefficient fields c_j(x) (coefficient 0 = background near flat)
    std::vector<double> coef[KMAX + 1][NV];
    srand(12345);
    for (int v = 0; v < NV; v++) {
        double base = (v == VAR::U_ALPHA || v == VAR::U_CHI || v == VAR::U_SYMGT0 || v == VAR::U_SYMGT3 || v == VAR::U_SYMGT5) ? 1.0 : 0.0;
        for (int j = 0; j <= KMAX; j++) {
            coef[j][v].assign(NP, 0.0);
            const double a = (j == 0 ? amp0 : ampj);
            double ph = 6.28 * rand() / RAND_MAX, kx = 1 + 4.0 * rand() / RAND_MAX, ky = 1 + 4.0 * rand() / RAND_MAX, kz = 1 + 4.0 * rand() / RAND_MAX;
            double ph2 = 6.28 * rand() / RAND_MAX;
            for (int kk = 0; kk < N; kk++) for (int jj = 0; jj < N; jj++) for (int ii = 0; ii < N; ii++) {
                double x = pmin[0] + ii * h, y = pmin[1] + jj * h, z = pmin[2] + kk * h;
                coef[j][v][ii + N * (jj + N * kk)] = (j == 0 ? base : 0.0) + a * (std::sin(kx * x + ph) * std::cos(ky * y + ph2) + 0.5 * std::sin(kz * z + ky * x + ph2));
            }
        }
    }
    // eps-dependent fields u(eps) = sum_j coef_j eps^j; jets use coef_j directly
    const double* fjet[KMAX + 1][NV];
    for (int j = 0; j <= KMAX; j++) for (int v = 0; v < NV; v++) fjet[j][v] = coef[j][v].data();
    for (int j = 0; j <= KMAX; j++) setup_coeff(j, fjet[j], hx, hy, hz, 0);

    // ---- pointwise Taylor evaluation over the interior
    std::vector<double> Rjet[KMAX + 1][NV];
    for (int j = 0; j <= KMAX; j++) for (int v = 0; v < NV; v++) Rjet[j][v].assign(NP, 0.0);
    const double dx_i = h, dt = BSSN_RK45_TIME_STEP_SIZE, h_ssl = BSSN_SSL_H, sig_ssl = BSSN_SSL_SIGMA, t = tcur;
    const unsigned lambda[4] = {BSSN_LAMBDA[0], BSSN_LAMBDA[1], BSSN_LAMBDA[2], BSSN_LAMBDA[3]};
    const double lambda_f[2] = {BSSN_LAMBDA_F[0], BSSN_LAMBDA_F[1]};
    const double BSSN_CAHD_C_l = BSSN_CAHD_C; (void)BSSN_CAHD_C_l;
    const double sigma = KO_DISS_SIGMA;
    const double bh1x = 0, bh1y = 0, bh1z = 0, bh2x = 0, bh2y = 0, bh2z = 0; (void)bh1x; (void)bh1y; (void)bh1z; (void)bh2x; (void)bh2y; (void)bh2z;
    (void)dt;
    for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) {
        const int pp = i + N * (j + N * k);
        const double x = pmin[0] + i * hx, y = pmin[1] + j * hy, z = pmin[2] + k * hz;
        const double r_coord = std::sqrt(x * x + y * y + z * z);
#include "eta_RIT.inc.cpp"
        // load inputs as jets
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
        // KO term: R += sigma * (ko0 + ko1 + ko2)  (linear, coefficient by coefficient)
#define STORE(n, v, f) \
        for (int q = 0; q <= KMAX; q++) { \
            double ko = ko_c["grad_0_" #f][q][pp] + ko_c["grad_1_" #f][q][pp] + ko_c["grad_2_" #f][q][pp]; \
            Rjet[q][VAR::v][pp] = OUT_##n.c[q] + sigma * ko; }
        TAYLOR_OUT_NAMES(STORE)
#undef STORE
    }

    // ---- production reference: RHS of the double field u(eps) via bssnrhs()
    BSSN_DERIV_WORKSPACE = new double[(size_t)BSSN_NUM_DERIVS * NP];
    auto prod_rhs = [&](double eps, std::vector<double> out[NV]) {
        std::vector<double> u[NV], uc[BSSN_CONSTRAINT_NUM_VARS];
        std::vector<const double*> up(NV), ucp(BSSN_CONSTRAINT_NUM_VARS);
        std::vector<double*> op(NV);
        for (int v = 0; v < NV; v++) {
            u[v].assign(NP, 0.0); out[v].assign(NP, 0.0);
            for (int p = 0; p < NP; p++) { double e = 1.0, s = 0; for (int j = 0; j <= KMAX; j++) { s += coef[j][v][p] * e; e *= eps; } u[v][p] = s; }
            up[v] = u[v].data(); op[v] = out[v].data();
        }
        for (int c = 0; c < BSSN_CONSTRAINT_NUM_VARS; c++) { uc[c].assign(NP, 0.0); ucp[c] = uc[c].data(); }
        unsigned sz[3] = {N, N, N}, bflag = 0, off = 0;
        bssnrhs(op.data(), up.data(), off, pmin, pmax, sz, bflag, tcur, ucp.data());
    };
    std::vector<double> R0[NV];
    prod_rhs(0.0, R0);
    // (A) fidelity of coefficient 0
    double maxabs = 0, maxdiff = 0;
    for (int v = 0; v < NV; v++) for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) {
        int pp = i + N * (j + N * k); maxabs = std::max(maxabs, std::fabs(R0[v][pp])); maxdiff = std::max(maxdiff, std::fabs(R0[v][pp] - Rjet[0][v][pp]));
    }
    printf("(A) coefficient-0 fidelity vs production bssnrhs: max|R| = %.4e, max|R_jet0 - R_prod| = %.3e (rel %.2e)\n", maxabs, maxdiff, maxdiff / maxabs);
    // (B) Taylor remainder test
    printf("(B) Taylor remainder: max over interior and variables of |R_prod(eps) - sum_{j<=m} R_j eps^j|\n");
    printf("   eps        m=0         m=1         m=2         m=3   | slopes (log2 ratio between successive eps): m=0 m=1 m=2 m=3\n");
    double prev[KMAX + 1] = {0, 0, 0, 0};
    const double epss[] = {0.4, 0.2, 0.1, 0.05};
    for (int ie = 0; ie < 4; ie++) {
        double eps = epss[ie]; std::vector<double> Re[NV]; prod_rhs(eps, Re);
        double rem[KMAX + 1] = {0, 0, 0, 0};
        for (int m = 0; m <= KMAX; m++) {
            for (int v = 0; v < NV; v++) for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) {
                int pp = i + N * (j + N * k); double s = 0, e = 1.0;
                for (int q = 0; q <= m; q++) { s += Rjet[q][v][pp] * e; e *= eps; }
                rem[m] = std::max(rem[m], std::fabs(Re[v][pp] - s));
            }
        }
        printf("  %.3f  %.4e  %.4e  %.4e  %.4e", eps, rem[0], rem[1], rem[2], rem[3]);
        if (ie > 0) { printf("   |"); for (int m = 0; m <= KMAX; m++) printf("  %.2f", std::log2(prev[m] / rem[m])); }
        printf("\n");
        for (int m = 0; m <= KMAX; m++) prev[m] = rem[m];
    }
    // (C) per-variable relative size of the jets (diagnostic)
    printf("(C) max|R_j| per order: ");
    for (int q = 0; q <= KMAX; q++) { double m = 0; for (int v = 0; v < NV; v++) for (int p = 0; p < NP; p++) m = std::max(m, std::fabs(Rjet[q][v][p])); printf(" R_%d=%.3e", q, m); }
    printf("\n");
    if (getenv("TAYLOR_TIMING")) {
        double sink = 0; const int reps = 300;
        double t0 = omp_get_wtime(); for (int r = 0; r < 20; r++) { std::vector<double> o[NV]; prod_rhs(0.0, o); } double tp = (omp_get_wtime() - t0) / 20;
        printf("(T) timing, single thread, 13^3 block (%d interior points): production bssnrhs() full RHS incl. stencils = %.3f ms (%.0f ns/interior point)\n", (N - 2 * PWD) * (N - 2 * PWD) * (N - 2 * PWD), tp * 1e3, tp * 1e9 / ((N - 2 * PWD) * (N - 2 * PWD) * (N - 2 * PWD)));
        double a0 = time_pointwise<0>(reps, pmin, h, &sink), a1 = time_pointwise<1>(reps, pmin, h, &sink), a2 = time_pointwise<2>(reps, pmin, h, &sink), a3 = time_pointwise<3>(reps, pmin, h, &sink);
        printf("    pointwise algebra only, ns per point: K=0 %.0f | K=1 %.0f (x%.2f) | K=2 %.0f (x%.2f) | K=3 %.0f (x%.2f)   [sink %g]\n", a0, a1, a1 / a0, a2, a2 / a0, a3, a3 / a0, sink);
        printf("    members per solve: K=1 jet (3 coefficients incl. fiducial? no: one univariate jet) -> cost per jet-coefficient: K=1 %.2f, K=2 %.2f, K=3 %.2f of the K=0 cost\n", a1 / a0 / 2, a2 / a0 / 3, a3 / a0 / 4);
    }
    return 0;
}
