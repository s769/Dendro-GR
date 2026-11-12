// B2: jet (truncated-Taylor) version of the production Hamiltonian/momentum constraints and psi4 (src/physcon.cpp physconeqs.cpp).
// Stencils (evar + constraint derivative fragments) are applied per Taylor coefficient; the generated pointwise algebra is evaluated in Tay<K>.
// Tests: order-0 vs production physical_constraints(); Taylor remainder with production evaluated at displaced fields u(eps).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include "derivs.h"
#include "parameters.h"
#include "rhs.h"
#include "gr.h"
#include "physcon.h"
#include "taylor.hpp"
#include "taylor_psi4_names.inc"
using namespace bssn;
#define KMAX 3
typedef Tay<KMAX> T;
static const int N = 13, PWD = 3, NP = N * N * N, NV = 24;
#define DECL(n, v) static const double* n##_c[KMAX + 1];
PSI_FIELD_USED(DECL)
#undef DECL
#define DECL(n) static const double* n##_c[KMAX + 1];
PSI_DERIV_NAMES(DECL)
#undef DECL
static std::vector<double> workspace[KMAX + 1];

static void setup_coeff(int k, const double* const* f, double hx, double hy, double hz, unsigned bflag) {
    unsigned sz[3] = {N, N, N};
    const unsigned BLK_SZ = NP;
    workspace[k].assign((size_t)BSSN_NUM_DERIVS * NP, 0.0);
    double* deriv_base = workspace[k].data();
    const double* const alpha = f[VAR::U_ALPHA]; const double* const chi = f[VAR::U_CHI]; const double* const K = f[VAR::U_K];
    const double* const gt0 = f[VAR::U_SYMGT0]; const double* const gt1 = f[VAR::U_SYMGT1]; const double* const gt2 = f[VAR::U_SYMGT2];
    const double* const gt3 = f[VAR::U_SYMGT3]; const double* const gt4 = f[VAR::U_SYMGT4]; const double* const gt5 = f[VAR::U_SYMGT5];
    const double* const beta0 = f[VAR::U_BETA0]; const double* const beta1 = f[VAR::U_BETA1]; const double* const beta2 = f[VAR::U_BETA2];
    const double* const At0 = f[VAR::U_SYMAT0]; const double* const At1 = f[VAR::U_SYMAT1]; const double* const At2 = f[VAR::U_SYMAT2];
    const double* const At3 = f[VAR::U_SYMAT3]; const double* const At4 = f[VAR::U_SYMAT4]; const double* const At5 = f[VAR::U_SYMAT5];
    const double* const Gt0 = f[VAR::U_GT0]; const double* const Gt1 = f[VAR::U_GT1]; const double* const Gt2 = f[VAR::U_GT2];
    const double* const B0 = f[VAR::U_B0]; const double* const B1 = f[VAR::U_B1]; const double* const B2 = f[VAR::U_B2];
    (void)alpha; (void)beta0; (void)beta1; (void)beta2; (void)B0; (void)B1; (void)B2;
    {
#include "bssnrhs_evar_derivs.h"
#include "constraint_derivs.h"
#define REG(n) n##_c[k] = n;
        PSI_DERIV_NAMES(REG)
#undef REG
    }
#define REG(n, v) n##_c[k] = f[VAR::v];
    PSI_FIELD_USED(REG)
#undef REG
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const double amp0 = argc > 1 ? atof(argv[1]) : 0.05, ampj = argc > 2 ? atof(argv[2]) : 0.02;
    BSSN_PADDING_WIDTH = PWD; BSSN_CAKO_ENABLED = false;
    set_appropriate_derivs(PWD);
    const double h = 0.1;
    const double pmin[3] = {-0.6 + 0.35, -0.6 - 0.2, -0.6 + 0.1}, pmax[3] = {pmin[0] + h * (N - 1), pmin[1] + h * (N - 1), pmin[2] + h * (N - 1)};
    std::vector<double> coef[KMAX + 1][NV];
    srand(4321);
    for (int v = 0; v < NV; v++) {
        double base = (v == VAR::U_ALPHA || v == VAR::U_CHI || v == VAR::U_SYMGT0 || v == VAR::U_SYMGT3 || v == VAR::U_SYMGT5) ? 1.0 : 0.0;
        for (int j = 0; j <= KMAX; j++) {
            coef[j][v].assign(NP, 0.0);
            const double a = (j == 0 ? amp0 : ampj);
            double ph = 6.28 * rand() / RAND_MAX, kx = 1 + 4.0 * rand() / RAND_MAX, ky = 1 + 4.0 * rand() / RAND_MAX, kz = 1 + 4.0 * rand() / RAND_MAX, ph2 = 6.28 * rand() / RAND_MAX;
            for (int kk = 0; kk < N; kk++) for (int jj = 0; jj < N; jj++) for (int ii = 0; ii < N; ii++) {
                double x = pmin[0] + ii * h, y = pmin[1] + jj * h, z = pmin[2] + kk * h;
                coef[j][v][ii + N * (jj + N * kk)] = (j == 0 ? base : 0.0) + a * (std::sin(kx * x + ph) * std::cos(ky * y + ph2) + 0.5 * std::sin(kz * z + ky * x + ph2));
            }
        }
    }
    for (int j = 0; j <= KMAX; j++) { const double* fj[NV]; for (int v = 0; v < NV; v++) fj[v] = coef[j][v].data(); setup_coeff(j, fj, h, h, h, 0); }
    const char* onames[6] = {"ham", "mom0", "mom1", "mom2", "psi4_real", "psi4_img"};
    std::vector<double> Rj[KMAX + 1][6];
    for (int j = 0; j <= KMAX; j++) for (int o = 0; o < 6; o++) Rj[j][o].assign(NP, 0.0);
    for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) {
        const int pp = i + N * (j + N * k);
        const double x = pmin[0] + i * h, y = pmin[1] + j * h, z = pmin[2] + k * h;
#define LD(n, v) T n##_T; for (int q = 0; q <= KMAX; q++) n##_T.c[q] = n##_c[q][pp];
        PSI_FIELD_USED(LD)
#undef LD
#define LD(n) T n##_T; for (int q = 0; q <= KMAX; q++) n##_T.c[q] = n##_c[q][pp];
        PSI_DERIV_NAMES(LD)
#undef LD
#define OD(n) T OUT_##n;
        PSI_OUT_NAMES(OD)
#undef OD
#include "taylor_psi4_point.inc"
        T* outs[6] = {&OUT_ham, &OUT_mom0, &OUT_mom1, &OUT_mom2, &OUT_psi4_real, &OUT_psi4_img};
        for (int o = 0; o < 6; o++) for (int q = 0; q <= KMAX; q++) Rj[q][o][pp] = outs[o]->c[q];
    }
    // production evaluation at displaced fields
    auto prod = [&](double eps, std::vector<double> out[6]) {
        std::vector<double> u[NV]; std::vector<const double*> up(NV);
        for (int v = 0; v < NV; v++) { u[v].assign(NP, 0.0); for (int p = 0; p < NP; p++) { double e = 1, s = 0; for (int j = 0; j <= KMAX; j++) { s += coef[j][v][p] * e; e *= eps; } u[v][p] = s; } up[v] = u[v].data(); }
        std::vector<double> uc[BSSN_CONSTRAINT_NUM_VARS]; std::vector<double*> ucp(BSSN_CONSTRAINT_NUM_VARS);
        for (int c = 0; c < BSSN_CONSTRAINT_NUM_VARS; c++) { uc[c].assign(NP, 0.0); ucp[c] = uc[c].data(); }
        unsigned sz[3] = {N, N, N}; unsigned bflag = 0, off = 0;
        physical_constraints(ucp.data(), up.data(), off, pmin, pmax, sz, bflag);
        const int ids[6] = {VAR_CONSTRAINT::C_HAM, VAR_CONSTRAINT::C_MOM0, VAR_CONSTRAINT::C_MOM1, VAR_CONSTRAINT::C_MOM2, VAR_CONSTRAINT::C_PSI4_REAL, VAR_CONSTRAINT::C_PSI4_IMG};
        for (int o = 0; o < 6; o++) out[o] = uc[ids[o]];
    };
    std::vector<double> R0[6]; prod(0.0, R0);
    printf("(A) order-0 fidelity vs production physical_constraints():\n");
    for (int o = 0; o < 6; o++) {
        double mx = 0, md = 0;
        for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) { int pp = i + N * (j + N * k); mx = std::max(mx, std::fabs(R0[o][pp])); md = std::max(md, std::fabs(R0[o][pp] - Rj[0][o][pp])); }
        printf("   %-9s max|prod|=%.3e  max|diff|=%.3e (rel %.2e)\n", onames[o], mx, md, md / std::max(mx, 1e-300));
    }
    printf("(B) Taylor remainder (slopes between successive eps = 0.4,0.2,0.1,0.05) for orders m=0..3, worst of 6 outputs:\n");
    double prev[6][KMAX + 1] = {};
    const double epss[] = {0.4, 0.2, 0.1, 0.05};
    for (int ie = 0; ie < 4; ie++) {
        double eps = epss[ie]; std::vector<double> Re[6]; prod(eps, Re);
        double rem[6][KMAX + 1] = {};
        for (int o = 0; o < 6; o++) for (int m = 0; m <= KMAX; m++)
            for (int k = PWD; k < N - PWD; k++) for (int j = PWD; j < N - PWD; j++) for (int i = PWD; i < N - PWD; i++) {
                int pp = i + N * (j + N * k); double s = 0, e = 1; for (int q = 0; q <= m; q++) { s += Rj[q][o][pp] * e; e *= eps; }
                rem[o][m] = std::max(rem[o][m], std::fabs(Re[o][pp] - s)); }
        if (ie) {
            printf("   eps=%.3f slopes:", eps);
            for (int m = 0; m <= KMAX; m++) { double mn = 1e9, mxx = -1e9; for (int o = 0; o < 6; o++) { double sl = std::log2(prev[o][m] / rem[o][m]); mn = std::min(mn, sl); mxx = std::max(mxx, sl); } printf("  m=%d [%.2f,%.2f]", m, mn, mxx); }
            printf("\n");
        }
        for (int o = 0; o < 6; o++) for (int m = 0; m <= KMAX; m++) prev[o][m] = rem[o][m];
    }
    MPI_Finalize(); return 0;
}
