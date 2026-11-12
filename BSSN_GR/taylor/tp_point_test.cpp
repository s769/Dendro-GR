// B2: end-to-end check of the TwoPunctures initial-data tangent: jets (U tangent from the IFT solve + parameter jets through the
// pointwise map) vs Richardson FD of (Newton re-solve -> spectral interpolation -> pointwise map), and order-0 vs production TwoPunctures().
// usage: tp_point_test <paramfile> <mp|mm|Py|Sz> [delta] [npts]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <mpi.h>
#include "TwoPunctures.h"
#include "grUtils.h"
#include "parameters.h"
#include "gr.h"
#include "taylor.hpp"
#include "tp_point.hpp"
using namespace bssn;
typedef Tay<1> T1;
static double* par_ptr(const std::string& s) {
    if (s == "mp") return &TPID::par_m_plus; if (s == "mm") return &TPID::par_m_minus;
    if (s == "Py") return &TPID::par_P_plus[1]; if (s == "Sz") return &TPID::par_S_plus[2]; return nullptr;
}
template <class S> static TPPar<S> params_from_globals() {
    TPPar<S> P; P.mp = S(TPID::par_m_plus); P.mm = S(TPID::par_m_minus); P.b = S(TPID::par_b);
    for (int i = 0; i < 3; i++) { P.Pp[i] = S(TPID::par_P_plus[i]); P.Pm[i] = S(TPID::par_P_minus[i]); P.Sp[i] = S(TPID::par_S_plus[i]); P.Sm[i] = S(TPID::par_S_minus[i]); }
    return P;
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    TP_OMP_THREADS = 16;
    bssn::readParamFile(argv[1], MPI_COMM_WORLD);
    std::string which = argv[2]; double* th = par_ptr(which);
    const double th0 = *th, delta = argc > 3 ? atof(argv[3]) : 1e-3; const int npts = argc > 4 ? atoi(argv[4]) : 200;
    const int nvar = 1, n1 = TPID::npoints_A, n2 = TPID::npoints_B, n3 = TPID::npoints_phi, N = n1 * n2 * n3;
    TPID::verbose = 0;
    CCTK_REAL* F0; derivs u0, v0, cf0; TPRestore(F0, u0, v0, cf0, TPID::FILE_PREFIX.c_str(), false);
    // ---- tangent of the solution (same procedure as tp_tangent)
    derivs u, v; allocate_derivs(&u, N); allocate_derivs(&v, N); memcpy(v.d0, v0.d0, N * sizeof(double));
    std::vector<double> Fp(N), Fm(N), Fp2(N), Fm2(N), rhs(N);
    auto Fat = [&](double val, std::vector<double>& out) { *th = val; derivs uu; allocate_derivs(&uu, N); F_of_v(nvar, n1, n2, n3, v, out.data(), uu); free_derivs(&uu, N); *th = th0; };
    const double hF = 1e-4 * std::max(1.0, std::fabs(th0));
    Fat(th0 + hF, Fp); Fat(th0 - hF, Fm); Fat(th0 + hF / 2, Fp2); Fat(th0 - hF / 2, Fm2);
    for (int i = 0; i < N; i++) { double d1 = (Fp[i] - Fm[i]) / (2 * hF), d2 = (Fp2[i] - Fm2[i]) / hF; rhs[i] = -(4 * d2 - d1) / 3; }
    derivs dv; allocate_derivs(&dv, N); double nr = 0;
    TPLinearSolve(nvar, n1, n2, n3, v, rhs.data(), dv, 0, 200, 1e-16, &nr);
    derivs cft; allocate_derivs(&cft, N); SpecCoef(n1, n2, n3, 0, dv.d0, cft.d0);   // spectral coefficients of the tangent
    // ---- sample points
    srand(99); std::vector<double> X(npts), Y(npts), Z(npts);
    for (int p = 0; p < npts; p++) for (;;) {
        double x = 40 * (rand() / (double)RAND_MAX - 0.5), y = 40 * (rand() / (double)RAND_MAX - 0.5), z = 40 * (rand() / (double)RAND_MAX - 0.5);
        double rp = std::sqrt((x - TPID::par_b) * (x - TPID::par_b) + y * y + z * z), rm = std::sqrt((x + TPID::par_b) * (x + TPID::par_b) + y * y + z * z);
        if (rp > 0.4 && rm > 0.4) { X[p] = x; Y[p] = y; Z[p] = z; break; }
    }
    // ---- jets
    TPPar<T1> PJ = params_from_globals<T1>();
    T1* tj = which == "mp" ? &PJ.mp : which == "mm" ? &PJ.mm : which == "Py" ? &PJ.Pp[1] : &PJ.Sp[2]; tj->c[1] = 1.0;
    std::vector<std::vector<double>> jet(npts, std::vector<double>(15)), j0(npts, std::vector<double>(15)); double maxfid = 0;
    for (int p = 0; p < npts; p++) {
        T1 U; U.c[0] = PunctIntPolAtArbitPositionFast(0, nvar, n1, n2, n3, cf0, X[p], Y[p], Z[p]);
        U.c[1] = PunctIntPolAtArbitPositionFast(0, nvar, n1, n2, n3, cft, X[p], Y[p], Z[p]);
        T1 o[15]; tp_point<T1>(PJ, T1(X[p]), T1(Y[p]), T1(Z[p]), U, o);
        for (int m = 0; m < 15; m++) { j0[p][m] = o[m].c[0]; jet[p][m] = o[m].c[1]; }
        // production TwoPunctures() at the fiducial
        double vars[24] = {0}, a, b, c, d, e, f, g, h; vars[VAR::U_ALPHA] = 1;
        TwoPunctures(X[p], Y[p], Z[p], vars, &a, &b, &c, &d, &e, &f, &g, &h);
        const int idx[15] = {VAR::U_SYMGT0, VAR::U_SYMGT1, VAR::U_SYMGT2, VAR::U_SYMGT3, VAR::U_SYMGT4, VAR::U_SYMGT5, VAR::U_SYMAT0, VAR::U_SYMAT1, VAR::U_SYMAT2, VAR::U_SYMAT3, VAR::U_SYMAT4, VAR::U_SYMAT5, VAR::U_K, VAR::U_CHI, VAR::U_ALPHA};
        for (int m = 0; m < 15; m++) maxfid = std::max(maxfid, std::fabs(vars[idx[m]] - o[m].c[0]) / (1e-30 + std::max(1e-3, std::fabs(vars[idx[m]]))));
    }
    printf("(A) order-0 jet map vs production TwoPunctures(): max relative diff over %d points x 15 vars = %.3e\n", npts, maxfid);
    // ---- oracle: re-solve, spectral coefficients, double map
    auto vars_at = [&](double val, std::vector<std::vector<double>>& out) {
        derivs vv; allocate_derivs(&vv, N); memcpy(vv.d0, v0.d0, N * sizeof(double)); *th = val; Newton(nvar, n1, n2, n3, vv, 1e-15, 8);
        derivs cf; allocate_derivs(&cf, N); SpecCoef(n1, n2, n3, 0, vv.d0, cf.d0);
        TPPar<T1> PP = params_from_globals<T1>(); out.assign(npts, std::vector<double>(15));
        for (int p = 0; p < npts; p++) { T1 U; U.c[0] = PunctIntPolAtArbitPositionFast(0, nvar, n1, n2, n3, cf, X[p], Y[p], Z[p]); T1 o[15]; tp_point<T1>(PP, T1(X[p]), T1(Y[p]), T1(Z[p]), U, o); for (int m = 0; m < 15; m++) out[p][m] = o[m].c[0]; }
        *th = th0; free_derivs(&vv, N); free_derivs(&cf, N);
    };
    std::vector<std::vector<double>> a, b, a2, b2; vars_at(th0 + delta, a); vars_at(th0 - delta, b); vars_at(th0 + delta / 2, a2); vars_at(th0 - delta / 2, b2);
    const char* names[15] = {"gt0", "gt1", "gt2", "gt3", "gt4", "gt5", "At0", "At1", "At2", "At3", "At4", "At5", "K", "chi", "alpha"};
    printf("(B) jet tangent vs FD of re-solved initial data (%s, delta=%g): per variable max|tan-FD| / max|tan|\n   var:  max|tan|   FD(d)    FD(d/2)  Richardson\n", which.c_str(), delta);
    for (int m = 0; m < 15; m++) {
        double mt = 0, e1 = 0, e2 = 0, eR = 0;
        for (int p = 0; p < npts; p++) mt = std::max(mt, std::fabs(jet[p][m]));
        for (int p = 0; p < npts; p++) {
            double f1 = (a[p][m] - b[p][m]) / (2 * delta), f2 = (a2[p][m] - b2[p][m]) / delta, fR = (4 * f2 - f1) / 3;
            e1 = std::max(e1, std::fabs(jet[p][m] - f1)); e2 = std::max(e2, std::fabs(jet[p][m] - f2)); eR = std::max(eR, std::fabs(jet[p][m] - fR));
        }
        printf("   %-5s %.3e  %.2e  %.2e  %.2e\n", names[m], mt, e1 / std::max(mt, 1e-300), e2 / std::max(mt, 1e-300), eR / std::max(mt, 1e-300));
    }
    MPI_Finalize(); return 0;
}
