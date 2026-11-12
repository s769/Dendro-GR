// B2: parameter tangent of the TwoPunctures spectral solution via the implicit function theorem.
//   F(v; theta) = 0  =>  J(v) dv/dtheta = -dF/dtheta |_v .   J = production Newton Jacobian (J_times_dv + FD preconditioner,
//   solved by the production BiCGStab); dF/dtheta by Richardson-extrapolated central difference of the production residual
//   F_of_v at FIXED nodal values v (F is analytic in theta, so this is exact to ~1e-12).
// Oracle: central FD of full Newton re-solves at theta +- delta (Richardson on two deltas).
// usage: tp_tangent <paramfile> <param: mp|mm|b|Py|Pminus_y|Sz|Sminus_z> [delta]   (run in the dir containing <prefix>_tpid_sol.bin)
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

static double ninf(const double* x, int n) { double m = 0; for (int i = 0; i < n; i++) m = std::max(m, std::fabs(x[i])); return m; }
static double* par_ptr(const std::string& s) {
    if (s == "mp") return &TPID::par_m_plus;
    if (s == "mm") return &TPID::par_m_minus;
    if (s == "b") return &TPID::par_b;
    if (s == "Py") return &TPID::par_P_plus[1];
    if (s == "Pminus_y") return &TPID::par_P_minus[1];
    if (s == "Sz") return &TPID::par_S_plus[2];
    if (s == "Sminus_z") return &TPID::par_S_minus[2];
    return nullptr;
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc < 3) { printf("usage: tp_tangent <paramfile> <param> [delta]\n"); return 1; }
    TP_OMP_THREADS = 16;
    bssn::readParamFile(argv[1], MPI_COMM_WORLD);
    double* th = par_ptr(argv[2]);
    if (!th) { printf("unknown param\n"); return 1; }
    const double th0 = *th, delta = argc > 3 ? atof(argv[3]) : 1e-3 * std::max(1.0, std::fabs(th0));
    const int nvar = 1, n1 = TPID::npoints_A, n2 = TPID::npoints_B, n3 = TPID::npoints_phi, N = n1 * n2 * n3 * nvar;
    printf("TP tangent: param %s = %.15g, grid %dx%dx%d, delta=%g; par_m_plus=%.15g par_m_minus=%.15g par_b=%.15g P+=(%g,%g,%g) S+=(%g,%g,%g)\n",
           argv[2], th0, n1, n2, n3, delta, TPID::par_m_plus, TPID::par_m_minus, TPID::par_b, TPID::par_P_plus[0], TPID::par_P_plus[1], TPID::par_P_plus[2],
           TPID::par_S_plus[0], TPID::par_S_plus[1], TPID::par_S_plus[2]);
    CCTK_REAL* F0; derivs u0, v0, cf0;
    TPRestore(F0, u0, v0, cf0, TPID::FILE_PREFIX.c_str(), false);
    derivs u, v; allocate_derivs(&u, N); allocate_derivs(&v, N);
    std::vector<double> F(N), Fp(N), Fm(N), Fp2(N), Fm2(N), dFth(N), rhs(N);
    memcpy(v.d0, v0.d0, N * sizeof(double));
    // derivative arrays of v are recomputed inside F_of_v (Derivatives_AB3) -- copy the stored ones too for safety
    F_of_v(nvar, n1, n2, n3, v, F.data(), u);
    printf("|F(v_stored)|_inf at theta0 = %.3e   |v|_inf = %.3e\n", ninf(F.data(), N), ninf(v.d0, N));
    // dF/dtheta at fixed v: Richardson (h, h/2)
    auto Fat = [&](double val, std::vector<double>& out) { *th = val; derivs uu; allocate_derivs(&uu, N); F_of_v(nvar, n1, n2, n3, v, out.data(), uu); free_derivs(&uu, N); *th = th0; };
    const double hF = 1e-4 * std::max(1.0, std::fabs(th0));
    Fat(th0 + hF, Fp); Fat(th0 - hF, Fm); Fat(th0 + hF / 2, Fp2); Fat(th0 - hF / 2, Fm2);
    for (int i = 0; i < N; i++) { double d1 = (Fp[i] - Fm[i]) / (2 * hF), d2 = (Fp2[i] - Fm2[i]) / hF; dFth[i] = (4 * d2 - d1) / 3; rhs[i] = -dFth[i]; }
    double c1 = 0; for (int i = 0; i < N; i++) c1 = std::max(c1, std::fabs((Fp[i] - Fm[i]) / (2 * hF) - (Fp2[i] - Fm2[i]) / hF));
    printf("|dF/dtheta|_inf = %.6e (Richardson self-consistency %.2e)\n", ninf(dFth.data(), N), c1);
    // tangent solve
    derivs dv; allocate_derivs(&dv, N);
    double normres = 0; const double tol = 1e-14 * std::max(1e-3, ninf(rhs.data(), N));
    int st = TPLinearSolve(nvar, n1, n2, n3, v, rhs.data(), dv, 0, 200, tol, &normres);
    printf("tangent solve: status %d, residual norm2 %.3e (tol %.3e)\n", st, normres, tol);
    std::vector<double> tan(dv.d0, dv.d0 + N);
    printf("|dv/dtheta|_inf = %.6e\n", ninf(tan.data(), N));
    // oracle: Newton re-solve at theta +- d
    auto resolve = [&](double val, std::vector<double>& out) {
        derivs vv; allocate_derivs(&vv, N); memcpy(vv.d0, v0.d0, N * sizeof(double));
        *th = val; Newton(nvar, n1, n2, n3, vv, 1e-15, 6);
        CCTK_REAL* Ff = dvector(0, N - 1); derivs uu; allocate_derivs(&uu, N); F_of_v(nvar, n1, n2, n3, vv, Ff, uu);
        printf("   re-solve at theta=%.15g: |F|=%.2e\n", val, ninf(Ff, N));
        out.assign(vv.d0, vv.d0 + N); *th = th0; free_dvector(Ff, 0, N - 1); free_derivs(&uu, N); free_derivs(&vv, N);
    };
    TPID::verbose = 0;
    std::vector<double> a(N), b(N), a2(N), b2(N);
    resolve(th0 + delta, a); resolve(th0 - delta, b); resolve(th0 + delta / 2, a2); resolve(th0 - delta / 2, b2);
    std::vector<double> fd1(N), fd2(N), fdR(N);
    for (int i = 0; i < N; i++) { fd1[i] = (a[i] - b[i]) / (2 * delta); fd2[i] = (a2[i] - b2[i]) / delta; fdR[i] = (4 * fd2[i] - fd1[i]) / 3; }
    double e1 = 0, e2 = 0, eR = 0, nrm = ninf(tan.data(), N);
    for (int i = 0; i < N; i++) { e1 = std::max(e1, std::fabs(tan[i] - fd1[i])); e2 = std::max(e2, std::fabs(tan[i] - fd2[i])); eR = std::max(eR, std::fabs(tan[i] - fdR[i])); }
    printf("(B) |tangent - FD|_inf / |tangent|_inf:  FD(delta) %.3e   FD(delta/2) %.3e   Richardson FD %.3e\n", e1 / nrm, e2 / nrm, eR / nrm);
    MPI_Finalize();
    return 0;
}
