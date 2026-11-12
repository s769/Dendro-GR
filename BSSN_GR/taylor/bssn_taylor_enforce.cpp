// B2 test: jet enforce_bssn_constraints vs production, plus Taylor remainder test with floors active at some points.
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include "parameters.h"
#include "gr.h"
#include "bssn_constraints.h"
#include "taylor_enforce.hpp"
using namespace bssn;
#define KMAX 3
typedef Tay<KMAX> T;
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CHI_FLOOR = 0.1;
    srand(7);
    const int NPT = 2000;
    auto rnd = [] { return 2.0 * rand() / RAND_MAX - 1.0; };
    // coefficient data: v(eps) = c0 + c1 eps + c2 eps^2 + c3 eps^3 for the 14 variables
    std::vector<std::vector<double>> c(NPT, std::vector<double>(14 * (KMAX + 1)));
    for (int p = 0; p < NPT; p++) {
        for (int v = 0; v < 14; v++) for (int j = 0; j <= KMAX; j++) {
            double base = (v == 0 || v == 3 || v == 5) ? 1.0 : 0.0;           // gt diagonal
            if (v == 12) base = 0.12 + 0.3 * (p % 2 ? 1 : 0.2);                 // chi near the floor for half of the points
            if (v == 13) base = 0.25 + 0.5 * (p % 3 ? 1 : 0);                   // alpha near/below floor sometimes
            c[p][v * (KMAX + 1) + j] = (j == 0 ? base : 0.0) + (j == 0 ? 0.1 : 0.05) * rnd();
        }
        // alpha below floor at some points: shift
        if (p % 5 == 0) c[p][13 * (KMAX + 1)] = 0.05;
    }
    // order: gt[6], At[6], chi, alpha
    auto eval_double = [&](int p, double eps, double out[14]) {
        double* u[14]; double val[14];
        for (int v = 0; v < 14; v++) { double e = 1, s = 0; for (int j = 0; j <= KMAX; j++) { s += c[p][v * (KMAX + 1) + j] * e; e *= eps; } val[v] = s; u[v] = &val[v]; }
        double* uiVar[BSSN_NUM_VARS]; for (int i = 0; i < BSSN_NUM_VARS; i++) uiVar[i] = &val[0];
        double gtv[6], Atv[6], chiv = val[12], alv = val[13];
        for (int i = 0; i < 6; i++) { gtv[i] = val[i]; Atv[i] = val[6 + i]; }
        double dgt[6], dAt[6], dchi = chiv, dal = alv;
        // use production routine through a node array
        std::vector<double> buf(BSSN_NUM_VARS);
        std::vector<double*> arr(BSSN_NUM_VARS);
        for (int i = 0; i < BSSN_NUM_VARS; i++) arr[i] = &buf[i];
        const int gtI[6] = {VAR::U_SYMGT0, VAR::U_SYMGT1, VAR::U_SYMGT2, VAR::U_SYMGT3, VAR::U_SYMGT4, VAR::U_SYMGT5};
        const int AtI[6] = {VAR::U_SYMAT0, VAR::U_SYMAT1, VAR::U_SYMAT2, VAR::U_SYMAT3, VAR::U_SYMAT4, VAR::U_SYMAT5};
        for (int i = 0; i < 6; i++) { buf[gtI[i]] = gtv[i]; buf[AtI[i]] = Atv[i]; }
        buf[VAR::U_CHI] = chiv; buf[VAR::U_ALPHA] = alv;
        enforce_bssn_constraints(arr.data(), 0);
        for (int i = 0; i < 6; i++) { dgt[i] = buf[gtI[i]]; dAt[i] = buf[AtI[i]]; }
        for (int i = 0; i < 6; i++) { out[i] = dgt[i]; out[6 + i] = dAt[i]; }
        out[12] = buf[VAR::U_CHI]; out[13] = buf[VAR::U_ALPHA];
    };
    FloorLog lg;
    std::vector<std::vector<double>> jet(NPT, std::vector<double>(14 * (KMAX + 1)));
    double fid = 0;
    for (int p = 0; p < NPT; p++) {
        T gt[6], At[6], chi, alpha;
        for (int i = 0; i < 6; i++) for (int j = 0; j <= KMAX; j++) { gt[i].c[j] = c[p][i * (KMAX + 1) + j]; At[i].c[j] = c[p][(6 + i) * (KMAX + 1) + j]; }
        for (int j = 0; j <= KMAX; j++) { chi.c[j] = c[p][12 * (KMAX + 1) + j]; alpha.c[j] = c[p][13 * (KMAX + 1) + j]; }
        enforce_jet(gt, At, chi, alpha, CHI_FLOOR, &lg);
        for (int v = 0; v < 14; v++) for (int j = 0; j <= KMAX; j++) {
            const T& s = v < 6 ? gt[v] : v < 12 ? At[v - 6] : v == 12 ? chi : alpha;
            jet[p][v * (KMAX + 1) + j] = s.c[j];
        }
        double o[14]; eval_double(p, 0.0, o);
        for (int v = 0; v < 14; v++) fid = std::max(fid, std::fabs(o[v] - jet[p][v * (KMAX + 1)]));
    }
    printf("floor log: points=%ld chi_floor_active=%ld alpha_floor_active=%ld (tangents zeroed there)\n", lg.points, lg.chi_active, lg.alpha_active);
    printf("(A) coefficient-0 fidelity vs production enforce_bssn_constraints: max|diff| = %.3e\n", fid);
    // (B) remainder test; points where the floor status changes between eps=0 and eps are excluded (kink; logged)
    const double epss[] = {0.04, 0.02, 0.01, 0.005};
    double prev[KMAX + 1] = {0, 0, 0, 0};
    printf("(B) remainder (floor status unchanged points only)\n");
    for (int ie = 0; ie < 4; ie++) {
        double eps = epss[ie], rem[KMAX + 1] = {0, 0, 0, 0}; int excl = 0;
        for (int p = 0; p < NPT; p++) {
            double o[14], o0[14]; eval_double(p, eps, o); eval_double(p, 0.0, o0);
            bool cf0 = o0[12] == CHI_FLOOR, cf1 = o[12] == CHI_FLOOR, af0 = o0[13] == CHI_FLOOR, af1 = o[13] == CHI_FLOOR;
            if (cf0 != cf1 || af0 != af1) { excl++; continue; }
            for (int m = 0; m <= KMAX; m++) for (int v = 0; v < 14; v++) {
                double s = 0, e = 1; for (int q = 0; q <= m; q++) { s += jet[p][v * (KMAX + 1) + q] * e; e *= eps; }
                rem[m] = std::max(rem[m], std::fabs(o[v] - s));
            }
        }
        printf("  eps=%.3f excl=%d  %.3e %.3e %.3e %.3e", eps, excl, rem[0], rem[1], rem[2], rem[3]);
        if (ie) { printf("  | slopes"); for (int m = 0; m <= KMAX; m++) printf(" %.2f", std::log2(prev[m] / rem[m])); }
        printf("\n");
        for (int m = 0; m <= KMAX; m++) prev[m] = rem[m];
    }
    MPI_Finalize();
    return 0;
}
