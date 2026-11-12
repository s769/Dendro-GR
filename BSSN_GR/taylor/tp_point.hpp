// B2: jet version of the pointwise part of production TwoPunctures() (psi1, Bowen-York Aij, ADM->BSSN conversion, lapse rule
// alpha = sqrt(chi) as used in production [TPID_REPLACE_LAPSE_WITH_SQRT_CHI=1]).  S = double or Tay<K>.
// Inputs: the interpolated solution value U (jet along the parameter direction) and parameters as jets.
// Not supported (asserted off): TP_Extend_Radius>0 branch (it is 0), other lapse options, multiply_old_lapse, conformal_state>0.
#pragma once
#include "taylor.hpp"
template <class S> struct TPPar { S mp, mm, b; S Pp[3], Pm[3], Sp[3], Sm[3]; };

template <class S>
inline void tp_point(const TPPar<S>& P, S xx, S yy, S zz, S U, S out[15] /* gt0..5, At0..5, K, chi, alpha */, double eps4 = 1e-24, double tiny = 1e-15) {
    auto val = [](const S& s) { return s.c[0]; };
    S rp2 = (xx - P.b) * (xx - P.b) + yy * yy + zz * zz, rm2 = (xx + P.b) * (xx + P.b) + yy * yy + zz * zz;
    // psi1 with epsilon-regularised radii: r = (r^4 + eps^4)^(1/4)
    S rp = pow(rp2 * rp2 + eps4, 0.25), rm = pow(rm2 * rm2 + eps4, 0.25);
    if (val(rp) < tiny) rp = S(tiny);
    if (val(rm) < tiny) rm = S(tiny);
    S psi1 = 1.0 + 0.5 * P.mp / rp + 0.5 * P.mm / rm + U;
    // Bowen-York Aij (BY_Aijofxyz: r2 = sqrt(r2^2 + eps^4))
    S r2p = sqrt(rp2 * rp2 + eps4), r2m = sqrt(rm2 * rm2 + eps4);
    if (val(r2p) < tiny * tiny) r2p = S(tiny * tiny);
    if (val(r2m) < tiny * tiny) r2m = S(tiny * tiny);
    S rpl = sqrt(r2p), rmi = sqrt(r2m), r3p = rpl * r2p, r3m = rmi * r2m;
    S np[3] = {(xx - P.b) / rpl, yy / rpl, zz / rpl}, nm[3] = {(xx + P.b) / rmi, yy / rmi, zz / rmi};
    S nPp = np[0] * P.Pp[0] + np[1] * P.Pp[1] + np[2] * P.Pp[2];
    S nPm = nm[0] * P.Pm[0] + nm[1] * P.Pm[1] + nm[2] * P.Pm[2];
    S nSp[3] = {np[1] * P.Sp[2] - np[2] * P.Sp[1], np[2] * P.Sp[0] - np[0] * P.Sp[2], np[0] * P.Sp[1] - np[1] * P.Sp[0]};
    S nSm[3] = {nm[1] * P.Sm[2] - nm[2] * P.Sm[1], nm[2] * P.Sm[0] - nm[0] * P.Sm[2], nm[0] * P.Sm[1] - nm[1] * P.Sm[0]};
    S Aij[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        Aij[i][j] = 1.5 * (P.Pp[i] * np[j] + P.Pp[j] * np[i] + nPp * np[i] * np[j]) / r2p +
                    1.5 * (P.Pm[i] * nm[j] + P.Pm[j] * nm[i] + nPm * nm[i] * nm[j]) / r2m -
                    3.0 * (nSp[i] * np[j] + nSp[j] * np[i]) / r3p - 3.0 * (nSm[i] * nm[j] + nSm[j] * nm[i]) / r3m;
        if (i == j) Aij[i][j] = Aij[i][j] - 1.5 * (nPp / r2p + nPm / r2m);
    }
    S psi4 = pow(psi1, 4), psim2 = pow(psi1, -2);
    S gd[3][3], Kd[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { gd[i][j] = S(0.0); if (i == j) gd[i][j] = psi4; Kd[i][j] = Aij[i][j] * psim2; }
    // adm2bssn.h
    S det = gd[0][0] * gd[1][1] * gd[2][2] - gd[0][0] * gd[1][2] * gd[1][2] - gd[0][1] * gd[0][1] * gd[2][2] + 2.0 * gd[0][1] * gd[0][2] * gd[1][2] - gd[0][2] * gd[0][2] * gd[1][1];
    S idet = 1.0 / det, gu[3][3];
    gu[0][0] = idet * (gd[1][1] * gd[2][2] - gd[1][2] * gd[1][2]);
    gu[0][1] = idet * (-gd[0][1] * gd[2][2] + gd[0][2] * gd[1][2]);
    gu[0][2] = idet * (gd[0][1] * gd[1][2] - gd[0][2] * gd[1][1]);
    gu[1][1] = idet * (gd[0][0] * gd[2][2] - gd[0][2] * gd[0][2]);
    gu[1][2] = idet * (-gd[0][0] * gd[1][2] + gd[0][1] * gd[0][2]);
    gu[2][2] = idet * (gd[0][0] * gd[1][1] - gd[0][1] * gd[0][1]);
    gu[1][0] = gu[0][1]; gu[2][0] = gu[0][2]; gu[2][1] = gu[1][2];
    S chi = pow(idet, 0.3333333333333333);
    S trK = gu[0][0] * Kd[0][0] + 2.0 * gu[0][1] * Kd[0][1] + 2.0 * gu[0][2] * Kd[0][2] + gu[1][1] * Kd[1][1] + 2.0 * gu[1][2] * Kd[1][2] + gu[2][2] * Kd[2][2];
    const int ii[6] = {0, 0, 0, 1, 1, 2}, jj[6] = {0, 1, 2, 1, 2, 2};
    for (int m = 0; m < 6; m++) {
        out[m] = chi * gd[ii[m]][jj[m]];
        out[6 + m] = chi * (Kd[ii[m]][jj[m]] - 0.3333333333333333 * gd[ii[m]][jj[m]] * trK);
    }
    out[12] = trK; out[13] = chi; out[14] = sqrt(chi);
}
