// B2: Taylor-mode (jet) version of the production enforce_bssn_constraints (include/bssn_constraints.h).
// det(gt)=1 rescale and tr(At)=0 projection are smooth => differentiated exactly with the jet type.
// The chi / alpha floors are piecewise: where the floor is ACTIVE (value < CHI_FLOOR) the output is the constant
// CHI_FLOOR, so all tangent coefficients (order >= 1) are zeroed ("frozen mask"); activity is counted so a run can
// report where the floor touches the solution (stop-and-ask condition if it affects the waveform).
#pragma once
#include "taylor.hpp"
struct FloorLog { long chi_active = 0, alpha_active = 0, points = 0; };
template <class S>
inline void enforce_jet(S gt[6], S At[6], S& chi, S& alpha, double chi_floor, FloorLog* lg) {
    // gt/At packed as (00,01,02,11,12,22)
    auto det = [&](const S* g) {
        return g[0] * (g[3] * g[5] - g[4] * g[4]) - g[1] * g[1] * g[5] + 2.0 * g[1] * g[2] * g[4] - g[2] * g[2] * g[3];
    };
    S d = det(gt);
    S f = 1.0 / pow(d, 1.0 / 3.0);
    for (int i = 0; i < 6; i++) gt[i] = gt[i] * f;
    d = det(gt);
    S id = 1.0 / d, gu[6];
    gu[0] = id * (gt[3] * gt[5] - gt[4] * gt[4]);
    gu[1] = id * (-gt[1] * gt[5] + gt[2] * gt[4]);
    gu[2] = id * (gt[1] * gt[4] - gt[2] * gt[3]);
    gu[3] = id * (gt[0] * gt[5] - gt[2] * gt[2]);
    gu[4] = id * (-gt[0] * gt[4] + gt[1] * gt[2]);
    gu[5] = id * (gt[0] * gt[3] - gt[1] * gt[1]);
    S tr = (At[0] * gu[0] + At[3] * gu[3] + At[5] * gu[5] + 2.0 * (At[1] * gu[1] + At[2] * gu[2] + At[4] * gu[4])) * (1.0 / 3.0);
    for (int i = 0; i < 6; i++) At[i] = At[i] - tr * gt[i];
    if (lg) lg->points++;
    if (chi.c[0] < chi_floor) { chi = S(chi_floor); if (lg) lg->chi_active++; }
    if (alpha.c[0] < chi_floor) { alpha = S(chi_floor); if (lg) lg->alpha_active++; }
}
