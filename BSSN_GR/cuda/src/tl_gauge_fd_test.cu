/**
 * Tier 3.1 — gauge (lapse) TL vs FD harness (host, no full BSSN).
 *
 * Lapse RHS (interior, no radiative BC):
 *   a_rhs = -2 K alpha + λ0 (β·∇α) + σ_KO (KO·α)
 * TL in direction v (K, β fixed; ∇ and KO act on v):
 *   Ja = -2 K v + λ0 (β·∇v) + σ_KO (KO·v)
 *
 * Usage: tl_gauge_fd_test [eps]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int NX = 13;
constexpr int N  = NX * NX * NX;
constexpr double DX = 1e-2 / (NX - 1);
constexpr double LAMBDA0 = 1.0;
constexpr double KO_SIGMA = 0.4;

inline int idx(int i, int j, int k) { return i + NX * (j + NX * k); }

// 4th-order central first derivative (interior); one-sided near boundary.
double d1(const double* a, int i, int j, int k, int dir) {
  const double inv = 1.0 / DX;
  auto at = [&](int ii, int jj, int kk) {
    ii = std::max(0, std::min(NX - 1, ii));
    jj = std::max(0, std::min(NX - 1, jj));
    kk = std::max(0, std::min(NX - 1, kk));
    return a[idx(ii, jj, kk)];
  };
  int i0 = i, j0 = j, k0 = k;
  auto shift = [&](int s) {
    if (dir == 0) return at(i0 + s, j0, k0);
    if (dir == 1) return at(i0, j0 + s, k0);
    return at(i0, j0, k0 + s);
  };
  // Use 2nd-order near edges for simplicity
  if ((dir == 0 && (i <= 1 || i >= NX - 2)) ||
      (dir == 1 && (j <= 1 || j >= NX - 2)) ||
      (dir == 2 && (k <= 1 || k >= NX - 2))) {
    return 0.5 * inv * (shift(1) - shift(-1));
  }
  return inv * ((-shift(2) + 8 * shift(1) - 8 * shift(-1) + shift(-2)) / 12.0);
}

// Simple KO-like 4th difference proxy (not full Dendro KO).
double ko(const double* a, int i, int j, int k) {
  double s = 0.0;
  for (int dir = 0; dir < 3; ++dir) {
    auto shift = [&](int t) {
      int ii = i, jj = j, kk = k;
      if (dir == 0) ii += t;
      if (dir == 1) jj += t;
      if (dir == 2) kk += t;
      ii = std::max(0, std::min(NX - 1, ii));
      jj = std::max(0, std::min(NX - 1, jj));
      kk = std::max(0, std::min(NX - 1, kk));
      return a[idx(ii, jj, kk)];
    };
    s += (shift(2) - 4 * shift(1) + 6 * shift(0) - 4 * shift(-1) + shift(-2)) /
         (DX * DX * DX * DX);
  }
  return s;
}

void lapse_rhs(const double* alpha, const double* K, const double* b0,
               const double* b1, const double* b2, double* a_rhs) {
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = idx(i, j, k);
        const double g0 = d1(alpha, i, j, k, 0);
        const double g1 = d1(alpha, i, j, k, 1);
        const double g2 = d1(alpha, i, j, k, 2);
        a_rhs[p] = -2.0 * K[p] * alpha[p] +
                   LAMBDA0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
                   KO_SIGMA * ko(alpha, i, j, k);
      }
}

void lapse_tl(const double* v, const double* K, const double* b0,
              const double* b1, const double* b2, double* Ja) {
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = idx(i, j, k);
        const double g0 = d1(v, i, j, k, 0);
        const double g1 = d1(v, i, j, k, 1);
        const double g2 = d1(v, i, j, k, 2);
        Ja[p] = -2.0 * K[p] * v[p] +
                LAMBDA0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
                KO_SIGMA * ko(v, i, j, k);
      }
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;

  std::vector<double> alpha(N), K(N), b0(N), b1(N), b2(N), v(N);
  std::vector<double> rhs0(N), rhs_p(N), rhs_m(N), fd(N), tl(N);

  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = idx(i, j, k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        alpha[p] = 1.0 + 0.01 * std::sin(20 * x) * std::cos(15 * y);
        K[p]     = 0.1 * std::cos(10 * x + 5 * z);
        b0[p]    = 0.02 * y;
        b1[p]    = -0.02 * x;
        b2[p]    = 0.01 * z;
        v[p]     = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  lapse_rhs(alpha.data(), K.data(), b0.data(), b1.data(), b2.data(),
            rhs0.data());

  std::vector<double> a_p(N), a_m(N);
  for (int p = 0; p < N; ++p) {
    a_p[p] = alpha[p] + eps * v[p];
    a_m[p] = alpha[p] - eps * v[p];
  }
  lapse_rhs(a_p.data(), K.data(), b0.data(), b1.data(), b2.data(), rhs_p.data());
  lapse_rhs(a_m.data(), K.data(), b0.data(), b1.data(), b2.data(), rhs_m.data());

  for (int p = 0; p < N; ++p) fd[p] = (rhs_p[p] - rhs_m[p]) / (2.0 * eps);
  lapse_tl(v.data(), K.data(), b0.data(), b1.data(), b2.data(), tl.data());

  double max_abs = 0.0, max_rel = 0.0, max_fd = 0.0;
  int worst = 0;
  // Compare on interior only
  for (int k = 2; k < NX - 2; ++k)
    for (int j = 2; j < NX - 2; ++j)
      for (int i = 2; i < NX - 2; ++i) {
        const int p = idx(i, j, k);
        const double diff = std::abs(tl[p] - fd[p]);
        const double denom = std::max(1e-16, std::abs(fd[p]));
        const double rel = diff / denom;
        if (diff > max_abs) {
          max_abs = diff;
          max_rel = rel;
          max_fd  = fd[p];
          worst   = p;
        }
      }

  std::printf("tl_gauge_fd_test eps=%.3e interior max|TL-FD|=%.3e "
              "rel=%.3e (fd=%.3e @p=%d)\n",
              eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE %s (want rel<1e-3 or abs<1e-8)\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 2;
}
