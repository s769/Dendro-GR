/**
 * Tier 3.1 — radiative BC linearization for lapse TL vs FD (device PT).
 *
 * Production radiative face (alpha: falloff=1, asymptotic=1):
 *   a_rhs = -r^{-1} (x ∂xα + y ∂yα + z ∂zα + 1·(α − 1))
 * TL (α → v, grads on v; r,x,y,z fixed):
 *   Ja    = -r^{-1} (x ∂xv + y ∂yv + z ∂zv + v)
 *
 * Domain: NX^3 block with LEFT face (i=PW) treated as radiative.
 * Interior still uses gauge RHS; face overwrites with rad BC.
 *
 * Usage: tl_gauge_radbc_test [eps]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device.h"

namespace device {
#include "derivs_point.cuh"
}

namespace {

constexpr int NX = 13;
constexpr int PW = 3;
constexpr int N  = NX * NX * NX;
constexpr double DX = 0.5;  // large enough that face is far from origin
constexpr double LAMBDA0 = 1.0;
constexpr double KO_SIGMA = 0.4;
constexpr double FALL = 1.0;
constexpr double ASYM = 1.0;
// Block origin so LEFT face (i=PW) sits at x ≈ -R
constexpr double X0 = -20.0;
constexpr double Y0 = -3.0 * DX;
constexpr double Z0 = -3.0 * DX;

__device__ __forceinline__ void face_xyz(int i, int j, int k, double* x,
                                         double* y, double* z) {
  *x = X0 + i * DX;
  *y = Y0 + j * DX;
  *z = Z0 + k * DX;
}

__global__ void k_lapse_rhs_rad(const double* alpha, const double* K,
                                const double* b0, const double* b1,
                                const double* b2, double* a_rhs) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = alpha[t];
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);

  // Interior gauge RHS
  a_rhs[p] = -2.0 * K[p] * alpha[p] +
             LAMBDA0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
             KO_SIGMA * (k0 + k1 + k2);

  // LEFT radiative face overwrites (si == PW)
  if (i == PW) {
    double x, y, z;
    face_xyz(i, j, k, &x, &y, &z);
    const double inv_r = 1.0 / sqrt(x * x + y * y + z * z);
    a_rhs[p] =
        -inv_r * (x * g0 + y * g1 + z * g2 + FALL * (alpha[p] - ASYM));
  }
}

__global__ void k_lapse_tl_rad(const double* v, const double* K, const double* b0,
                               const double* b1, const double* b2, double* Ja) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = v[t];
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);

  Ja[p] = -2.0 * K[p] * v[p] +
          LAMBDA0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
          KO_SIGMA * (k0 + k1 + k2);

  if (i == PW) {
    double x, y, z;
    face_xyz(i, j, k, &x, &y, &z);
    const double inv_r = 1.0 / sqrt(x * x + y * y + z * z);
    Ja[p] = -inv_r * (x * g0 + y * g1 + z * g2 + FALL * v[p]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;

  std::vector<double> h_alpha(N), h_K(N), h_b0(N), h_b1(N), h_b2(N), h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        double x, y, z;
        // host copy of face_xyz
        x = X0 + i * DX;
        y = Y0 + j * DX;
        z = Z0 + k * DX;
        h_alpha[p] = 1.0 + 0.02 * std::sin(0.3 * x) * std::cos(0.2 * y);
        h_K[p]     = 0.05 * std::cos(0.15 * x + 0.1 * z);
        h_b0[p]    = 0.01 * y;
        h_b1[p]    = -0.01 * x;
        h_b2[p]    = 0.005 * z;
        h_v[p]     = std::exp(-(y * y + z * z) / (8 * DX * DX)) *
                 (0.5 + 0.5 * std::cos(0.25 * x));
      }

  double *d_alpha, *d_K, *d_b0, *d_b1, *d_b2, *d_v;
  double *d_rhs0, *d_rhsp, *d_rhsm, *d_tl;
  cudaMalloc(&d_alpha, N * sizeof(double));
  cudaMalloc(&d_K, N * sizeof(double));
  cudaMalloc(&d_b0, N * sizeof(double));
  cudaMalloc(&d_b1, N * sizeof(double));
  cudaMalloc(&d_b2, N * sizeof(double));
  cudaMalloc(&d_v, N * sizeof(double));
  cudaMalloc(&d_rhs0, N * sizeof(double));
  cudaMalloc(&d_rhsp, N * sizeof(double));
  cudaMalloc(&d_rhsm, N * sizeof(double));
  cudaMalloc(&d_tl, N * sizeof(double));

  cudaMemcpy(d_alpha, h_alpha.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  k_lapse_rhs_rad<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, d_rhs0);

  std::vector<double> h_ap(N), h_am(N);
  for (int p = 0; p < N; ++p) {
    h_ap[p] = h_alpha[p] + eps * h_v[p];
    h_am[p] = h_alpha[p] - eps * h_v[p];
  }
  cudaMemcpy(d_alpha, h_ap.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lapse_rhs_rad<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, d_rhsp);
  cudaMemcpy(d_alpha, h_am.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lapse_rhs_rad<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, d_rhsm);
  k_lapse_tl_rad<<<1, tb>>>(d_v, d_K, d_b0, d_b1, d_b2, d_tl);
  cudaDeviceSynchronize();

  std::vector<double> rhsp(N), rhsm(N), tl(N);
  cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);

  auto report = [&](const char* tag, bool face_only) {
    double max_abs = 0.0, max_rel = 0.0, max_fd = 0.0;
    int worst = 0, npts = 0;
    for (int k = PW; k < NX - PW; ++k)
      for (int j = PW; j < NX - PW; ++j)
        for (int i = PW; i < NX - PW; ++i) {
          if (face_only && i != PW) continue;
          if (!face_only && i == PW) continue;
          const int p = i + NX * (j + NX * k);
          const double fd = (rhsp[p] - rhsm[p]) / (2.0 * eps);
          const double diff = std::abs(tl[p] - fd);
          const double rel = diff / std::max(1e-16, std::abs(fd));
          ++npts;
          if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_fd  = fd;
            worst   = p;
          }
        }
    std::printf(
        "tl_gauge_radbc_test %s eps=%.3e n=%d max|TL-FD|=%.3e rel=%.3e "
        "(fd=%.3e @p=%d)\n",
        tag, eps, npts, max_abs, max_rel, max_fd, worst);
    const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
    std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
    return pass;
  };

  const int pass_face = report("FACE", true);
  const int pass_int  = report("INTERIOR", false);

  cudaFree(d_alpha);
  cudaFree(d_K);
  cudaFree(d_b0);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_v);
  cudaFree(d_rhs0);
  cudaFree(d_rhsp);
  cudaFree(d_rhsm);
  cudaFree(d_tl);
  return (pass_face && pass_int) ? 0 : 2;
}
