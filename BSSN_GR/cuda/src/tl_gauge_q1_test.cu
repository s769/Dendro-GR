/**
 * Tier 3.1 #3 — device PT-stencil lapse TL vs FD on a real q1 unzipped block.
 *
 * Input: binary from DENDRO_DUMP_TL_GAUGE (magic TLG1). Requires nx=ny=nz=13
 * (Dendro pencil block). Interior points only (pw=3); no radiative BC.
 *
 * Usage: tl_gauge_q1_test <dump.bin> [eps]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "device.h"

namespace device {
#include "derivs_point.cuh"
}

namespace {

constexpr int NX = 13;
constexpr int PW = 3;
constexpr int N  = NX * NX * NX;

__global__ void k_lapse_rhs(const double* alpha, const double* K,
                            const double* b0, const double* b1, const double* b2,
                            double dx, double lambda0, double ko_sigma,
                            double* a_rhs) {
  __shared__ double su[N];
  const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = alpha[t];
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / dx;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  a_rhs[p] = -2.0 * K[p] * alpha[p] +
             lambda0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
             ko_sigma * (k0 + k1 + k2);
}

__global__ void k_lapse_tl(const double* v, const double* K, const double* b0,
                           const double* b1, const double* b2, double dx,
                           double lambda0, double ko_sigma, double* Ja) {
  __shared__ double su[N];
  const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = v[t];
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / dx;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  Ja[p] = -2.0 * K[p] * v[p] +
          lambda0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
          ko_sigma * (k0 + k1 + k2);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <tl_gauge_dump.bin> [eps]\n", argv[0]);
    return 1;
  }
  const char* path = argv[1];
  const double eps = (argc > 2) ? std::atof(argv[2]) : 1e-6;

  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "open failed: %s\n", path);
    return 1;
  }
  char magic[4];
  unsigned int nx = 0, ny = 0, nz = 0, bid = 0, bflag = 0;
  double dx = 0, ko = 0, lam0 = 0;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "TLG1", 4) != 0) {
    std::fprintf(stderr, "bad magic in %s\n", path);
    return 1;
  }
  std::fread(&nx, sizeof(nx), 1, f);
  std::fread(&ny, sizeof(ny), 1, f);
  std::fread(&nz, sizeof(nz), 1, f);
  std::fread(&dx, sizeof(dx), 1, f);
  std::fread(&ko, sizeof(ko), 1, f);
  std::fread(&lam0, sizeof(lam0), 1, f);
  std::fread(&bid, sizeof(bid), 1, f);
  std::fread(&bflag, sizeof(bflag), 1, f);
  if (nx != NX || ny != NX || nz != NX) {
    std::fprintf(stderr, "unsupported block %ux%ux%u (need %d^3)\n", nx, ny, nz,
                 NX);
    return 1;
  }
  std::vector<double> h_alpha(N), h_K(N), h_b0(N), h_b1(N), h_b2(N), h_v(N);
  if (std::fread(h_alpha.data(), sizeof(double), N, f) != (size_t)N ||
      std::fread(h_K.data(), sizeof(double), N, f) != (size_t)N ||
      std::fread(h_b0.data(), sizeof(double), N, f) != (size_t)N ||
      std::fread(h_b1.data(), sizeof(double), N, f) != (size_t)N ||
      std::fread(h_b2.data(), sizeof(double), N, f) != (size_t)N) {
    std::fprintf(stderr, "short read in %s\n", path);
    return 1;
  }
  std::fclose(f);

  // Smooth tangent in index space (independent of puncture singularity scale).
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * dx;
        const double y = (j - NX / 2) * dx;
        const double z = (k - NX / 2) * dx;
        h_v[p] = std::exp(-(x * x + y * y + z * z) / (4 * dx * dx));
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
  k_lapse_rhs<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, dx, lam0, ko, d_rhs0);

  std::vector<double> h_ap(N), h_am(N);
  for (int p = 0; p < N; ++p) {
    h_ap[p] = h_alpha[p] + eps * h_v[p];
    h_am[p] = h_alpha[p] - eps * h_v[p];
  }
  cudaMemcpy(d_alpha, h_ap.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lapse_rhs<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, dx, lam0, ko, d_rhsp);
  cudaMemcpy(d_alpha, h_am.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lapse_rhs<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, dx, lam0, ko, d_rhsm);
  k_lapse_tl<<<1, tb>>>(d_v, d_K, d_b0, d_b1, d_b2, dx, lam0, ko, d_tl);
  cudaDeviceSynchronize();

  std::vector<double> rhsp(N), rhsm(N), tl(N);
  cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);

  double max_abs = 0.0, max_rel = 0.0, max_fd = 0.0;
  int worst = 0;
  for (int k = PW; k < NX - PW; ++k)
    for (int j = PW; j < NX - PW; ++j)
      for (int i = PW; i < NX - PW; ++i) {
        const int p = i + NX * (j + NX * k);
        const double fd = (rhsp[p] - rhsm[p]) / (2.0 * eps);
        const double diff = std::abs(tl[p] - fd);
        const double rel = diff / std::max(1e-16, std::abs(fd));
        if (diff > max_abs) {
          max_abs = diff;
          max_rel = rel;
          max_fd  = fd;
          worst   = p;
        }
      }

  std::printf(
      "tl_gauge_q1_test dump=%s blk=%u bflag=%u dx=%.6e ko=%.3g lam0=%.3g "
      "eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n",
      path, bid, bflag, dx, ko, lam0, eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE %s\n", pass ? "PASS" : "FAIL");

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
  return pass ? 0 : 2;
}
