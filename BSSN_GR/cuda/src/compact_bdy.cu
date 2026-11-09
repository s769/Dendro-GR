/**
 * @file compact_bdy.cu
 * @brief Device-side compaction of domain-boundary BlockGPU3D entries.
 */
#include "block_gpu.h"
#include "device.h"

namespace device {

namespace {
__global__ void compact_bdy_blocks_kernel(const BlockGPU3D* __restrict__ in,
                                          BlockGPU3D* __restrict__ out,
                                          unsigned* __restrict__ count,
                                          unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (in[i].m_bflag != 0) {
        const unsigned slot = atomicAdd(count, 1u);
        out[slot]           = in[i];
    }
}
}  // namespace

void launch_compact_bdy_blocks(const BlockGPU3D* in, BlockGPU3D* out,
                               unsigned* count, unsigned n) {
    if (n == 0) return;
    const unsigned tpb = 128;
    const unsigned nb  = (n + tpb - 1) / tpb;
    compact_bdy_blocks_kernel<<<nb, tpb>>>(in, out, count, n);
}

}  // namespace device
