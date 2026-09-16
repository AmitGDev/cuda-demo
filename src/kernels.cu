// kernels.cu - CUDA kernel definitions. Compiled by nvcc only.

#include <cstdint>

#include "kernels.cuh"

namespace cuda_demo {
namespace {

// One thread per element. Bounds-checked so the grid can be oversized.
__global__ void VectorAddKernel(const float* __restrict__ lhs,
                                const float* __restrict__ rhs,
                                float* __restrict__ result,
                                std::int32_t count) {
  const std::int32_t idx = (static_cast<std::int32_t>(blockIdx.x) *
                            static_cast<std::int32_t>(blockDim.x)) +
                           static_cast<std::int32_t>(threadIdx.x);
  if (idx < count) {
    // Indexed device-pointer access is how CUDA kernels address their
    // per-thread element; there is no bounds-checked alternative available
    // inside __global__ code, and the range check above already guards it.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    result[idx] = lhs[idx] + rhs[idx];
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }
}

}  // namespace

void LaunchVectorAdd(const float* lhs, const float* rhs, float* result,
                     std::int32_t count) {
  if (count <= 0) {
    return;
  }

  constexpr std::int32_t kBlockSize = 256;
  const std::int32_t grid_size = (count + kBlockSize - 1) / kBlockSize;

  VectorAddKernel<<<grid_size, kBlockSize>>>(lhs, rhs, result, count);
}

}  // namespace cuda_demo
