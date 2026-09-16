/*
    kernels.cu
    Copyright (c) 2026, Amit Gefen

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
*/

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
