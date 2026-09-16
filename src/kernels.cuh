// kernels.cuh - public interface for GPU kernels.
//
// Intentionally free of CUDA-specific keywords so that MSVC cl.exe can
// include this header without nvcc involvement.

#ifndef CUDA_DEMO_SRC_KERNELS_CUH_
#define CUDA_DEMO_SRC_KERNELS_CUH_

#include <cstdint>

namespace cuda_demo {

// Adds two device vectors element-wise: result[i] = lhs[i] + rhs[i].
//
// All three pointers must be valid device allocations of at least `count`
// floats. The kernel is launched asynchronously on the default stream;
// call cudaDeviceSynchronize() (or insert an explicit sync) before
// reading results from the host.
//
// Does nothing and returns immediately when count <= 0.
//
// called from main.cpp's translation unit; clang-tidy only sees kernels.cu
// in isolation and can't tell it's used externally, so it incorrectly
// suggests static/anonymous-namespace linkage, which would break the link.
// NOLINTNEXTLINE(misc-use-internal-linkage): this is the public entry point
void LaunchVectorAdd(const float* lhs, const float* rhs, float* result,
                     std::int32_t count);

}  // namespace cuda_demo

#endif  // CUDA_DEMO_SRC_KERNELS_CUH_
