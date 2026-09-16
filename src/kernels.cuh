/*
    kernels.cuh
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
