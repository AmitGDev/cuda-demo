// main.cpp - host orchestration. Compiled by MSVC cl.exe (toolset v145).
//
// Demonstrates the canonical host/device split:
//   - All CUDA runtime calls (cudaMalloc, cudaMemcpy, etc.) live here.
//   - All __global__ kernels live in kernels.cu.
//   - The boundary is the plain-C++ launcher declared in kernels.cuh.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <print>
#include <stdexcept>
#include <vector>

#include "kernels.cuh"
#include "scope.hpp"

namespace cuda_demo {

// misc-include-cleaner has no symbol-to-header mapping for the CUDA
// toolkit, so it flags every cudaXxx symbol below as "not provided" by
// <cuda_runtime.h> even though that is the correct, complete header for
// all of them. Suppressed locally (not in the shared config) for this
// block only.
// NOLINTBEGIN(misc-include-cleaner)

// 1 M elements, ~4 MB per buffer - large enough to make the host/device
// copies and the kernel launch visible in a profiler, small enough to
// keep the demo instant. Written as a plain literal (not `1 << 20`)
// because a left-shift on a signed operand is flagged by clang-tidy.
constexpr std::int32_t kElementCount = 1'048'576;

// Converts a CUDA runtime failure into an exception carrying the call site,
// instead of terminating the process. Letting failures unwind normally (a)
// keeps main() the single place that decides the process exit code and
// (b) lets RAII guards - see the scope_exit below - release device memory
// before the process ends. Takes plain __FILE__/__LINE__ rather than
// std::source_location (ruled out as the cause of a null-string crash
// here, but kept as the simpler, more portable choice regardless).
static void CheckCuda(cudaError_t result, const char* file = __FILE__,
                      int line = __LINE__) {
  if (result != cudaSuccess) {
    // cudaGetErrorString() is documented to always return a valid string,
    // but that only holds when the headers and the linked cudart agree on
    // the error-code table. Guard against null rather than trust it: on a
    // mismatch this is the difference between a readable message and a
    // std::format crash that hides which CUDA call actually failed.
    const char* message = cudaGetErrorString(result);
    throw std::runtime_error(std::format(
        "CUDA error {} at {}:{} - {}", static_cast<int>(result), file, line,
        message != nullptr ? message
                           : "(cudaGetErrorString returned "
                             "null - possible header/runtime "
                             "version mismatch)"));
  }
}

// Fills host vectors, runs the GPU kernel, copies the result back, and
// verifies it. Throws std::runtime_error on any CUDA failure or on a
// verification mismatch; the caller decides how to report that.
static void RunVectorAdd() {
  std::vector<float> h_a(static_cast<std::size_t>(kElementCount));
  std::vector<float> h_b(static_cast<std::size_t>(kElementCount));
  std::vector<float> h_c(static_cast<std::size_t>(kElementCount), 0.0F);

  for (std::int32_t i = 0; i < kElementCount; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    h_a.at(idx) = static_cast<float>(i);
    h_b.at(idx) = static_cast<float>(kElementCount - i);  // a[i]+b[i]==N
  }

  const std::size_t byte_count =
      static_cast<std::size_t>(kElementCount) * sizeof(float);

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  CheckCuda(cudaMalloc(&device_a, byte_count));
  CheckCuda(cudaMalloc(&device_b, byte_count));
  CheckCuda(cudaMalloc(&device_c, byte_count));

  // Guarantees the three device allocations are freed on every exit path,
  // including a CheckCuda() throw below. cudaFree failures are deliberately
  // discarded since a destructor-driven cleanup can't itself throw. Captured
  // by value: the guard owns a copy of each pointer, matching the ownership
  // a dedicated guard class would have had. `const` documents that this
  // guard is never released early - it always runs.
  const amitgdev::scope_exit device_guard{
      [device_a, device_b, device_c] noexcept {
        static_cast<void>(cudaFree(device_a));
        static_cast<void>(cudaFree(device_b));
        static_cast<void>(cudaFree(device_c));
      }};

  CheckCuda(
      cudaMemcpy(device_a, h_a.data(), byte_count, cudaMemcpyHostToDevice));
  CheckCuda(
      cudaMemcpy(device_b, h_b.data(), byte_count, cudaMemcpyHostToDevice));

  // Kernel launch is asynchronous on the default stream.
  LaunchVectorAdd(device_a, device_b, device_c, kElementCount);

  // Catches launch-configuration errors (surfaced synchronously) before
  // the more expensive synchronize below.
  CheckCuda(cudaGetLastError());
  CheckCuda(cudaDeviceSynchronize());

  CheckCuda(
      cudaMemcpy(h_c.data(), device_c, byte_count, cudaMemcpyDeviceToHost));

  // Every element must equal kElementCount: a[i] + b[i] == N by construction.
  const auto expected = static_cast<float>(kElementCount);
  for (std::int32_t i = 0; i < kElementCount; ++i) {
    const float actual = h_c.at(static_cast<std::size_t>(i));
    if (actual != expected) {
      throw std::runtime_error(std::format(
          "Mismatch at index {}: got {:.1f}, expected {:.1f}", i,
          static_cast<double>(actual), static_cast<double>(expected)));
    }
  }

  std::println("PASSED - {} elements, each = {:.0f}", kElementCount,
               static_cast<double>(expected));
}

// NOLINTEND(misc-include-cleaner)

}  // namespace cuda_demo

int main() noexcept {
  // main() is the single boundary that turns a thrown failure into a
  // process exit code, so every function below it can report errors by
  // throwing rather than checking a return value at every call site.
  // The outer catch(...) exists only because std::println() itself, used
  // to report the failure below, can theoretically throw (a locale-facet
  // lookup inside <format> can raise std::bad_cast on this STL) - without
  // it, an exception from *reporting* the error would itself escape main.
  try {
    try {
      cuda_demo::RunVectorAdd();
      return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
      std::println(stderr, "{}", ex.what());
      return EXIT_FAILURE;
    }
  } catch (...) {
    return EXIT_FAILURE;
  }
}
