# CUDA demo

Cross-platform CUDA C++23 demo for Windows and Linux, tailored for CUDA 13.3 and NVIDIA SM 7.5

**Author:** Amit Gefen
**License:** MIT License

## Overview

A cross-platform Windows/Linux CUDA C++23 demo tailored for CUDA 13.3 and NVIDIA SM 7.5. The project demonstrates a clean separation between host orchestration and CUDA device code, with CUDA runtime calls kept on the host side and kernels isolated in `.cu` source files.

The project uses CMake and Ninja, with clang-format and clang-tidy enforcing consistent code quality. Device memory is managed through RAII, CUDA runtime failures are converted to exceptions, and the host application verifies the GPU computation after execution.

The project is intentionally small and focused on providing a clear reference for the structure, build, and analysis of a modern C++23 CUDA application.

## Prerequisites

* Windows or Linux x64.
* C++23 host compiler:
  * Windows: MSVC (`cl.exe`).
  * Linux: Clang (`clang++`).
* CUDA Toolkit 13.3 with an NVIDIA GPU supporting SM 7.5 (compute capability 7.5).
  * Linux: GCC 12 is used as NVCC's host compiler.
* CMake 3.25 or newer.
* Ninja.
* LLVM 23.1 for `clang-format` and `clang-tidy`.

**Configure and build:**

```powershell
.\build-x64.ps1                          # Debug (default)
.\build-x64.ps1 -Configuration Release   # Release
.\build-x64.ps1 -Clean                   # wipe build\ first, then Debug
```

or the equivalent CMake preset commands:

```powershell
cmake --preset x64-debug
cmake --build --preset x64-debug
```

This produces `build\x64-debug\bin\cuda_demo.exe` (or `build\x64-release\bin\cuda_demo.exe` for Release). Run it to see the full test suite output:

```powershell
.\build\x64-debug\bin\cuda_demo.exe
```

**Formatting and static analysis:**

```powershell
clang-format --dry-run --Werror src/*.cpp src/*.cu src/*.cuh
clang-format -i src/*.cpp src/*.cu src/*.cuh
clang-tidy -p build src/*.cpp
```

## Example Usage
See `main.cpp`.
