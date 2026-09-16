# CUDA demo

Cross-platform CUDA C++23 demo for Windows and Linux, tailored for CUDA 13.4 and NVIDIA SM 7.5

## Overview

A cross-platform Windows/Linux CUDA C++23 demo tailored for CUDA 13.4 and NVIDIA SM 7.5. The project demonstrates a clean separation between host orchestration and CUDA device code, with CUDA runtime calls kept on the host side and kernels isolated in `.cu` source files.

The project uses CMake and Ninja, with clang-format and clang-tidy for code formatting and static analysis. Device memory is managed through RAII, CUDA runtime failures are converted to exceptions, and the host application verifies the GPU computation after execution.

The project is intentionally small and focused on providing a clear reference for the structure, build, and analysis of a modern C++23 CUDA application.

## Requirements

* Windows or Linux x64.
* C++23 host compiler:
  * Windows: MSVC (`cl.exe`).
  * Linux: Clang (`clang++`).
* CUDA Toolkit 13.4 with an NVIDIA GPU supporting SM 7.5 (compute capability 7.5).
  * Linux: GCC 15 is used as NVCC's host compiler.
* CMake 3.25 or newer.
* Ninja.
* LLVM 23.1 for `clang-format` and `clang-tidy`.

## Build and Run

- [Build on Windows](https://github.com/AmitGDev/cpp-ci-template#build-on-windows)
- [Build on Linux](https://github.com/AmitGDev/cpp-ci-template#build-on-linux)

## Code Style

- [C++](https://github.com/AmitGDev/cpp-coding-standard)
- [Python](https://github.com/AmitGDev/cpp-ci-template#python)
- [CMake](https://github.com/AmitGDev/cpp-ci-template#cmake)

## License

MIT License. Copyright © 2026 Amit Gefen.

---

**Last Updated:** 2026-10-25  
**Maintainer:** AmitGDev
