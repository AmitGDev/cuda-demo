cmake_minimum_required(VERSION 3.25)

set(PROJECT_NAME_INPUT "cuda_demo" CACHE STRING "Project name")
set(PROJECT_DESCRIPTION_INPUT "CUDA demo" CACHE STRING "Project description")
set(PROJECT_VERSION_INPUT "0.1.0" CACHE STRING "Project version")

project(${PROJECT_NAME_INPUT}
    VERSION ${PROJECT_VERSION_INPUT}
    DESCRIPTION "${PROJECT_DESCRIPTION_INPUT}"
    LANGUAGES CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- C++ standards ----
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- CUDA architecture ----
# CUDA architecture - Quadro RTX 4000 is Turing = sm_75.
# Generates PTX + SASS specifically for this chip; no JIT overhead at launch.
set(CMAKE_CUDA_ARCHITECTURES 75)

# ---- Sources ----
file(GLOB HOST_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

file(GLOB DEVICE_SOURCES CONFIGURE_DEPENDS
    src/*.cu
)

# ---- Executable ----
add_executable(${PROJECT_NAME}
    ${HOST_SOURCES}
    ${DEVICE_SOURCES}
)

# ---- Host compiler options ----
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            /W4
            /permissive-
        >
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Wextra
            -Wpedantic
        >
    )
endif()

# ---- CUDA compiler options ----
# -Xcompiler flags must match the actual CUDA host compiler (see
# CMAKE_CUDA_HOST_COMPILER in the presets), not just CMAKE_CXX_COMPILER,
# so this mirrors the host-compiler branch above rather than reusing it.
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/W4,/permissive-,/wd4211
            --extended-lambda
            --expt-relaxed-constexpr
        >
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=-Wall,-Wextra
            --extended-lambda
            --expt-relaxed-constexpr
        >
    )
endif()

# ---- Include paths ----
target_include_directories(${PROJECT_NAME} PRIVATE
    src
)

# ---- CUDA runtime ----
find_package(CUDAToolkit REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
)

# ---- Output directory ----
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---- compile_commands.json for clangd ----
# clangd only reads from a single CompilationDatabase directory (see .clangd:
# CompilationDatabase: build), so whichever preset was just built also needs
# mirroring to that fixed root-level location. A plain copy is not enough:
# clangd is Clang-based, just like clang-tidy, so the raw NVCC-generated
# entries for .cu files (-Xcompiler, --generate-code, missing --cuda-path,
# etc.) would produce the same false errors in the editor that clang-tidy hit
# in CI before scripts/post_process_compile_commands.py existed. Run that
# same script here so clangd gets the sanitized database, not the raw one.
find_package(Python3 COMPONENTS Interpreter)

set(POST_PROCESS_COMPILE_COMMANDS_SCRIPT
    "${CMAKE_SOURCE_DIR}/scripts/post_process_compile_commands.py"
)

if(Python3_FOUND AND EXISTS "${POST_PROCESS_COMPILE_COMMANDS_SCRIPT}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${Python3_EXECUTABLE}
            "${POST_PROCESS_COMPILE_COMMANDS_SCRIPT}"
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/build"
        COMMENT "Sanitizing and syncing compile_commands.json for clangd"
    )
else()
    # Falls back to an unsanitized copy so the build never breaks over a
    # missing interpreter or script, at the cost of clangd potentially
    # showing the same false CUDA errors clang-tidy used to hit in CI.
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/build/compile_commands.json"
        COMMENT "Syncing compile_commands.json for clangd (unsanitized - Python3 or post_process_compile_commands.py not found)"
    )
endif()

# ---- Configuration summary ----
message(STATUS "")
message(STATUS "${PROJECT_NAME} Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Description:    ${PROJECT_DESCRIPTION}")
message(STATUS "  C++ standard:   C++${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard:  C++${CMAKE_CUDA_STANDARD}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")
