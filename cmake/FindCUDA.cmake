# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# FindCUDA.cmake - DTL CUDA Backend Discovery
# =============================================================================
#
# This module is intentionally minimal. It serves as a hook point in DTL's
# CMake module path for any project-specific CUDA configuration that may be
# needed before the standard CMake CUDA discovery runs.
#
# Actual CUDA detection is performed by CMake's built-in FindCUDAToolkit
# module, which is invoked in the root CMakeLists.txt via:
#
#     find_package(CUDAToolkit REQUIRED)
#
# The CUDAToolkit package provides the following imported targets used by DTL:
#
#   CUDA::cudart       - CUDA runtime library (linked by DTL::dtl)
#   CUDA::cuda_driver  - CUDA driver API
#   CUDA::nvToolsExt   - NVIDIA Tools Extension (profiling)
#
# If you need to customize CUDA discovery (e.g., for non-standard install
# paths or cross-compilation), you can extend this file. For most users,
# setting CMAKE_CUDA_COMPILER is sufficient:
#
#     cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
#
# Requirements:
#   - CUDA Toolkit 11.4+ (for C++20 host compiler compatibility)
#   - A compatible host compiler (GCC 11+ or Clang 15+)
#
# See also:
#   - CMake documentation: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
#   - DTL build guide in CLAUDE.md for WSL2-specific CUDA setup
# =============================================================================
