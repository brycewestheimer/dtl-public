# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# FindHIP.cmake - DTL HIP/AMD Backend Discovery
# =============================================================================
#
# This module is intentionally minimal. It serves as a hook point in DTL's
# CMake module path for any project-specific HIP configuration that may be
# needed before the standard HIP discovery runs.
#
# For ROCm 5.0+, HIP discovery is handled by CMake's built-in support:
#
#     find_package(hip REQUIRED)          # ROCm CMake config
#     find_package(HIPToolkit REQUIRED)   # CMake 3.28+ built-in module
#
# The HIP package provides the following imported targets:
#
#   hip::device   - HIP device compilation interface
#   hip::host     - HIP host-side runtime
#
# For ROCm installations, set the following environment or CMake variables:
#
#   HIP_PATH       - Root of HIP installation (e.g., /opt/rocm/hip)
#   ROCM_PATH      - Root of ROCm installation (e.g., /opt/rocm)
#   HIP_PLATFORM   - Target platform: "amd" (default) or "nvidia"
#
# Example CMake invocation:
#
#     cmake .. -DDTL_ENABLE_HIP=ON \
#              -DCMAKE_PREFIX_PATH=/opt/rocm \
#              -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc
#
# Requirements:
#   - ROCm 5.0+ with HIP runtime
#   - A compatible host compiler (GCC 11+ or Clang 15+)
#
# DTL's HIP backend (backends/hip/) provides the same abstractions as the
# CUDA backend through the vendor-agnostic dtl::device:: namespace,
# with compile-time selection between CUDA and HIP via DTL_ENABLE_HIP.
#
# See also:
#   - ROCm documentation: https://rocm.docs.amd.com/
#   - DTL Unified Device Abstraction
# =============================================================================
