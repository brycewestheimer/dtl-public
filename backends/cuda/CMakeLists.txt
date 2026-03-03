# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL CUDA Backend - CMake Configuration
# =============================================================================
# CUDA-based memory spaces and executors.
# =============================================================================

# Create interface library for CUDA backend
add_library(dtl_cuda_backend INTERFACE)
add_library(DTL::cuda ALIAS dtl_cuda_backend)

# Include path (project root for backends/ includes)
target_include_directories(dtl_cuda_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Require C++20
target_compile_features(dtl_cuda_backend INTERFACE cxx_std_20)

# Link to main DTL library
target_link_libraries(dtl_cuda_backend INTERFACE DTL::dtl)

# Link CUDA runtime if enabled
if(DTL_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(dtl_cuda_backend INTERFACE CUDA::cudart)
    target_compile_definitions(dtl_cuda_backend INTERFACE DTL_ENABLE_CUDA)
endif()

# Installation
install(TARGETS dtl_cuda_backend
    EXPORT DTLTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES
    cuda_memory_space.hpp
    cuda_managed_memory_space.hpp
    cuda_executor.hpp
    cuda_event.hpp
    cuda_memory_transfer.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/backends/cuda
)
