# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL SHMEM Backend - CMake Configuration
# =============================================================================
# OpenSHMEM-based communication backend for PGAS-style distributed computing.
# Supports symmetric memory allocation and one-sided put/get operations.
# =============================================================================

# Create interface library for SHMEM backend
add_library(dtl_shmem_backend INTERFACE)
add_library(DTL::shmem ALIAS dtl_shmem_backend)

# Include path (project root for backends/ includes)
target_include_directories(dtl_shmem_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Require C++20
target_compile_features(dtl_shmem_backend INTERFACE cxx_std_20)

# Link to main DTL library
target_link_libraries(dtl_shmem_backend INTERFACE DTL::dtl)

# Link SHMEM if enabled
if(DTL_ENABLE_SHMEM)
    # Use our custom FindSHMEM module
    find_package(SHMEM REQUIRED)
    target_link_libraries(dtl_shmem_backend INTERFACE SHMEM::shmem)
    target_compile_definitions(dtl_shmem_backend INTERFACE DTL_ENABLE_SHMEM)
endif()

# Installation
install(TARGETS dtl_shmem_backend
    EXPORT DTLTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES
    shmem_communicator.hpp
    shmem_memory_space.hpp
    shmem_memory_window_impl.hpp
    shmem_rma_adapter.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/backends/shmem
)
