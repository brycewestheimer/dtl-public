# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL HIP Backend - CMake Configuration
# =============================================================================
# HIP-based memory spaces, executors, and device algorithms for AMD GPUs.
# =============================================================================

# Create interface library for HIP backend
add_library(dtl_hip_backend INTERFACE)
add_library(DTL::hip ALIAS dtl_hip_backend)

# Include path (project root for backends/ includes)
target_include_directories(dtl_hip_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Require C++20
target_compile_features(dtl_hip_backend INTERFACE cxx_std_20)

# Link to main DTL library
target_link_libraries(dtl_hip_backend INTERFACE DTL::dtl)

# Link HIP runtime and rocThrust if enabled
if(DTL_ENABLE_HIP)
    find_package(hip QUIET)
    if(hip_FOUND)
        target_link_libraries(dtl_hip_backend INTERFACE hip::device)
        target_compile_definitions(dtl_hip_backend INTERFACE DTL_ENABLE_HIP)
    endif()

    # rocThrust provides Thrust API for HIP (required for hip_algorithms.hpp)
    find_package(rocthrust QUIET)
    if(rocthrust_FOUND)
        target_link_libraries(dtl_hip_backend INTERFACE roc::rocthrust)
    endif()
endif()

# Installation
install(TARGETS dtl_hip_backend
    EXPORT DTLTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES
    hip_memory_space.hpp
    hip_memory_transfer.hpp
    hip_executor.hpp
    hip_event.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/backends/hip
)
