# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL CPU Backend - CMake Configuration
# =============================================================================
# CPU executor using thread pool for parallel execution.
# =============================================================================

# Create interface library for CPU backend
add_library(dtl_cpu_backend INTERFACE)
add_library(DTL::cpu ALIAS dtl_cpu_backend)

# Include path (backends directory is the root for backend includes)
target_include_directories(dtl_cpu_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Require C++20
target_compile_features(dtl_cpu_backend INTERFACE cxx_std_20)

# Link to main DTL library
target_link_libraries(dtl_cpu_backend INTERFACE DTL::dtl)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(dtl_cpu_backend INTERFACE Threads::Threads)

# Installation
install(TARGETS dtl_cpu_backend
    EXPORT DTLTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES cpu_executor.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/backends/cpu
)
