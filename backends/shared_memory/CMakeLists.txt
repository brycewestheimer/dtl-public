# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Shared Memory Backend - CMake Configuration
# =============================================================================
# Shared memory communicator for intra-node communication (stub implementation).
# =============================================================================

# Create interface library for shared memory backend
add_library(dtl_shared_memory_backend INTERFACE)
add_library(DTL::shared_memory ALIAS dtl_shared_memory_backend)

# Include path (project root for backends/ includes)
target_include_directories(dtl_shared_memory_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Require C++20
target_compile_features(dtl_shared_memory_backend INTERFACE cxx_std_20)

# Link to main DTL library
target_link_libraries(dtl_shared_memory_backend INTERFACE DTL::dtl)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(dtl_shared_memory_backend INTERFACE Threads::Threads)

# Installation
install(TARGETS dtl_shared_memory_backend
    EXPORT DTLTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES
    shared_memory_communicator.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/backends/shared_memory
)
