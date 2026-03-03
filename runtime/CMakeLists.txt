# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Runtime Subproject (libdtl_runtime.so)
# =============================================================================
# First-class subproject for runtime services: process-global backend state,
# backend discovery, plugin loading, and connection pooling.
# All DTL consumers link this transitively via the dtl INTERFACE target.
#
# =============================================================================

add_library(dtl_runtime SHARED
    src/runtime_registry.cpp
    src/backend_discovery.cpp
    src/plugin_loader.cpp
    src/connection_pool.cpp
)
add_library(DTL::runtime ALIAS dtl_runtime)

set_target_properties(dtl_runtime PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "dtl_runtime"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_features(dtl_runtime PUBLIC cxx_std_20)

# DTL_RUNTIME_EXPORTS enables __attribute__((visibility("default"))) on
# exported symbols. Only defined when building the DSO itself.
target_compile_definitions(dtl_runtime PRIVATE DTL_RUNTIME_EXPORTS)

target_include_directories(dtl_runtime PUBLIC
    $<BUILD_INTERFACE:${DTL_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# -----------------------------------------------------------------------------
# Backend dependencies (PRIVATE — not leaked to consumers)
# Consumers get backend definitions via the dtl INTERFACE target.
# The runtime DSO only needs them at build time for its .cpp file.
# -----------------------------------------------------------------------------
if(DTL_MPI_AVAILABLE)
    target_link_libraries(dtl_runtime PRIVATE MPI::MPI_CXX)
    target_compile_definitions(dtl_runtime PRIVATE DTL_ENABLE_MPI=1 DTL_HAS_MPI)
endif()

if(DTL_ENABLE_CUDA)
    target_link_libraries(dtl_runtime PRIVATE CUDA::cudart)
    target_compile_definitions(dtl_runtime PRIVATE DTL_ENABLE_CUDA=1)
endif()

if(DTL_ENABLE_HIP)
    target_compile_definitions(dtl_runtime PRIVATE DTL_ENABLE_HIP=1)
endif()

if(DTL_ENABLE_NCCL)
    target_compile_definitions(dtl_runtime PRIVATE DTL_ENABLE_NCCL=1)
endif()

if(DTL_ENABLE_SHMEM)
    target_compile_definitions(dtl_runtime PRIVATE DTL_ENABLE_SHMEM=1)
    if(TARGET SHMEM::shmem)
        target_link_libraries(dtl_runtime PRIVATE SHMEM::shmem)
    endif()
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install(TARGETS dtl_runtime EXPORT DTLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/dtl/runtime
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dtl
    FILES_MATCHING PATTERN "*.hpp"
)
