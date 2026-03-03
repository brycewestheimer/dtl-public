# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL C Bindings Library
# =============================================================================

# Source files
set(DTL_C_SOURCES
    dtl_types.cpp
    dtl_status.cpp
    dtl_environment.cpp
    dtl_context.cpp
    dtl_communicator.cpp
    dtl_policies_v2.cpp
    dtl_vector_v2.cpp
    dtl_array_v2.cpp
    dtl_span.cpp
    dtl_tensor_v2.cpp
    dtl_algorithms.cpp
    dtl_rma.cpp
    dtl_mpmd.cpp
    dtl_topology.cpp
    dtl_futures.cpp
)

# Create shared library
add_library(dtl_c SHARED ${DTL_C_SOURCES})

# Set target properties
set_target_properties(dtl_c PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "dtl_c"
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Define DTL_C_EXPORTS for building the library
target_compile_definitions(dtl_c PRIVATE DTL_C_EXPORTS)

# Include directories
target_include_directories(dtl_c
    PUBLIC
        $<BUILD_INTERFACE:${DTL_GENERATED_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link against DTL (header-only)
target_link_libraries(dtl_c PRIVATE dtl)

# MPI support
if(DTL_MPI_AVAILABLE)
    target_link_libraries(dtl_c PRIVATE MPI::MPI_CXX)
    target_compile_definitions(dtl_c PRIVATE DTL_HAS_MPI)
endif()

# CUDA support
if(DTL_ENABLE_CUDA)
    target_compile_definitions(dtl_c PRIVATE DTL_ENABLE_CUDA=1)
endif()

# HIP support
if(DTL_ENABLE_HIP)
    target_compile_definitions(dtl_c PRIVATE DTL_ENABLE_HIP=1)
endif()

# Static library option
option(DTL_BUILD_C_STATIC "Build static C bindings library" OFF)
if(DTL_BUILD_C_STATIC)
    add_library(dtl_c_static STATIC ${DTL_C_SOURCES})
    set_target_properties(dtl_c_static PROPERTIES
        OUTPUT_NAME "dtl_c"
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_definitions(dtl_c_static PRIVATE DTL_C_EXPORTS)
    target_include_directories(dtl_c_static
        PUBLIC
            $<BUILD_INTERFACE:${DTL_GENERATED_INCLUDE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(dtl_c_static PRIVATE dtl)
    if(DTL_MPI_AVAILABLE)
        target_link_libraries(dtl_c_static PRIVATE MPI::MPI_CXX)
        target_compile_definitions(dtl_c_static PRIVATE DTL_HAS_MPI)
    endif()
endif()

# Installation
install(TARGETS dtl_c
    EXPORT DTLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install C headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/dtl/bindings/c/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dtl/bindings/c
    FILES_MATCHING PATTERN "*.h"
)
