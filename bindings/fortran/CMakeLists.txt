# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Fortran Bindings
# =============================================================================
# Provides native Fortran module wrapping the DTL C API via ISO_C_BINDING.
# Enable with -DDTL_BUILD_FORTRAN=ON (requires Fortran compiler)
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# -----------------------------------------------------------------------------
# Fortran Module Library
# -----------------------------------------------------------------------------

# Create the Fortran module library
# Order matters: modules must be compiled before modules that USE them.
add_library(dtl_fortran
    dtl_core.f90
    dtl_environment.f90
    dtl_context.f90
    dtl_containers.f90
    dtl_communication.f90
    dtl_algorithms.f90
    dtl_rma.f90
    dtl_mpmd.f90
    dtl_futures.f90
    dtl_policies.f90
    dtl_topology.f90
    dtl_helpers.f90
    dtl.f90
)

# Set Fortran standard (2008 for advanced features)
set_target_properties(dtl_fortran PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod
)

# Link against C bindings library
target_link_libraries(dtl_fortran PUBLIC dtl_c)

# Include module directory for consumers
target_include_directories(dtl_fortran PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dtl/fortran>
)

# MPI Fortran support (if available)
if(DTL_MPI_AVAILABLE)
    find_package(MPI COMPONENTS Fortran QUIET)
    if(MPI_Fortran_FOUND)
        target_link_libraries(dtl_fortran PUBLIC MPI::MPI_Fortran)
        message(STATUS "  MPI Fortran bindings: Available")
    else()
        message(STATUS "  MPI Fortran bindings: Not found (MPI C only)")
    endif()
endif()

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------

if(DTL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

if(DTL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

# Install the library
install(TARGETS dtl_fortran
    EXPORT DTLFortranTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install Fortran module files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dtl/fortran
    FILES_MATCHING PATTERN "*.mod"
)

# Install source for reference
install(FILES
    dtl.f90
    dtl_core.f90
    dtl_environment.f90
    dtl_context.f90
    dtl_containers.f90
    dtl_communication.f90
    dtl_algorithms.f90
    dtl_helpers.f90
    dtl_rma.f90
    dtl_mpmd.f90
    dtl_futures.f90
    dtl_policies.f90
    dtl_topology.f90
    DESTINATION ${CMAKE_INSTALL_DATADIR}/dtl/fortran
)

# Export targets
install(EXPORT DTLFortranTargets
    FILE DTLFortranTargets.cmake
    NAMESPACE DTL::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DTL
)
