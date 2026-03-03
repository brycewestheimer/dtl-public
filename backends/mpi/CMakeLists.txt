# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL MPI Backend - CMake Configuration
# =============================================================================
# MPI-based communicator implementation.
# =============================================================================

# Create interface library for MPI backend
add_library(dtl_mpi_backend INTERFACE)
add_library(DTL::mpi ALIAS dtl_mpi_backend)

# Include path (project root for backends/ includes)
target_include_directories(dtl_mpi_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Require C++20
target_compile_features(dtl_mpi_backend INTERFACE cxx_std_20)

# Link to main DTL library
target_link_libraries(dtl_mpi_backend INTERFACE DTL::dtl)

# Link MPI if available
if(DTL_MPI_AVAILABLE)
    target_link_libraries(dtl_mpi_backend INTERFACE MPI::MPI_CXX)
    target_compile_definitions(dtl_mpi_backend INTERFACE DTL_ENABLE_MPI=1)
endif()

# Installation
install(TARGETS dtl_mpi_backend
    EXPORT DTLTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES
    mpi_communicator.hpp
    mpi_comm_adapter.hpp
    mpi_lifecycle.hpp
    mpi_environment.hpp
    mpi_rma_adapter.hpp
    mpi_topology.hpp
    mpi_window.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/backends/mpi
)
