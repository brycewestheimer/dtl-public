# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL (Distributed Template Library) - Root CMake Configuration
# =============================================================================
# Version: 0.1.0
# C++20 header-only template library for distributed computing
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# -----------------------------------------------------------------------------
# Project Definition
# -----------------------------------------------------------------------------
project(DTL
    VERSION 0.1.0
    DESCRIPTION "C++20 header-first template library for distributed and heterogeneous computing"
    HOMEPAGE_URL "https://github.com/brycewestheimer/dtl-public"
    LANGUAGES CXX
)

set(DTL_VERSION_PRERELEASE "alpha.1")
set(DTL_VERSION_FULL "${PROJECT_VERSION}-${DTL_VERSION_PRERELEASE}")
set(DTL_VERSION_PYTHON "0.1.0a1")

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory.")
endif()

# -----------------------------------------------------------------------------
# CMake Module Path
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -----------------------------------------------------------------------------
# C++20 Requirements
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Build Options
# -----------------------------------------------------------------------------
option(DTL_ENABLE_CUDA       "Enable CUDA backend support"              OFF)
option(DTL_ENABLE_HIP        "Enable HIP/AMD backend support"           OFF)
option(DTL_ENABLE_NCCL       "Enable NCCL communicator support"         OFF)
option(DTL_ENABLE_SHMEM      "Enable OpenSHMEM backend support"         OFF)
option(DTL_ENABLE_MPI        "Enable MPI backend support"               ON)
option(DTL_BUILD_TESTS       "Build test suite"                         ON)
option(DTL_BUILD_INTEGRATION_TESTS "Build integration tests (MPI/CUDA/SHMEM)" OFF)
option(DTL_USE_SYSTEM_GTEST  "Prefer system-installed GoogleTest when available" ON)
option(DTL_BUILD_BENCHMARKS  "Build performance benchmarks"             OFF)
option(DTL_BUILD_EXAMPLES    "Build example programs"                   ON)
option(DTL_BUILD_DOCS        "Build documentation"                      OFF)
option(DTL_ENABLE_SANITIZERS "Enable address/undefined sanitizers"      OFF)
option(DTL_ENABLE_COVERAGE   "Enable code coverage instrumentation"     OFF)
option(DTL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"       OFF)
option(DTL_BUILD_C_BINDINGS   "Build C ABI bindings"                    OFF)
option(DTL_BUILD_PYTHON       "Build Python bindings"                   OFF)
option(DTL_BUILD_FORTRAN      "Build Fortran bindings (requires Fortran compiler)" OFF)
option(DTL_BUILD_FUZZ_TARGETS "Build fuzz testing targets"              OFF)
option(DTL_USE_WORK_STEALING  "Use work-stealing thread pool instead of mutex-based pool" ON)

# -----------------------------------------------------------------------------
# Optional Serialization Adapter Options
# -----------------------------------------------------------------------------
option(DTL_ENABLE_CEREAL      "Enable Cereal serialization adapter (requires Cereal)" OFF)
option(DTL_ENABLE_BOOST_SERIALIZATION "Enable Boost.Serialization adapter" OFF)

# -----------------------------------------------------------------------------
# C ABI Instantiation Budgeting Options
# See docs/capabilities/instantiation_budget.md for details
# -----------------------------------------------------------------------------
option(DTL_C_ABI_MINIMAL         "Minimal C ABI matrix (f32/f64 × block × host × seq)" OFF)
option(DTL_C_ABI_FULL_MATRIX     "Full C ABI matrix (all combinations)"                OFF)
option(DTL_C_ABI_ENABLE_DEVICE   "Enable device_only placement in C ABI"               OFF)
option(DTL_C_ABI_ENABLE_PINNED   "Enable pinned_host placement in C ABI"               OFF)
option(DTL_C_ABI_ENABLE_BLOCK_CYCLIC "Enable block-cyclic partition in C ABI"          OFF)
option(DTL_C_ABI_ENABLE_HASH     "Enable hash partition in C ABI"                      OFF)
option(DTL_C_ABI_ENABLE_HALF     "Enable f16/bf16 in C ABI"                            OFF)
option(DTL_C_ABI_ENABLE_ASYNC    "Enable async execution in C ABI"                     OFF)

# -----------------------------------------------------------------------------
# Include CMake Modules
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

set(DTL_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY
    "${DTL_GENERATED_INCLUDE_DIR}/dtl/generated"
    "${CMAKE_CURRENT_BINARY_DIR}/docs"
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/dtl_version_config.hpp.in"
    "${DTL_GENERATED_INCLUDE_DIR}/dtl/generated/version_config.hpp"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/dtl_version_config.h.in"
    "${DTL_GENERATED_INCLUDE_DIR}/dtl/generated/version_config.h"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile"
    @ONLY
)

# Project-specific modules
include(CompilerWarnings)
include(CXX20Concepts)

if(DTL_ENABLE_SANITIZERS)
    include(Sanitizers)
endif()

if(DTL_ENABLE_COVERAGE)
    include(CodeCoverage)

    # Coverage data generation should not depend on lcov/genhtml availability.
    # We always enable compiler instrumentation for GNU/Clang so gcovr-based
    # CI reports work on hosted runners.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage -O0)
        add_link_options(--coverage -fprofile-arcs -ftest-coverage)
        message(STATUS "Coverage instrumentation flags enabled (GNU/Clang).")
    else()
        message(WARNING "Coverage instrumentation currently supports GNU/Clang only.")
    endif()
endif()

# -----------------------------------------------------------------------------
# Find Dependencies
# -----------------------------------------------------------------------------
# MPI is optional and can be disabled explicitly even if installed.
if(DTL_ENABLE_MPI)
    find_package(MPI)
    if(NOT MPI_FOUND)
        message(STATUS "MPI not found - building in single-process mode")
        set(DTL_MPI_AVAILABLE OFF)
    else()
        message(STATUS "MPI found - distributed mode available")
        set(DTL_MPI_AVAILABLE ON)
    endif()
else()
    message(STATUS "MPI disabled (DTL_ENABLE_MPI=OFF) - building in single-process mode")
    set(DTL_MPI_AVAILABLE OFF)
endif()

# Optional GPU backends
if(DTL_ENABLE_CUDA)
    include(FindCUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(DTL_ENABLE_HIP)
    include(FindHIP)
endif()

if(DTL_ENABLE_NCCL)
    include(FindNCCL)
    find_package(NCCL REQUIRED)
endif()

if(DTL_ENABLE_SHMEM)
    include(FindSHMEM)
endif()

# Optional serialization library adapters
if(DTL_ENABLE_CEREAL)
    find_package(Cereal QUIET)
    if(Cereal_FOUND)
        message(STATUS "Cereal found - serialization adapter available")
        set(DTL_CEREAL_AVAILABLE ON)
    else()
        message(WARNING "Cereal requested but not found. Adapter will be disabled.")
        set(DTL_CEREAL_AVAILABLE OFF)
    endif()
else()
    set(DTL_CEREAL_AVAILABLE OFF)
endif()

if(DTL_ENABLE_BOOST_SERIALIZATION)
    find_package(BoostSerialization QUIET)
    if(BoostSerialization_FOUND)
        message(STATUS "Boost.Serialization found - serialization adapter available")
        set(DTL_BOOST_SERIALIZATION_AVAILABLE ON)
    else()
        message(WARNING "Boost.Serialization requested but not found. Adapter will be disabled.")
        set(DTL_BOOST_SERIALIZATION_AVAILABLE OFF)
    endif()
else()
    set(DTL_BOOST_SERIALIZATION_AVAILABLE OFF)
endif()

# -----------------------------------------------------------------------------
# DTL Header-Only Library Target
# -----------------------------------------------------------------------------
add_library(dtl INTERFACE)
add_library(DTL::dtl ALIAS dtl)

target_include_directories(dtl INTERFACE
    $<BUILD_INTERFACE:${DTL_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(dtl INTERFACE cxx_std_20)

# Link MPI if available
if(DTL_MPI_AVAILABLE)
    target_link_libraries(dtl INTERFACE MPI::MPI_CXX)
    target_compile_definitions(dtl INTERFACE DTL_HAS_MPI DTL_ENABLE_MPI=1)
endif()

# Conditional GPU support
if(DTL_ENABLE_CUDA)
    target_compile_definitions(dtl INTERFACE DTL_ENABLE_CUDA)
    target_link_libraries(dtl INTERFACE CUDA::cudart)
endif()

if(DTL_ENABLE_HIP)
    target_compile_definitions(dtl INTERFACE DTL_ENABLE_HIP)
endif()

if(DTL_ENABLE_NCCL)
    target_compile_definitions(dtl INTERFACE DTL_ENABLE_NCCL)
endif()

if(DTL_ENABLE_SHMEM)
    target_compile_definitions(dtl INTERFACE DTL_ENABLE_SHMEM)
endif()

if(DTL_USE_WORK_STEALING)
    target_compile_definitions(dtl INTERFACE DTL_USE_WORK_STEALING=1)
endif()

# Optional serialization adapters
if(DTL_CEREAL_AVAILABLE)
    target_compile_definitions(dtl INTERFACE DTL_HAS_CEREAL=1)
    target_link_libraries(dtl INTERFACE Cereal::Cereal)
endif()

if(DTL_BOOST_SERIALIZATION_AVAILABLE)
    target_compile_definitions(dtl INTERFACE DTL_HAS_BOOST_SERIALIZATION=1)
    target_link_libraries(dtl INTERFACE BoostSerialization::BoostSerialization)
endif()

# -----------------------------------------------------------------------------
# Backend Libraries
# -----------------------------------------------------------------------------
add_subdirectory(backends/mpi)
add_subdirectory(backends/cpu)
add_subdirectory(backends/shared_memory)

if(DTL_ENABLE_CUDA)
    add_subdirectory(backends/cuda)
endif()

if(DTL_ENABLE_HIP)
    add_subdirectory(backends/hip)
endif()

if(DTL_ENABLE_NCCL)
    add_subdirectory(backends/nccl)
endif()

if(DTL_ENABLE_SHMEM)
    add_subdirectory(backends/shmem)
endif()

# -----------------------------------------------------------------------------
# Runtime Shared Library (libdtl_runtime.so)
# Contains the Meyer's singleton for process-global backend state.
# All consumers of DTL::dtl transitively link this DSO.
#
# -----------------------------------------------------------------------------
add_subdirectory(runtime)
target_link_libraries(dtl INTERFACE DTL::runtime)

# -----------------------------------------------------------------------------
# Language Bindings
# -----------------------------------------------------------------------------
if(DTL_BUILD_C_BINDINGS)
    add_subdirectory(src/bindings/c)
endif()

if(DTL_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

if(DTL_BUILD_FORTRAN)
    # Enable Fortran language (requires C bindings)
    if(NOT DTL_BUILD_C_BINDINGS)
        message(FATAL_ERROR "DTL_BUILD_FORTRAN requires DTL_BUILD_C_BINDINGS=ON")
    endif()
    
    enable_language(Fortran)
    add_subdirectory(bindings/fortran)
    set(DTL_FORTRAN_AVAILABLE ON)
else()
    set(DTL_FORTRAN_AVAILABLE OFF)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(DTL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)

    # Inform users about integration tests when they are not enabled.
    # Integration tests (MPI multi-rank, CUDA device, SHMEM, NCCL) are
    # controlled by a separate option because they require external runtime
    # infrastructure (mpirun, GPU hardware, SHMEM launcher) that may not be
    # available in all build environments.
    if(NOT DTL_BUILD_INTEGRATION_TESTS)
        message(STATUS "")
        message(STATUS "NOTE: Integration tests are not enabled.")
        message(STATUS "  Unit tests (DTL_BUILD_TESTS=ON) build standalone test binaries.")
        message(STATUS "  Integration tests require MPI, CUDA, or SHMEM runtime infrastructure.")
        message(STATUS "  To also build integration tests, reconfigure with:")
        message(STATUS "    cmake .. -DDTL_BUILD_INTEGRATION_TESTS=ON")
        message(STATUS "")
    endif()

    # Setup coverage targets after tests are defined
    if(DTL_ENABLE_COVERAGE)
        if(COMMAND dtl_setup_coverage_target)
            dtl_setup_coverage_target(
                TARGET dtl_unit_tests
                OUTPUT_DIR "${CMAKE_BINARY_DIR}/coverage_report"
                EXCLUDE_PATTERNS
                    "/usr/*"
                    "*/tests/*"
                    "*/test/*"
                    "*/_deps/*"
                    "*/googletest/*"
                    "*/googlemock/*"
                    "*/benchmark/*"
            )
        else()
            message(STATUS "Coverage targets unavailable (lcov/genhtml not found); continue with gcovr-compatible instrumentation only.")
        endif()
    endif()
endif()

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
if(DTL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
if(DTL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
if(DTL_BUILD_DOCS)
    find_package(Doxygen)
    find_program(SPHINX_EXECUTABLE NAMES sphinx-build)

    if(DOXYGEN_FOUND)
        add_custom_target(docs_api
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/docs/_generated/doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        message(STATUS "Doxygen found - docs_api target available")
    else()
        message(WARNING "Doxygen not found - docs_api target unavailable")
    endif()

    if(SPHINX_EXECUTABLE)
        add_custom_target(docs_site
            COMMAND ${CMAKE_COMMAND} -E env
                    DTL_DOXYGEN_XML_DIR=${CMAKE_SOURCE_DIR}/docs/_generated/doxygen/xml
                    ${SPHINX_EXECUTABLE}
                    -b html
                    ${CMAKE_SOURCE_DIR}/docs
                    ${CMAKE_BINARY_DIR}/docs/html
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating Sphinx site (RTD theme)"
            VERBATIM
        )
        if(TARGET docs_api)
            add_dependencies(docs_site docs_api)
        endif()
        message(STATUS "sphinx-build found - docs_site target available")
    else()
        message(WARNING "sphinx-build not found - docs_site target unavailable")
    endif()

    if(TARGET docs_api AND TARGET docs_site)
        add_custom_target(docs DEPENDS docs_api docs_site)
    elseif(TARGET docs_api)
        add_custom_target(docs DEPENDS docs_api)
    elseif(TARGET docs_site)
        add_custom_target(docs DEPENDS docs_site)
    else()
        message(WARNING "No documentation generators found; docs target not created")
    endif()
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

# Install headers (C++ and C)
install(DIRECTORY include/dtl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)
install(FILES
    "${DTL_GENERATED_INCLUDE_DIR}/dtl/generated/version_config.hpp"
    "${DTL_GENERATED_INCLUDE_DIR}/dtl/generated/version_config.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dtl/generated"
)

# Install CMake package configuration
install(TARGETS dtl
    EXPORT DTLTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT DTLTargets
    FILE DTLTargets.cmake
    NAMESPACE DTL::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DTL
)

# Generate and install package configuration files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DTLConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DTLConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DTL
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DTLConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DTLConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DTLConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DTL
)

# Generate and install pkg-config file
configure_file(dtl.pc.in "${CMAKE_BINARY_DIR}/dtl.pc" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/dtl.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)

# -----------------------------------------------------------------------------
# Package Configuration Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== DTL Build Configuration ===")
message(STATUS "Version:          ${DTL_VERSION_FULL}")
message(STATUS "Python version:   ${DTL_VERSION_PYTHON}")
message(STATUS "C++ Standard:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  CUDA backend:   ${DTL_ENABLE_CUDA}")
message(STATUS "  HIP backend:    ${DTL_ENABLE_HIP}")
message(STATUS "  NCCL support:   ${DTL_ENABLE_NCCL}")
message(STATUS "  SHMEM support:  ${DTL_ENABLE_SHMEM}")
message(STATUS "  MPI support:    ${DTL_ENABLE_MPI} (available: ${DTL_MPI_AVAILABLE})")
message(STATUS "  Build tests:    ${DTL_BUILD_TESTS}")
message(STATUS "  Integration tests: ${DTL_BUILD_INTEGRATION_TESTS}")
message(STATUS "  Use system GTest:  ${DTL_USE_SYSTEM_GTEST}")
message(STATUS "  Build benchmarks: ${DTL_BUILD_BENCHMARKS}")
message(STATUS "  Build examples: ${DTL_BUILD_EXAMPLES}")
message(STATUS "  Build docs:     ${DTL_BUILD_DOCS}")
message(STATUS "  Sanitizers:     ${DTL_ENABLE_SANITIZERS}")
message(STATUS "  Coverage:       ${DTL_ENABLE_COVERAGE}")
message(STATUS "  C bindings:     ${DTL_BUILD_C_BINDINGS}")
message(STATUS "  Python bindings: ${DTL_BUILD_PYTHON}")
message(STATUS "  Fortran bindings: ${DTL_BUILD_FORTRAN}")
message(STATUS "  Work-stealing pool: ${DTL_USE_WORK_STEALING}")
message(STATUS "")
message(STATUS "Serialization Adapters:")
message(STATUS "  Cereal adapter: ${DTL_ENABLE_CEREAL} (available: ${DTL_CEREAL_AVAILABLE})")
message(STATUS "  Boost adapter:  ${DTL_ENABLE_BOOST_SERIALIZATION} (available: ${DTL_BOOST_SERIALIZATION_AVAILABLE})")

# C ABI instantiation matrix summary (only if C bindings enabled)
if(DTL_BUILD_C_BINDINGS)
    message(STATUS "")
    message(STATUS "C ABI Instantiation Matrix:")
    if(DTL_C_ABI_MINIMAL)
        message(STATUS "  Mode: MINIMAL (2 instantiations)")
        message(STATUS "    dtypes: f32, f64")
        message(STATUS "    partitions: block")
        message(STATUS "    placements: host")
        message(STATUS "    executions: seq")
    elseif(DTL_C_ABI_FULL_MATRIX)
        message(STATUS "  Mode: FULL (~600 instantiations)")
        message(STATUS "  Warning: Long compile times expected")
    else()
        message(STATUS "  Mode: DEFAULT (96 instantiations)")
        message(STATUS "    dtypes: i32, i64, u32, u64, f32, f64, c64, c128")
        message(STATUS "    partitions: block, cyclic, replicated")
        if(DTL_C_ABI_ENABLE_DEVICE)
            message(STATUS "    placements: host, device, unified")
        else()
            message(STATUS "    placements: host, unified")
        endif()
        message(STATUS "    executions: seq, par")
    endif()
    if(DTL_C_ABI_ENABLE_DEVICE AND NOT DTL_ENABLE_CUDA AND NOT DTL_ENABLE_HIP)
        message(WARNING "DTL_C_ABI_ENABLE_DEVICE requires DTL_ENABLE_CUDA or DTL_ENABLE_HIP")
    endif()
endif()

message(STATUS "================================")
message(STATUS "")
