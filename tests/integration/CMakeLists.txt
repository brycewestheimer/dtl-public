# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# Integration Tests — MPI, CUDA, NCCL multi-rank tests
# =============================================================================
# These tests require real MPI execution (mpirun) and optionally GPU hardware.
# They are built only when DTL_BUILD_INTEGRATION_TESTS is ON (checked by the
# parent tests/CMakeLists.txt before add_subdirectory(integration)).
# =============================================================================

# -------------------------------------------------------------------------
# MPI Multi-Rank Integration Tests
# -------------------------------------------------------------------------
if(DTL_MPI_AVAILABLE)
    add_executable(test_mpi_multirank test_mpi_multirank.cpp)
    target_compile_features(test_mpi_multirank PRIVATE cxx_std_20)
    target_link_libraries(test_mpi_multirank PRIVATE
        DTL::dtl
        DTL::cpu
        MPI::MPI_CXX
        GTest::gtest
    )
    target_include_directories(test_mpi_multirank PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/backends
    )
    # Suppress deprecation warnings from legacy constructors used in tests
    target_compile_definitions(test_mpi_multirank PRIVATE DTL_SUPPRESS_DEPRECATION_WARNINGS=1)

    # Apply compiler warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_mpi_multirank PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME MpiMultiRank_2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_mpi_multirank>)
    add_test(NAME MpiMultiRank_4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                $<TARGET_FILE:test_mpi_multirank>)
    set_tests_properties(MpiMultiRank_2 MpiMultiRank_4 PROPERTIES
        LABELS "integration;mpi"
        TIMEOUT 60
    )
endif()

# -------------------------------------------------------------------------
# CUDA + MPI Integration Tests
# -------------------------------------------------------------------------
if(DTL_ENABLE_CUDA AND DTL_MPI_AVAILABLE)
    add_executable(test_cuda_multirank test_cuda_multirank.cpp)
    target_compile_features(test_cuda_multirank PRIVATE cxx_std_20)
    target_link_libraries(test_cuda_multirank PRIVATE
        DTL::dtl
        DTL::cpu
        DTL::cuda
        MPI::MPI_CXX
        GTest::gtest
        CUDA::cudart
    )
    target_include_directories(test_cuda_multirank PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/backends
    )
    target_compile_definitions(test_cuda_multirank PRIVATE DTL_SUPPRESS_DEPRECATION_WARNINGS=1)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_cuda_multirank PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME CudaMultiRank_1
        COMMAND $<TARGET_FILE:test_cuda_multirank>)
    add_test(NAME CudaMultiRank_2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_cuda_multirank>)
    set_tests_properties(CudaMultiRank_1 CudaMultiRank_2 PROPERTIES
        LABELS "integration;cuda"
        TIMEOUT 60
    )
endif()

# -------------------------------------------------------------------------
# NCCL + CUDA + MPI Integration Tests
# -------------------------------------------------------------------------
if(DTL_ENABLE_NCCL AND DTL_ENABLE_CUDA AND DTL_MPI_AVAILABLE)
    add_executable(test_nccl_multirank test_nccl_multirank.cpp)
    target_compile_features(test_nccl_multirank PRIVATE cxx_std_20)
    target_link_libraries(test_nccl_multirank PRIVATE
        DTL::dtl
        DTL::cpu
        DTL::cuda
        DTL::nccl
        MPI::MPI_CXX
        GTest::gtest
        CUDA::cudart
    )
    target_include_directories(test_nccl_multirank PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/backends
    )
    target_compile_definitions(test_nccl_multirank PRIVATE DTL_SUPPRESS_DEPRECATION_WARNINGS=1)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_nccl_multirank PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME NcclMultiRank_2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_nccl_multirank>)
    set_tests_properties(NcclMultiRank_2 PROPERTIES
        LABELS "integration;nccl;cuda"
        TIMEOUT 60
    )
endif()

# -------------------------------------------------------------------------
# C Binding MPI Integration Tests
# -------------------------------------------------------------------------
if(DTL_MPI_AVAILABLE AND DTL_BUILD_C_BINDINGS)
    add_executable(test_c_bindings_mpi test_c_bindings_mpi.cpp)
    target_compile_features(test_c_bindings_mpi PRIVATE cxx_std_20)
    target_link_libraries(test_c_bindings_mpi PRIVATE
        dtl_c
        DTL::dtl
        DTL::cpu
        MPI::MPI_CXX
        GTest::gtest
    )
    target_include_directories(test_c_bindings_mpi PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/backends
    )
    target_compile_definitions(test_c_bindings_mpi PRIVATE DTL_SUPPRESS_DEPRECATION_WARNINGS=1)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_c_bindings_mpi PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME CBindingsMpi_1
        COMMAND $<TARGET_FILE:test_c_bindings_mpi>)
    add_test(NAME CBindingsMpi_2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:test_c_bindings_mpi>)
    add_test(NAME CBindingsMpi_4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                $<TARGET_FILE:test_c_bindings_mpi>)
    set_tests_properties(CBindingsMpi_1 CBindingsMpi_2 CBindingsMpi_4 PROPERTIES
        LABELS "integration;c;mpi"
        TIMEOUT 60
    )
endif()

# -------------------------------------------------------------------------
# Fortran MPI Integration Tests
# -------------------------------------------------------------------------
if(DTL_BUILD_FORTRAN)
    add_subdirectory(fortran)
endif()
