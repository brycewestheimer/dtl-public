# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# NCCL Integration Tests
# =============================================================================
# Tests for NCCL domain creation and collective operations.
# Requires MPI, CUDA, and NCCL to be enabled.
# =============================================================================

if(DTL_ENABLE_NCCL AND DTL_ENABLE_CUDA AND DTL_ENABLE_MPI)
    message(STATUS "Building NCCL integration tests")

    # Find NCCL and CUDA
    find_package(NCCL QUIET)
    find_package(CUDAToolkit QUIET)

    if(NCCL_FOUND AND CUDAToolkit_FOUND)
        # Test: NCCL domain from MPI
        add_executable(test_nccl_domain_from_mpi test_nccl_domain_from_mpi.cpp)
        target_link_libraries(test_nccl_domain_from_mpi PRIVATE
            DTL::dtl
            DTL::nccl
            GTest::gtest
            MPI::MPI_CXX
            CUDA::cudart
            ${NCCL_LIBRARIES}
        )
        target_include_directories(test_nccl_domain_from_mpi PRIVATE
            ${CMAKE_SOURCE_DIR}/backends
            ${NCCL_INCLUDE_DIRS}
        )
        target_compile_definitions(test_nccl_domain_from_mpi PRIVATE
            DTL_ENABLE_NCCL=1
            DTL_ENABLE_CUDA=1
            DTL_ENABLE_MPI=1
        )
        target_compile_features(test_nccl_domain_from_mpi PRIVATE cxx_std_20)

        # Test: NCCL allreduce
        add_executable(test_nccl_allreduce test_nccl_allreduce.cpp)
        target_link_libraries(test_nccl_allreduce PRIVATE
            DTL::dtl
            DTL::nccl
            GTest::gtest
            MPI::MPI_CXX
            CUDA::cudart
            ${NCCL_LIBRARIES}
        )
        target_include_directories(test_nccl_allreduce PRIVATE
            ${CMAKE_SOURCE_DIR}/backends
            ${NCCL_INCLUDE_DIRS}
        )
        target_compile_definitions(test_nccl_allreduce PRIVATE
            DTL_ENABLE_NCCL=1
            DTL_ENABLE_CUDA=1
            DTL_ENABLE_MPI=1
        )
        target_compile_features(test_nccl_allreduce PRIVATE cxx_std_20)

        # Test: Context with NCCL
        add_executable(test_context_with_nccl test_context_with_nccl.cpp)
        target_link_libraries(test_context_with_nccl PRIVATE
            DTL::dtl
            DTL::nccl
            GTest::gtest
            MPI::MPI_CXX
            CUDA::cudart
            ${NCCL_LIBRARIES}
        )
        target_include_directories(test_context_with_nccl PRIVATE
            ${CMAKE_SOURCE_DIR}/backends
            ${NCCL_INCLUDE_DIRS}
        )
        target_compile_definitions(test_context_with_nccl PRIVATE
            DTL_ENABLE_NCCL=1
            DTL_ENABLE_CUDA=1
            DTL_ENABLE_MPI=1
        )
        target_compile_features(test_context_with_nccl PRIVATE cxx_std_20)

        # Add tests with MPI runner (2 ranks)
        add_test(
            NAME nccl_domain_from_mpi_2ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_nccl_domain_from_mpi>
                    ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(nccl_domain_from_mpi_2ranks PROPERTIES
            LABELS "mpi;nccl;integration"
            TIMEOUT 120
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=0,1"
        )

        add_test(
            NAME nccl_allreduce_2ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_nccl_allreduce>
                    ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(nccl_allreduce_2ranks PROPERTIES
            LABELS "mpi;nccl;integration"
            TIMEOUT 120
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=0,1"
        )

        add_test(
            NAME context_with_nccl_2ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_context_with_nccl>
                    ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(context_with_nccl_2ranks PROPERTIES
            LABELS "mpi;nccl;integration"
            TIMEOUT 120
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=0,1"
        )

    else()
        message(STATUS "NCCL or CUDA not found, skipping NCCL integration tests")
    endif()
else()
    message(STATUS "NCCL integration tests disabled (requires DTL_ENABLE_NCCL, DTL_ENABLE_CUDA, DTL_ENABLE_MPI)")
endif()
