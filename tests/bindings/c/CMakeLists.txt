# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL C Bindings Tests
# =============================================================================

# Find GoogleTest
find_package(GTest REQUIRED)

# Test sources
set(DTL_C_TEST_SOURCES
    test_main.cpp
    test_types.cpp
    test_status.cpp
    test_environment.cpp
    test_context.cpp
    test_communicator.cpp
    test_policies.cpp
    test_policies_dispatch.cpp
    test_vector.cpp
    test_array.cpp
    test_span.cpp
    test_tensor.cpp
    test_tensor_v2.cpp
    test_algorithms.cpp
    test_rma.cpp
    test_mpmd.cpp
    test_topology.cpp
    test_futures.cpp
)

# Create test executable
add_executable(test_c_bindings ${DTL_C_TEST_SOURCES})

# Link against dtl_c and GTest
target_link_libraries(test_c_bindings
    PRIVATE
        dtl_c
        GTest::gtest
)

if(DTL_MPI_AVAILABLE)
    target_compile_definitions(test_c_bindings PRIVATE DTL_HAS_MPI)
    target_link_libraries(test_c_bindings PRIVATE MPI::MPI_CXX)
endif()

if(DTL_ENABLE_CUDA)
    target_compile_definitions(test_c_bindings PRIVATE DTL_ENABLE_CUDA=1 DTL_HAS_CUDA=1)
    target_link_libraries(test_c_bindings PRIVATE CUDA::cudart)
endif()

if(DTL_ENABLE_NCCL)
    target_compile_definitions(test_c_bindings PRIVATE DTL_HAS_NCCL=1 DTL_HAS_CUDA=1)
    target_link_libraries(test_c_bindings PRIVATE NCCL::NCCL CUDA::cudart)
endif()

# Include directories
target_include_directories(test_c_bindings
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Register with CTest
add_test(NAME CBindingsTests COMMAND test_c_bindings)

# If MPI is available, add MPI multi-rank test
if(DTL_MPI_AVAILABLE)
    add_test(
        NAME CBindingsTests_MPI_2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_c_bindings>
    )
    add_test(
        NAME CBindingsTests_MPI_4
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_c_bindings>
    )
endif()
