# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# DTL Fortran Tests
# =============================================================================

# Helper function to add a Fortran test
function(dtl_add_fortran_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE dtl_fortran)
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/fortran
    )
    # Single-rank test
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${TEST_NAME}
    )
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "fortran;integration"
        TIMEOUT 30
    )
    # MPI 2-rank test (if available)
    if(DTL_MPI_AVAILABLE AND MPI_Fortran_FOUND)
        add_test(
            NAME ${TEST_NAME}_mpi_2ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${CMAKE_BINARY_DIR}/bin/tests/fortran/${TEST_NAME}
        )
        set_tests_properties(${TEST_NAME}_mpi_2ranks PROPERTIES
            LABELS "fortran;mpi;integration"
            TIMEOUT 60
        )
    endif()
endfunction()

# Basic tests (existing)
dtl_add_fortran_test(test_fortran_basic test_fortran_basic.f90)

# Environment and context tests
dtl_add_fortran_test(test_fortran_environment test_fortran_environment.f90)

# Communication tests
dtl_add_fortran_test(test_fortran_communication test_fortran_communication.f90)

# Tensor tests
dtl_add_fortran_test(test_fortran_tensor test_fortran_tensor.f90)

# Algorithm tests (built-in)
dtl_add_fortran_test(test_fortran_algorithms test_fortran_algorithms.f90)

# Callback algorithm tests
dtl_add_fortran_test(test_fortran_callbacks test_fortran_callbacks.f90)

# RMA tests
dtl_add_fortran_test(test_fortran_rma test_fortran_rma.f90)

# Advanced tests (topology, policies, futures)
dtl_add_fortran_test(test_fortran_advanced test_fortran_advanced.f90)
