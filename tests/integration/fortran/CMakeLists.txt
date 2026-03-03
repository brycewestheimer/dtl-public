# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# Fortran MPI Integration Tests and Benchmarks
# =============================================================================
# Requires DTL_BUILD_FORTRAN=ON and Fortran compiler.
# MPI tests also require DTL_MPI_AVAILABLE and MPI Fortran support.
# =============================================================================

if(DTL_BUILD_FORTRAN)

    # -----------------------------------------------------------------
    # Fortran MPI Integration Test
    # -----------------------------------------------------------------
    add_executable(test_fortran_mpi test_fortran_mpi.f90)
    target_link_libraries(test_fortran_mpi PRIVATE dtl_fortran)

    set_target_properties(test_fortran_mpi PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration/fortran
    )

    # Single-rank test (always)
    add_test(
        NAME FortranMpi_1
        COMMAND $<TARGET_FILE:test_fortran_mpi>
    )
    set_tests_properties(FortranMpi_1 PROPERTIES
        LABELS "integration;fortran"
        TIMEOUT 60
    )

    # Multi-rank MPI tests (if MPI Fortran available)
    if(DTL_MPI_AVAILABLE AND MPI_Fortran_FOUND)
        target_link_libraries(test_fortran_mpi PRIVATE MPI::MPI_Fortran)

        add_test(
            NAME FortranMpi_2
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    $<TARGET_FILE:test_fortran_mpi>
        )
        add_test(
            NAME FortranMpi_4
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                    $<TARGET_FILE:test_fortran_mpi>
        )
        set_tests_properties(FortranMpi_2 FortranMpi_4 PROPERTIES
            LABELS "integration;fortran;mpi"
            TIMEOUT 60
        )
    endif()

    # -----------------------------------------------------------------
    # Fortran API Benchmark
    # -----------------------------------------------------------------
    add_executable(bench_fortran_api bench_fortran_api.f90)
    target_link_libraries(bench_fortran_api PRIVATE dtl_fortran)

    set_target_properties(bench_fortran_api PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/integration/fortran
    )

    if(DTL_MPI_AVAILABLE AND MPI_Fortran_FOUND)
        target_link_libraries(bench_fortran_api PRIVATE MPI::MPI_Fortran)
    endif()

endif()
