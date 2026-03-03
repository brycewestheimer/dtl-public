! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_advanced.f90
!> @brief Integration tests for DTL Fortran advanced bindings (topology, policies, futures)
!>
!> Tests:
!> 1. Topology CPU count query
!> 2. Policy constants verification
!> 3. Container options initialization and defaults
!> 4. Vector creation with options and policy queries
!>
!> Exit code 0 on success, non-zero on failure.

program test_fortran_advanced
    use dtl
    implicit none

    integer :: num_passed, num_failed
    logical :: test_result

    num_passed = 0
    num_failed = 0

    print *, '================================================'
    print *, ' DTL Fortran Advanced Integration Tests'
    print *, '================================================'
    print *, ''

    ! Run all tests
    call test_topology_cpus(test_result)
    call report_test('topology_cpus', test_result, num_passed, num_failed)

    call test_policies_constants(test_result)
    call report_test('policies_constants', test_result, num_passed, num_failed)

    call test_container_options(test_result)
    call report_test('container_options', test_result, num_passed, num_failed)

    call test_vector_with_options(test_result)
    call report_test('vector_with_options', test_result, num_passed, num_failed)

    ! Summary
    print *, ''
    print *, '================================================'
    print '(A,I0,A,I0)', '  Passed: ', num_passed, '  Failed: ', num_failed
    print *, '================================================'

    if (num_failed > 0) then
        print *, 'TESTS FAILED'
        stop 1
    else
        print *, 'ALL TESTS PASSED'
    end if

contains

    subroutine report_test(name, passed, num_passed, num_failed)
        character(len=*), intent(in) :: name
        logical, intent(in) :: passed
        integer, intent(inout) :: num_passed, num_failed

        if (passed) then
            print '(A,A,A)', '  [PASS] ', name, ''
            num_passed = num_passed + 1
        else
            print '(A,A,A)', '  [FAIL] ', name, ''
            num_failed = num_failed + 1
        end if
    end subroutine

    !> Test topology: query number of CPUs, verify count > 0
    subroutine test_topology_cpus(passed)
        logical, intent(out) :: passed
        integer(c_int) :: status, cpu_count

        passed = .false.

        cpu_count = 0
        status = dtl_topology_num_cpus(cpu_count)
        if (status /= DTL_SUCCESS) return

        ! Any system should have at least 1 CPU
        if (cpu_count <= 0) return

        passed = .true.
    end subroutine

    !> Test policy constants: verify expected enumeration values
    subroutine test_policies_constants(passed)
        logical, intent(out) :: passed

        passed = .false.

        ! Verify partition policies
        if (DTL_PARTITION_BLOCK /= 0) return
        if (DTL_PARTITION_CYCLIC /= 1) return
        if (DTL_PARTITION_BLOCK_CYCLIC /= 2) return
        if (DTL_PARTITION_HASH /= 3) return
        if (DTL_PARTITION_REPLICATED /= 4) return

        ! Verify placement policies
        if (DTL_PLACEMENT_HOST /= 0) return
        if (DTL_PLACEMENT_DEVICE /= 1) return
        if (DTL_PLACEMENT_UNIFIED /= 2) return

        ! Verify execution policies
        if (DTL_EXEC_SEQ /= 0) return
        if (DTL_EXEC_PAR /= 1) return
        if (DTL_EXEC_ASYNC /= 2) return

        ! Verify consistency policies
        if (DTL_CONSISTENCY_BULK_SYNCHRONOUS /= 0) return
        if (DTL_CONSISTENCY_RELAXED /= 1) return

        ! Verify error policies
        if (DTL_ERROR_POLICY_RETURN_STATUS /= 0) return

        passed = .true.
    end subroutine

    !> Test container options: create, init, verify defaults
    subroutine test_container_options(passed)
        logical, intent(out) :: passed
        type(dtl_container_options) :: opts

        passed = .false.

        ! Initialize with defaults via C API
        call dtl_container_options_init(opts)

        ! Verify defaults: block partition, host placement, sequential exec
        if (opts%partition /= DTL_PARTITION_BLOCK) return
        if (opts%placement /= DTL_PLACEMENT_HOST) return
        if (opts%execution /= DTL_EXEC_SEQ) return
        if (opts%device_id /= 0) return

        passed = .true.
    end subroutine

    !> Test vector creation with options and policy query-back
    subroutine test_vector_with_options(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        type(dtl_container_options) :: opts
        integer(c_int64_t) :: global_size

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Initialize options
        call dtl_container_options_init(opts)
        ! Use block partition, host placement, sequential execution
        opts%partition = DTL_PARTITION_BLOCK
        opts%placement = DTL_PLACEMENT_HOST
        opts%execution = DTL_EXEC_SEQ

        global_size = 200_c_int64_t
        status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, &
                                                global_size, opts, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify validity
        if (dtl_vector_is_valid(vec) /= 1) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Query policies back
        if (dtl_vector_partition_policy(vec) /= DTL_PARTITION_BLOCK) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        if (dtl_vector_placement_policy(vec) /= DTL_PLACEMENT_HOST) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        if (dtl_vector_execution_policy(vec) /= DTL_EXEC_SEQ) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify global size
        if (dtl_vector_global_size(vec) /= global_size) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

end program test_fortran_advanced
