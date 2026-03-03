! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_environment.f90
!> @brief Integration tests for DTL Fortran environment and advanced context bindings

program test_fortran_environment
    use dtl
    implicit none

    integer :: num_passed, num_failed
    logical :: test_result

    num_passed = 0
    num_failed = 0

    print *, '================================================'
    print *, ' DTL Fortran Environment Integration Tests'
    print *, '================================================'
    print *, ''

    call test_environment_lifecycle(test_result)
    call report_test('environment_lifecycle', test_result, &
                     num_passed, num_failed)

    call test_environment_backend_queries(test_result)
    call report_test('environment_backend_queries', test_result, &
                     num_passed, num_failed)

    call test_environment_context_factory(test_result)
    call report_test('environment_context_factory', test_result, &
                     num_passed, num_failed)

    call test_context_with_options(test_result)
    call report_test('context_with_options', test_result, &
                     num_passed, num_failed)

    call test_context_dup(test_result)
    call report_test('context_dup', test_result, num_passed, num_failed)

    call test_context_split(test_result)
    call report_test('context_split', test_result, num_passed, num_failed)

    call test_context_fence(test_result)
    call report_test('context_fence', test_result, num_passed, num_failed)

    call test_context_domain_queries(test_result)
    call report_test('context_domain_queries', test_result, &
                     num_passed, num_failed)

    call test_context_policy_queries(test_result)
    call report_test('context_policy_queries', test_result, &
                     num_passed, num_failed)

    call test_shape_helpers(test_result)
    call report_test('shape_helpers', test_result, num_passed, num_failed)

    call test_status_utilities(test_result)
    call report_test('status_utilities', test_result, &
                     num_passed, num_failed)

    call test_backend_global_queries(test_result)
    call report_test('backend_global_queries', test_result, &
                     num_passed, num_failed)

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

    !> Test environment create/destroy lifecycle
    subroutine test_environment_lifecycle(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: env
        integer(c_int) :: status

        passed = .false.

        status = dtl_environment_create(env)
        if (status /= DTL_SUCCESS) return

        ! Ref count should be >= 1
        if (dtl_environment_ref_count() < 1) then
            call dtl_environment_destroy(env)
            return
        end if

        ! Should be initialized
        if (dtl_environment_is_initialized() /= 1) then
            call dtl_environment_destroy(env)
            return
        end if

        call dtl_environment_destroy(env)
        passed = .true.
    end subroutine

    !> Test environment backend availability queries
    subroutine test_environment_backend_queries(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: env
        integer(c_int) :: status, has_mpi

        passed = .false.

        status = dtl_environment_create(env)
        if (status /= DTL_SUCCESS) return

        ! These should return 0 or 1 without crashing
        has_mpi = dtl_environment_has_mpi()
        if (has_mpi /= 0 .and. has_mpi /= 1) then
            call dtl_environment_destroy(env)
            return
        end if

        ! CUDA/HIP/NCCL/SHMEM queries should not crash
        status = dtl_environment_has_cuda()
        status = dtl_environment_has_hip()
        status = dtl_environment_has_nccl()
        status = dtl_environment_has_shmem()

        call dtl_environment_destroy(env)
        passed = .true.
    end subroutine

    !> Test creating a context from environment
    subroutine test_environment_context_factory(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: env, ctx
        integer(c_int) :: status

        passed = .false.

        status = dtl_environment_create(env)
        if (status /= DTL_SUCCESS) return

        status = dtl_environment_make_world_context(env, ctx)
        if (status /= DTL_SUCCESS) then
            call dtl_environment_destroy(env)
            return
        end if

        if (dtl_context_is_valid(ctx) /= 1) then
            call dtl_context_destroy(ctx)
            call dtl_environment_destroy(env)
            return
        end if

        if (dtl_context_size(ctx) < 1) then
            call dtl_context_destroy(ctx)
            call dtl_environment_destroy(env)
            return
        end if

        call dtl_context_destroy(ctx)
        call dtl_environment_destroy(env)
        passed = .true.
    end subroutine

    !> Test context creation with options
    subroutine test_context_with_options(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status
        type(dtl_context_options) :: opts

        passed = .false.

        call dtl_context_options_init(opts)
        opts%device_id = -1
        opts%init_mpi = 1
        opts%finalize_mpi = 0

        status = dtl_context_create(ctx, opts)
        if (status /= DTL_SUCCESS) return

        if (dtl_context_is_valid(ctx) /= 1) then
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test context duplication
    subroutine test_context_dup(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, dup_ctx
        integer(c_int) :: status

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        status = dtl_context_dup(ctx, dup_ctx)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Duplicated context should have same rank and size
        if (dtl_context_rank(dup_ctx) /= dtl_context_rank(ctx)) then
            call dtl_context_destroy(dup_ctx)
            call dtl_context_destroy(ctx)
            return
        end if

        if (dtl_context_size(dup_ctx) /= dtl_context_size(ctx)) then
            call dtl_context_destroy(dup_ctx)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_context_destroy(dup_ctx)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test context split
    subroutine test_context_split(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, split_ctx
        integer(c_int) :: status, color, key

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Split with color=0 (all same group), key=rank
        color = 0
        key = dtl_context_rank(ctx)

        status = dtl_context_split(ctx, color, key, split_ctx)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        if (dtl_context_is_valid(split_ctx) /= 1) then
            call dtl_context_destroy(split_ctx)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Same color means same size
        if (dtl_context_size(split_ctx) /= dtl_context_size(ctx)) then
            call dtl_context_destroy(split_ctx)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_context_destroy(split_ctx)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test context fence
    subroutine test_context_fence(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        status = dtl_context_fence(ctx)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test context domain queries (has_mpi, has_cuda, etc.)
    subroutine test_context_domain_queries(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, val

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! These should return 0 or 1 without crashing
        val = dtl_context_has_mpi(ctx)
        if (val /= 0 .and. val /= 1) then
            call dtl_context_destroy(ctx)
            return
        end if

        val = dtl_context_has_cuda(ctx)
        if (val /= 0 .and. val /= 1) then
            call dtl_context_destroy(ctx)
            return
        end if

        val = dtl_context_has_nccl(ctx)
        val = dtl_context_has_shmem(ctx)

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test context policy queries
    subroutine test_context_policy_queries(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, mode

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        mode = dtl_context_determinism_mode(ctx)
        if (mode /= DTL_DETERMINISM_THROUGHPUT .and. &
            mode /= DTL_DETERMINISM_DETERMINISTIC) then
            call dtl_context_destroy(ctx)
            return
        end if

        mode = dtl_context_reduction_schedule_policy(ctx)
        mode = dtl_context_progress_ordering_policy(ctx)

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test shape helper functions
    subroutine test_shape_helpers(passed)
        logical, intent(out) :: passed
        type(dtl_shape) :: s

        passed = .false.

        ! Test Fortran convenience constructors
        s = dtl_make_shape_1d(10_c_int64_t)
        if (s%ndim /= 1) return
        if (s%dims(1) /= 10) return

        s = dtl_make_shape_2d(3_c_int64_t, 4_c_int64_t)
        if (s%ndim /= 2) return
        if (s%dims(1) /= 3) return
        if (s%dims(2) /= 4) return

        s = dtl_make_shape_3d(2_c_int64_t, 3_c_int64_t, 5_c_int64_t)
        if (s%ndim /= 3) return
        if (s%dims(1) /= 2) return
        if (s%dims(2) /= 3) return
        if (s%dims(3) /= 5) return

        ! Test C API shape constructors
        s = dtl_shape_1d(100_c_int64_t)
        if (s%ndim /= 1) return
        if (s%dims(1) /= 100) return

        s = dtl_shape_2d(10_c_int64_t, 20_c_int64_t)
        if (s%ndim /= 2) return

        passed = .true.
    end subroutine

    !> Test extended status utilities
    subroutine test_status_utilities(passed)
        logical, intent(out) :: passed
        integer(c_int) :: val
        character(len=:), allocatable :: msg

        passed = .false.

        ! status_ok
        val = dtl_status_ok(DTL_SUCCESS)
        if (val /= 1) return

        val = dtl_status_ok(DTL_ERROR_MEMORY)
        if (val /= 0) return

        ! status_is_error
        val = dtl_status_is_error(DTL_ERROR_MEMORY)
        if (val /= 1) return

        val = dtl_status_is_error(DTL_SUCCESS)
        if (val /= 0) return

        ! category code
        val = dtl_status_category_code(DTL_ERROR_MEMORY)
        if (val /= DTL_CATEGORY_MEMORY) return

        ! is_category
        val = dtl_status_is_category(DTL_ERROR_MEMORY, DTL_CATEGORY_MEMORY)
        if (val /= 1) return

        ! get_error_message (Fortran helper)
        msg = dtl_get_error_message(DTL_SUCCESS)
        if (len(msg) == 0) return

        passed = .true.
    end subroutine

    !> Test global backend queries
    subroutine test_backend_global_queries(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: env
        integer(c_int) :: status, count
        character(len=:), allocatable :: name_str, ver_str

        passed = .false.

        status = dtl_environment_create(env)
        if (status /= DTL_SUCCESS) return

        count = dtl_backend_count()
        if (count < 0) then
            call dtl_environment_destroy(env)
            return
        end if

        name_str = dtl_c_to_f_string(dtl_backend_name())
        ver_str = dtl_c_to_f_string(dtl_version())

        call dtl_environment_destroy(env)
        passed = .true.
    end subroutine

end program test_fortran_environment
