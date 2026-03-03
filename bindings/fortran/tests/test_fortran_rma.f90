! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_rma.f90
!> @brief Integration tests for DTL Fortran RMA (Remote Memory Access) bindings
!>
!> Tests:
!> 1. Window lifecycle (allocate, query, destroy)
!> 2. Fence put/get (2+ ranks: put to remote, fence, get back)
!>
!> Exit code 0 on success, non-zero on failure.

program test_fortran_rma
    use dtl
    implicit none

    integer :: num_passed, num_failed
    logical :: test_result

    num_passed = 0
    num_failed = 0

    print *, '================================================'
    print *, ' DTL Fortran RMA Integration Tests'
    print *, '================================================'
    print *, ''

    ! Run all tests
    call test_window_lifecycle(test_result)
    call report_test('window_lifecycle', test_result, num_passed, num_failed)

    call test_fence_put_get(test_result)
    call report_test('fence_put_get', test_result, num_passed, num_failed)

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

    !> Test window lifecycle: allocate, verify valid, check size, destroy
    subroutine test_window_lifecycle(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, win
        integer(c_int) :: status
        integer(c_int64_t) :: win_size

        passed = .false.
        ctx = c_null_ptr
        win = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Allocate a window of 1024 bytes
        status = dtl_window_allocate(ctx, 1024_c_int64_t, win)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify validity
        if (dtl_window_is_valid(win) /= 1) then
            call dtl_window_destroy(win)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Check size
        win_size = dtl_window_size(win)
        if (win_size /= 1024_c_int64_t) then
            call dtl_window_destroy(win)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify base pointer is valid
        if (.not. c_associated(dtl_window_base(win))) then
            call dtl_window_destroy(win)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_window_destroy(win)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test fence-based put/get.
    !> If 2+ ranks: rank 0 puts a double to rank 1's window, rank 1 reads it.
    !> If single rank: test fence with self (put to own window).
    subroutine test_fence_put_get(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, win
        integer(c_int) :: status, rank, nprocs
        integer(c_int64_t) :: win_bytes
        real(c_double), target :: put_val, get_val
        real(c_double), pointer :: win_data(:)
        type(c_ptr) :: base_ptr

        passed = .false.
        ctx = c_null_ptr
        win = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        rank = dtl_context_rank(ctx)
        nprocs = dtl_context_size(ctx)

        ! Allocate window large enough for one double (8 bytes)
        win_bytes = 8_c_int64_t
        status = dtl_window_allocate(ctx, win_bytes, win)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Initialize window data to 0.0
        base_ptr = dtl_window_base(win)
        if (.not. c_associated(base_ptr)) then
            call dtl_window_destroy(win)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(base_ptr, win_data, [1])
        win_data(1) = 0.0_c_double
        nullify(win_data)

        ! Opening fence
        status = dtl_window_fence(win)
        if (status /= DTL_SUCCESS) then
            call dtl_window_destroy(win)
            call dtl_context_destroy(ctx)
            return
        end if

        if (nprocs >= 2) then
            ! Multi-rank test: rank 0 puts a value into rank 1's window
            if (rank == 0) then
                put_val = 42.5_c_double
                status = dtl_rma_put(win, 1, 0_c_int64_t, &
                                     c_loc(put_val), win_bytes)
                if (status /= DTL_SUCCESS) then
                    call dtl_window_destroy(win)
                    call dtl_context_destroy(ctx)
                    return
                end if
            end if

            ! Closing fence (synchronizes all puts)
            status = dtl_window_fence(win)
            if (status /= DTL_SUCCESS) then
                call dtl_window_destroy(win)
                call dtl_context_destroy(ctx)
                return
            end if

            ! Rank 1 verifies its window data
            if (rank == 1) then
                base_ptr = dtl_window_base(win)
                call c_f_pointer(base_ptr, win_data, [1])
                if (abs(win_data(1) - 42.5_c_double) > 1.0d-10) then
                    nullify(win_data)
                    call dtl_window_destroy(win)
                    call dtl_context_destroy(ctx)
                    return
                end if
                nullify(win_data)
            end if
        else
            ! Single rank: put to own window at offset 0
            put_val = 99.9_c_double
            status = dtl_rma_put(win, 0, 0_c_int64_t, &
                                 c_loc(put_val), win_bytes)
            if (status /= DTL_SUCCESS) then
                call dtl_window_destroy(win)
                call dtl_context_destroy(ctx)
                return
            end if

            ! Closing fence
            status = dtl_window_fence(win)
            if (status /= DTL_SUCCESS) then
                call dtl_window_destroy(win)
                call dtl_context_destroy(ctx)
                return
            end if

            ! Get the value back
            get_val = 0.0_c_double
            status = dtl_rma_get(win, 0, 0_c_int64_t, &
                                 c_loc(get_val), win_bytes)
            if (status /= DTL_SUCCESS) then
                call dtl_window_destroy(win)
                call dtl_context_destroy(ctx)
                return
            end if

            ! Need another fence to complete the get
            status = dtl_window_fence(win)
            if (status /= DTL_SUCCESS) then
                call dtl_window_destroy(win)
                call dtl_context_destroy(ctx)
                return
            end if

            if (abs(get_val - 99.9_c_double) > 1.0d-10) then
                call dtl_window_destroy(win)
                call dtl_context_destroy(ctx)
                return
            end if
        end if

        call dtl_window_destroy(win)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

end program test_fortran_rma
