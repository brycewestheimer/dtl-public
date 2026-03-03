! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_communication.f90
!> @brief Integration tests for DTL Fortran communication bindings
!>
!> Tests:
!> 1. Barrier synchronization
!> 2. Broadcast and allreduce
!> 3. Reduce to root
!> 4. Scan (inclusive prefix sum)
!> 5. Point-to-point send/recv (2+ ranks)
!> 6. Probe before receive (2+ ranks)
!>
!> Exit code 0 on success, non-zero on failure.

program test_fortran_communication
    use dtl
    implicit none

    integer :: num_passed, num_failed
    logical :: test_result

    num_passed = 0
    num_failed = 0

    print *, '================================================'
    print *, ' DTL Fortran Communication Integration Tests'
    print *, '================================================'
    print *, ''

    ! Run all tests
    call test_barrier(test_result)
    call report_test('barrier', test_result, num_passed, num_failed)

    call test_broadcast_allreduce(test_result)
    call report_test('broadcast_allreduce', test_result, num_passed, num_failed)

    call test_reduce_to_root(test_result)
    call report_test('reduce_to_root', test_result, num_passed, num_failed)

    call test_scan(test_result)
    call report_test('scan', test_result, num_passed, num_failed)

    call test_send_recv(test_result)
    call report_test('send_recv', test_result, num_passed, num_failed)

    call test_probe(test_result)
    call report_test('probe', test_result, num_passed, num_failed)

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

    !> Test barrier synchronization
    subroutine test_barrier(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Call barrier via the communication module
        status = dtl_barrier(ctx)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test broadcast from root and allreduce sum
    subroutine test_broadcast_allreduce(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, rank, nprocs
        real(c_double), target :: bcast_data(4)
        real(c_double), target :: send_data(4), recv_data(4)
        integer :: i

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        rank = dtl_context_rank(ctx)
        nprocs = dtl_context_size(ctx)

        ! --- Broadcast test ---
        ! Root fills data; others should receive it
        if (rank == 0) then
            do i = 1, 4
                bcast_data(i) = real(i * 10, c_double)
            end do
        else
            bcast_data = 0.0_c_double
        end if

        status = dtl_broadcast_double(ctx, bcast_data, 4_c_int64_t, 0)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify broadcast results
        do i = 1, 4
            if (abs(bcast_data(i) - real(i * 10, c_double)) > 1.0d-10) then
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        ! --- Allreduce sum test ---
        do i = 1, 4
            send_data(i) = 1.0_c_double
        end do

        status = dtl_allreduce_sum_double(ctx, send_data, recv_data, 4_c_int64_t)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Each element should equal nprocs (each rank contributed 1.0)
        do i = 1, 4
            if (abs(recv_data(i) - real(nprocs, c_double)) > 1.0d-10) then
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test reduce to root with DTL_OP_SUM
    subroutine test_reduce_to_root(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, rank, nprocs
        real(c_double), target :: send_data(3), recv_data(3)
        integer :: i

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        rank = dtl_context_rank(ctx)
        nprocs = dtl_context_size(ctx)

        ! Each rank contributes its rank+1
        do i = 1, 3
            send_data(i) = real(rank + 1, c_double)
        end do
        recv_data = 0.0_c_double

        status = dtl_reduce_double(ctx, send_data, recv_data, &
                                   3_c_int64_t, DTL_OP_SUM, 0)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Only root verifies the result
        if (rank == 0) then
            ! Sum of (rank+1) for rank=0..nprocs-1 = nprocs*(nprocs+1)/2
            do i = 1, 3
                if (abs(recv_data(i) - real(nprocs * (nprocs + 1) / 2, &
                        c_double)) > 1.0d-10) then
                    call dtl_context_destroy(ctx)
                    return
                end if
            end do
        end if

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test inclusive scan with DTL_OP_SUM
    subroutine test_scan(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, rank
        real(c_double), target :: send_data(2), recv_data(2)
        real(c_double) :: expected
        integer :: i

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        rank = dtl_context_rank(ctx)

        ! Each rank contributes 1.0
        do i = 1, 2
            send_data(i) = 1.0_c_double
        end do
        recv_data = 0.0_c_double

        status = dtl_scan_double(ctx, send_data, recv_data, &
                                 2_c_int64_t, DTL_OP_SUM)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! After inclusive scan with all 1.0, rank r should have r+1
        expected = real(rank + 1, c_double)
        do i = 1, 2
            if (abs(recv_data(i) - expected) > 1.0d-10) then
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test point-to-point send/recv (requires 2+ ranks)
    subroutine test_send_recv(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, rank, nprocs
        real(c_double), target :: send_data(5), recv_data(5)
        integer :: i

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        rank = dtl_context_rank(ctx)
        nprocs = dtl_context_size(ctx)

        if (dtl_context_size(ctx) >= 2) then
            if (rank == 0) then
                ! Rank 0 sends data to rank 1
                do i = 1, 5
                    send_data(i) = real(i * 100, c_double)
                end do
                status = dtl_send_double(ctx, send_data, 5_c_int64_t, 1, 42)
                if (status /= DTL_SUCCESS) then
                    call dtl_context_destroy(ctx)
                    return
                end if
            else if (rank == 1) then
                ! Rank 1 receives data from rank 0
                recv_data = 0.0_c_double
                status = dtl_recv_double(ctx, recv_data, 5_c_int64_t, 0, 42)
                if (status /= DTL_SUCCESS) then
                    call dtl_context_destroy(ctx)
                    return
                end if

                ! Verify received data
                do i = 1, 5
                    if (abs(recv_data(i) - real(i * 100, c_double)) > &
                            1.0d-10) then
                        call dtl_context_destroy(ctx)
                        return
                    end if
                end do
            end if
        else
            ! Single rank: test passes trivially
            print *, '    (skipped: requires 2+ ranks)'
        end if

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test probe before receive (requires 2+ ranks)
    subroutine test_probe(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, rank
        real(c_double), target :: send_data(3), recv_data(3)
        type(dtl_message_info) :: info
        integer :: i

        passed = .false.

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        rank = dtl_context_rank(ctx)

        if (dtl_context_size(ctx) >= 2) then
            if (rank == 0) then
                ! Rank 0 sends data
                do i = 1, 3
                    send_data(i) = real(i * 7, c_double)
                end do
                status = dtl_send_double(ctx, send_data, 3_c_int64_t, 1, 99)
                if (status /= DTL_SUCCESS) then
                    call dtl_context_destroy(ctx)
                    return
                end if
            else if (rank == 1) then
                ! Rank 1 probes then receives
                status = dtl_probe(ctx, 0, 99, DTL_DTYPE_FLOAT64, info)
                if (status /= DTL_SUCCESS) then
                    call dtl_context_destroy(ctx)
                    return
                end if

                ! Verify probe info
                if (info%source /= 0) then
                    call dtl_context_destroy(ctx)
                    return
                end if
                if (info%tag /= 99) then
                    call dtl_context_destroy(ctx)
                    return
                end if

                ! Now receive
                recv_data = 0.0_c_double
                status = dtl_recv_double(ctx, recv_data, 3_c_int64_t, 0, 99)
                if (status /= DTL_SUCCESS) then
                    call dtl_context_destroy(ctx)
                    return
                end if

                ! Verify data
                do i = 1, 3
                    if (abs(recv_data(i) - real(i * 7, c_double)) > &
                            1.0d-10) then
                        call dtl_context_destroy(ctx)
                        return
                    end if
                end do
            end if
        else
            print *, '    (skipped: requires 2+ ranks)'
        end if

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

end program test_fortran_communication
