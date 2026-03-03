! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file rma_operations.f90
!> @brief RMA Put/Get operations with DTL Fortran bindings
!>
!> Demonstrates:
!> - dtl_window_create / dtl_window_destroy for window lifecycle
!> - dtl_window_fence for active-target synchronization
!> - dtl_rma_put / dtl_rma_get for one-sided data transfer
!>
!> Note: RMA may fail on WSL2 OpenMPI 4.1.6.
!>
!> Run:
!>   mpirun -np 2 ./fortran_rma_operations

program rma_operations
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx, win
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    integer(c_int32_t), target :: local_buf(4)
    integer(c_int32_t), target :: put_val, get_val
    integer :: i

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) stop 1

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (num_ranks < 2) then
        if (rank == 0) print '(A)', 'This example requires at least 2 ranks.'
        call dtl_context_destroy(ctx)
        stop 1
    end if

    if (rank == 0) then
        print '(A)', 'DTL RMA Operations (Fortran)'
        print '(A)', '=============================='
        print '(A,I0,A)', 'Running with ', num_ranks, ' ranks'
        print '(A)', ''
    end if
    status = dtl_barrier(ctx)

    ! Initialize local buffer
    do i = 1, 4
        local_buf(i) = rank * 100 + i - 1
    end do

    ! Create RMA window
    status = dtl_window_create(ctx, c_loc(local_buf(1)), &
                               int(4 * c_sizeof(local_buf(1)), c_int64_t), win)
    if (.not. is_success(status)) then
        print '(A,I0,A)', 'Rank ', rank, ': Window creation failed'
        call dtl_context_destroy(ctx)
        stop 1
    end if

    ! Initial fence
    status = dtl_window_fence(win)

    ! 1. Put: rank 0 writes 999 to rank 1's buffer[0]
    if (rank == 0) then
        print '(A)', '1. RMA Put (rank 0 -> rank 1):'
        put_val = 999
        status = dtl_rma_put(win, 1, 0_c_int64_t, c_loc(put_val), &
                             int(c_sizeof(put_val), c_int64_t))
        if (is_success(status)) then
            print '(A,I0,A)', '  Rank 0: put value ', put_val, ' to rank 1'
        else
            print '(A)', '  Rank 0: put failed'
        end if
    end if

    ! Fence to complete put
    status = dtl_window_fence(win)

    if (rank == 1) then
        print '(A,I0,A)', '  Rank 1: buffer(1) = ', local_buf(1), ' (expected 999)'
    end if
    status = dtl_barrier(ctx)

    ! 2. Get: rank 1 reads rank 0's buffer[2]
    if (rank == 0) then
        print '(A)', ''
        print '(A)', '2. RMA Get (rank 1 reads rank 0):'
    end if
    status = dtl_barrier(ctx)

    if (rank == 1) then
        get_val = -1
        status = dtl_rma_get(win, 0, int(2 * c_sizeof(get_val), c_int64_t), &
                             c_loc(get_val), int(c_sizeof(get_val), c_int64_t))
        if (is_success(status)) then
            print '(A,I0,A,I0,A)', '  Rank 1: got value ', get_val, &
                ' from rank 0 (expected ', 0 * 100 + 2, ')'
        else
            print '(A)', '  Rank 1: get failed'
        end if
    end if

    ! Final fence
    status = dtl_window_fence(win)

    ! Cleanup
    call dtl_window_destroy(win)
    call dtl_context_destroy(ctx)

    if (rank == 0) then
        print '(A)', ''
        print '(A)', 'Done!'
    end if

end program rma_operations
