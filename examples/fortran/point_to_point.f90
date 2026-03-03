! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file point_to_point.f90
!> @brief Point-to-point communication with DTL Fortran bindings
!>
!> Demonstrates:
!> - dtl_send / dtl_recv for blocking P2P
!> - dtl_sendrecv for combined send/receive (ring pattern)
!> - Verification of received values
!>
!> Run:
!>   mpirun -np 4 ./fortran_point_to_point

program point_to_point
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    integer(c_int32_t), target :: send_val, recv_val
    integer(c_int32_t) :: next_rank, prev_rank
    integer(c_int32_t), target :: ring_send, ring_recv
    integer(c_int32_t) :: expected

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create context'
        stop 1
    end if

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (num_ranks < 2) then
        if (rank == 0) print '(A)', 'This example requires at least 2 ranks.'
        call dtl_context_destroy(ctx)
        stop 1
    end if

    if (rank == 0) then
        print '(A)', 'DTL Point-to-Point Communication (Fortran)'
        print '(A)', '============================================'
        print '(A,I0,A)', 'Running with ', num_ranks, ' ranks'
        print '(A)', ''
    end if
    status = dtl_barrier(ctx)

    ! =========================================================================
    ! 1. Simple send/recv between rank 0 and rank 1
    ! =========================================================================
    if (rank == 0) print '(A)', '1. Send/Recv (rank 0 -> rank 1):'
    status = dtl_barrier(ctx)

    if (rank == 0) then
        send_val = 42
        status = dtl_send(ctx, c_loc(send_val), 1_c_int64_t, DTL_DTYPE_INT32, 1, 0)
        if (.not. is_success(status)) then
            print '(A)', 'ERROR: Send failed'
            stop 1
        end if
        print '(A,I0)', '  Rank 0 sent: ', send_val
    else if (rank == 1) then
        recv_val = 0
        status = dtl_recv(ctx, c_loc(recv_val), 1_c_int64_t, DTL_DTYPE_INT32, 0, 0)
        if (.not. is_success(status)) then
            print '(A)', 'ERROR: Recv failed'
            stop 1
        end if
        print '(A,I0,A)', '  Rank 1 received: ', recv_val, &
            merge(' -> OK  ', ' -> FAIL', recv_val == 42)
    end if
    status = dtl_barrier(ctx)

    ! =========================================================================
    ! 2. Sendrecv ring: each rank exchanges with next/prev
    ! =========================================================================
    if (rank == 0) then
        print '(A)', ''
        print '(A)', '2. Sendrecv Ring:'
    end if
    status = dtl_barrier(ctx)

    next_rank = mod(rank + 1, num_ranks)
    prev_rank = mod(rank + num_ranks - 1, num_ranks)

    ring_send = rank * 100
    ring_recv = -1

    status = dtl_sendrecv(ctx, &
        c_loc(ring_send), 1_c_int64_t, DTL_DTYPE_INT32, next_rank, 10, &
        c_loc(ring_recv), 1_c_int64_t, DTL_DTYPE_INT32, prev_rank, 10)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Sendrecv failed'
        stop 1
    end if

    expected = prev_rank * 100
    print '(A,I0,A,I0,A,I0,A,I0,A,I0,A)', &
        '  Rank ', rank, ': sent ', ring_send, ' to rank ', next_rank, &
        ', received ', ring_recv, ' from rank ', prev_rank, &
        merge(' -> OK  ', ' -> FAIL', ring_recv == expected)
    status = dtl_barrier(ctx)

    ! =========================================================================
    ! 3. Bidirectional exchange: rank 0 <-> rank 1
    ! =========================================================================
    if (rank == 0) then
        print '(A)', ''
        print '(A)', '3. Bidirectional exchange (rank 0 <-> rank 1):'
    end if
    status = dtl_barrier(ctx)

    if (rank == 0) then
        send_val = 111
        recv_val = 0
        status = dtl_sendrecv(ctx, &
            c_loc(send_val), 1_c_int64_t, DTL_DTYPE_INT32, 1, 20, &
            c_loc(recv_val), 1_c_int64_t, DTL_DTYPE_INT32, 1, 21)
        if (.not. is_success(status)) then
            print '(A)', 'ERROR: Sendrecv failed'
            stop 1
        end if
        print '(A,I0,A,I0,A)', '  Rank 0: sent ', send_val, &
            ', received ', recv_val, merge(' -> OK  ', ' -> FAIL', recv_val == 222)
    else if (rank == 1) then
        send_val = 222
        recv_val = 0
        status = dtl_sendrecv(ctx, &
            c_loc(send_val), 1_c_int64_t, DTL_DTYPE_INT32, 0, 21, &
            c_loc(recv_val), 1_c_int64_t, DTL_DTYPE_INT32, 0, 20)
        if (.not. is_success(status)) then
            print '(A)', 'ERROR: Sendrecv failed'
            stop 1
        end if
        print '(A,I0,A,I0,A)', '  Rank 1: sent ', send_val, &
            ', received ', recv_val, merge(' -> OK  ', ' -> FAIL', recv_val == 111)
    end if
    status = dtl_barrier(ctx)

    ! Cleanup
    call dtl_context_destroy(ctx)

    if (rank == 0) then
        print '(A)', ''
        print '(A)', 'Done!'
    end if

end program point_to_point
