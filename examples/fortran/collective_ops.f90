! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file collective_ops.f90
!> @brief Collective communication operations with DTL Fortran bindings
!>
!> Demonstrates:
!> - dtl_broadcast for root-to-all distribution
!> - dtl_allreduce for global reduction
!> - dtl_barrier for synchronization
!> - dtl_gather / dtl_scatter for data collection/distribution
!>
!> Run:
!>   mpirun -np 4 ./fortran_collective_ops

program collective_ops
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    integer(c_int32_t), target :: bcast_val
    real(c_double), target :: local_val, global_sum
    integer(c_int32_t), target :: send_val, recv_val
    integer(c_int32_t), allocatable, target :: gathered(:), scattered(:)
    integer :: i

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create context'
        stop 1
    end if

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (rank == 0) then
        print '(A)', 'DTL Collective Operations (Fortran)'
        print '(A)', '====================================='
        print '(A,I0,A)', 'Running with ', num_ranks, ' ranks'
        print '(A)', ''
    end if

    status = dtl_barrier(ctx)

    ! 1. Broadcast: rank 0 sends 42 to all
    if (rank == 0) print '(A)', '1. Broadcast:'
    status = dtl_barrier(ctx)

    if (rank == 0) then
        bcast_val = 42
    else
        bcast_val = 0
    end if

    print '(A,I0,A,I0)', '  Rank ', rank, ' before: ', bcast_val

    status = dtl_broadcast(ctx, c_loc(bcast_val), 1_c_int64_t, DTL_DTYPE_INT32, 0)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Broadcast failed'
        stop 1
    end if

    print '(A,I0,A,I0)', '  Rank ', rank, ' after:  ', bcast_val
    status = dtl_barrier(ctx)

    ! 2. Allreduce: sum of rank values
    if (rank == 0) then
        print '(A)', ''
        print '(A)', '2. Allreduce (sum):'
    end if
    status = dtl_barrier(ctx)

    local_val = real(rank + 1, c_double)
    global_sum = 0.0_c_double

    status = dtl_allreduce(ctx, c_loc(local_val), c_loc(global_sum), &
                           1_c_int64_t, DTL_DTYPE_FLOAT64, DTL_OP_SUM)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Allreduce failed'
        stop 1
    end if

    print '(A,I0,A,F6.1,A,F6.1)', '  Rank ', rank, &
        ': local=', local_val, ', global_sum=', global_sum
    status = dtl_barrier(ctx)

    if (rank == 0) then
        block
            real(c_double) :: expected
            expected = real(num_ranks * (num_ranks + 1), c_double) / 2.0_c_double
            print '(A,F6.1,A)', '  Expected: ', expected, &
                merge(' -> OK  ', ' -> FAIL', abs(global_sum - expected) < 1.0e-10_c_double)
        end block
    end if
    status = dtl_barrier(ctx)

    ! 3. Gather: each rank sends rank*10 to root
    if (rank == 0) then
        print '(A)', ''
        print '(A)', '3. Gather:'
    end if
    status = dtl_barrier(ctx)

    send_val = rank * 10

    if (rank == 0) then
        allocate(gathered(num_ranks))
    else
        allocate(gathered(1))
    end if

    status = dtl_gather(ctx, c_loc(send_val), 1_c_int64_t, DTL_DTYPE_INT32, &
                        c_loc(gathered(1)), 1_c_int64_t, DTL_DTYPE_INT32, 0)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Gather failed'
        stop 1
    end if

    if (rank == 0) then
        print '(A)', '  Root gathered: ['
        do i = 1, num_ranks
            if (i > 1) then
                write(*, '(A,I0)', advance='no') ', ', gathered(i)
            else
                write(*, '(A,I0)', advance='no') '  ', gathered(i)
            end if
        end do
        print '(A)', ']'
    end if

    deallocate(gathered)
    status = dtl_barrier(ctx)

    ! 4. Scatter: root distributes values
    if (rank == 0) then
        print '(A)', ''
        print '(A)', '4. Scatter:'
        allocate(scattered(num_ranks))
        do i = 1, num_ranks
            scattered(i) = i * 100
        end do
    else
        allocate(scattered(1))
    end if

    recv_val = 0
    status = dtl_scatter(ctx, c_loc(scattered(1)), 1_c_int64_t, DTL_DTYPE_INT32, &
                         c_loc(recv_val), 1_c_int64_t, DTL_DTYPE_INT32, 0)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Scatter failed'
        stop 1
    end if

    print '(A,I0,A,I0)', '  Rank ', rank, ' received: ', recv_val

    deallocate(scattered)
    status = dtl_barrier(ctx)

    ! Cleanup
    call dtl_context_destroy(ctx)

    if (rank == 0) then
        print '(A)', ''
        print '(A)', 'Done!'
    end if

end program collective_ops
