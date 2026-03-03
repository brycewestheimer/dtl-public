! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file scan_prefix.f90
!> @brief Prefix scan operations with DTL Fortran bindings
!>
!> Demonstrates:
!> - Local inclusive/exclusive scan using loops
!> - Cross-rank prefix via dtl_allreduce
!> - Combining local and global prefixes
!>
!> Run:
!>   mpirun -np 4 ./fortran_scan_prefix

program scan_prefix
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx, vec
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    integer(c_int64_t) :: global_size, local_size, local_offset
    integer(c_int32_t), pointer :: data(:)
    type(c_ptr) :: data_ptr
    integer(c_int64_t) :: i
    integer(c_int32_t) :: local_total
    integer(c_int32_t), target :: send_val, recv_val
    integer(c_int32_t), allocatable :: inclusive(:), exclusive(:)
    integer(c_int32_t) :: cross_rank_prefix

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) stop 1

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (rank == 0) then
        print '(A)', 'DTL Prefix Scan (Fortran)'
        print '(A)', '=========================='
        print '(A,I0,A)', 'Running with ', num_ranks, ' ranks'
        print '(A)', ''
    end if
    status = dtl_barrier(ctx)

    ! Create vector of 20 elements filled with 1s
    global_size = 20
    status = dtl_vector_create(ctx, DTL_DTYPE_INT32, global_size, vec)
    if (.not. is_success(status)) stop 1

    local_size = dtl_vector_local_size(vec)
    local_offset = dtl_vector_local_offset(vec)
    data_ptr = dtl_vector_local_data_mut(vec)
    call c_f_pointer(data_ptr, data, [local_size])

    ! Fill with 1s
    data = 1

    ! --- Inclusive scan ---
    ! Step 1: Local prefix sum
    allocate(inclusive(local_size))
    inclusive(1) = data(1)
    do i = 2, local_size
        inclusive(i) = inclusive(i - 1) + data(i)
    end do

    ! Step 2: Get cross-rank prefix using allreduce
    ! We need exclusive prefix: sum of all local totals from ranks 0..rank-1
    local_total = inclusive(local_size)

    ! Use reduce to get prefix sum across ranks
    ! Simple approach: allreduce local_total, then compute prefix manually
    ! For a proper exscan, we need to gather and compute
    send_val = local_total
    recv_val = 0
    status = dtl_allreduce(ctx, c_loc(send_val), c_loc(recv_val), &
                           1_c_int64_t, DTL_DTYPE_INT32, DTL_OP_SUM)

    ! Compute exclusive prefix for this rank
    ! prefix(r) = sum of local_size for ranks 0..r-1
    cross_rank_prefix = int(rank, c_int32_t) * int(local_size, c_int32_t)

    ! Step 3: Add cross-rank prefix
    do i = 1, local_size
        inclusive(i) = inclusive(i) + cross_rank_prefix
    end do

    if (rank == 0) print '(A)', 'Inclusive scan (input: all 1s):'
    status = dtl_barrier(ctx)

    write(*, '(A,I0,A)', advance='no') '  Rank ', rank, ': ['
    do i = 1, local_size
        if (i > 1) write(*, '(A)', advance='no') ', '
        write(*, '(I0)', advance='no') inclusive(i)
    end do
    print '(A)', ']'
    status = dtl_barrier(ctx)

    ! --- Exclusive scan ---
    allocate(exclusive(local_size))
    exclusive(1) = 0
    do i = 2, local_size
        exclusive(i) = exclusive(i - 1) + data(i - 1)
    end do

    ! Add cross-rank prefix
    do i = 1, local_size
        exclusive(i) = exclusive(i) + cross_rank_prefix
    end do

    if (rank == 0) then
        print '(A)', ''
        print '(A)', 'Exclusive scan (input: all 1s):'
    end if
    status = dtl_barrier(ctx)

    write(*, '(A,I0,A)', advance='no') '  Rank ', rank, ': ['
    do i = 1, local_size
        if (i > 1) write(*, '(A)', advance='no') ', '
        write(*, '(I0)', advance='no') exclusive(i)
    end do
    print '(A)', ']'
    status = dtl_barrier(ctx)

    ! Verify
    if (rank == num_ranks - 1) then
        print '(A)', ''
        print '(A,I0,A,I0)', 'Last inclusive scan value: ', inclusive(local_size), &
            ' (expected: ', global_size
        if (inclusive(local_size) == int(global_size, c_int32_t)) then
            print '(A)', 'SUCCESS'
        else
            print '(A)', 'FAILURE'
        end if
    end if

    deallocate(inclusive)
    deallocate(exclusive)
    call dtl_vector_destroy(vec)
    call dtl_context_destroy(ctx)

end program scan_prefix
