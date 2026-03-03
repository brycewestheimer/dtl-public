! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file distributed_tensor.f90
!> @brief Distributed tensor ND operations with DTL Fortran bindings
!>
!> Demonstrates:
!> - dtl_tensor_create for multi-dimensional distributed data
!> - shape_2d helper for 2D shape creation
!> - dtl_tensor_local_data_mut for data access
!> - dtl_allreduce for cross-rank aggregation
!>
!> Run:
!>   mpirun -np 4 ./fortran_distributed_tensor

program distributed_tensor_example
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx, tensor
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    type(dtl_shape) :: global_shape
    integer(c_int64_t) :: nrows, ncols, local_size, global_size, local_rows
    real(c_double), pointer :: data(:)
    type(c_ptr) :: data_ptr
    integer(c_int64_t) :: i, global_idx, row, col
    real(c_double) :: local_sum_sq, global_sum_sq, frobenius_norm
    real(c_double), target :: send_val, recv_val
    real(c_double) :: expected_sq, expected_norm, val

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create context'
        stop 1
    end if

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (rank == 0) then
        print '(A)', 'DTL Distributed Tensor (Fortran)'
        print '(A)', '=================================='
        print '(A,I0,A)', 'Running with ', num_ranks, ' ranks'
        print '(A)', ''
    end if
    status = dtl_barrier(ctx)

    ! Create 2D tensor (100 x 64)
    nrows = 100
    ncols = 64
    global_shape = shape_2d(nrows, ncols)

    status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, global_shape, tensor)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create tensor'
        stop 1
    end if

    ! Query properties
    local_size = dtl_tensor_local_size(tensor)
    global_size = dtl_tensor_global_size(tensor)
    local_rows = local_size / ncols

    print '(A,I0,A,I0,A,I0,A)', 'Rank ', rank, ': local_size=', local_size, &
        ' (', local_rows, ' rows)'
    status = dtl_barrier(ctx)

    ! Get data pointer
    data_ptr = dtl_tensor_local_data_mut(tensor)
    call c_f_pointer(data_ptr, data, [local_size])

    ! Fill: element = row * 100 + col
    ! Approximate global offset (assumes even distribution)
    do i = 1, local_size
        global_idx = int(rank, c_int64_t) * (nrows / int(num_ranks, c_int64_t)) * ncols + (i - 1)
        row = global_idx / ncols
        col = mod(global_idx, ncols)
        data(i) = real(row * 100 + col, c_double)
    end do

    ! Compute Frobenius norm: sqrt(sum of squares)
    local_sum_sq = 0.0_c_double
    do i = 1, local_size
        local_sum_sq = local_sum_sq + data(i) * data(i)
    end do

    ! Allreduce sum of squares
    send_val = local_sum_sq
    recv_val = 0.0_c_double
    status = dtl_allreduce(ctx, c_loc(send_val), c_loc(recv_val), &
                           1_c_int64_t, DTL_DTYPE_FLOAT64, DTL_OP_SUM)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Allreduce failed'
        stop 1
    end if

    global_sum_sq = recv_val
    frobenius_norm = sqrt(global_sum_sq)

    if (rank == 0) then
        print '(A)', ''
        print '(A,I0,A,I0)', 'Tensor shape: ', nrows, ' x ', ncols
        print '(A,I0)', 'Total elements: ', global_size
        print '(A,F12.4)', 'Frobenius norm: ', frobenius_norm

        ! Compute expected norm
        expected_sq = 0.0_c_double
        do i = 0, nrows - 1
            do col = 0, ncols - 1
                val = real(i * 100 + col, c_double)
                expected_sq = expected_sq + val * val
            end do
        end do
        expected_norm = sqrt(expected_sq)

        print '(A,F12.4)', 'Expected norm:  ', expected_norm
        if (abs(frobenius_norm - expected_norm) < 1.0_c_double) then
            print '(A)', 'SUCCESS'
        else
            print '(A)', 'FAILURE'
        end if
    end if

    ! Cleanup
    call dtl_tensor_destroy(tensor)
    call dtl_context_destroy(ctx)

end program distributed_tensor_example
