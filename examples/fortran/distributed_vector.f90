! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file distributed_vector.f90
!> @brief Distributed vector operations with DTL Fortran bindings
!>
!> Demonstrates:
!> - Creating distributed vectors
!> - Accessing local data
!> - Computing local statistics
!>
!> Compile with:
!>   gfortran -c dtl_fortran.f90
!>   gfortran -L../../build/src/bindings/c distributed_vector.f90 dtl_fortran.o -ldtl_c -o distributed_vector
!>
!> Run with:
!>   ./distributed_vector
!>   mpirun -np 4 ./distributed_vector

program distributed_vector_example
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx, vec
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, size
    integer(c_int64_t) :: global_size, local_size, local_offset
    real(c_double), pointer :: data(:)
    type(c_ptr) :: data_ptr
    integer(c_int64_t) :: i
    real(c_double) :: local_sum, local_min, local_max, local_mean
    real(c_double), target :: fill_value

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create context'
        stop 1
    end if

    rank = dtl_context_rank(ctx)
    size = dtl_context_size(ctx)

    print '(A,I0,A,I0,A)', 'Rank ', rank, ' of ', size, ': Distributed Vector Example'

    ! Create a distributed vector of 1000 doubles
    global_size = 1000
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create vector'
        stop 1
    end if

    ! Query vector properties
    local_size = dtl_vector_local_size(vec)
    local_offset = dtl_vector_local_offset(vec)

    print '(A,I0,A,I0,A,I0,A,I0)', 'Rank ', rank, ': global_size=', &
        dtl_vector_global_size(vec), ', local_size=', local_size, &
        ', local_offset=', local_offset

    ! Get local data pointer and convert to Fortran pointer
    data_ptr = dtl_vector_local_data_mut(vec)
    call c_f_pointer(data_ptr, data, [local_size])

    ! Initialize: each element is its global index
    do i = 1, local_size
        data(i) = real(local_offset + i - 1, c_double)
    end do

    ! Compute local statistics
    local_sum = 0.0_c_double
    local_min = data(1)
    local_max = data(1)

    do i = 1, local_size
        local_sum = local_sum + data(i)
        if (data(i) < local_min) local_min = data(i)
        if (data(i) > local_max) local_max = data(i)
    end do

    local_mean = local_sum / real(local_size, c_double)

    print '(A,I0,A,F10.2,A,F10.2,A,F10.2,A,F10.2)', &
        'Rank ', rank, ': sum=', local_sum, ', min=', local_min, &
        ', max=', local_max, ', mean=', local_mean

    ! Synchronize
    status = dtl_vector_barrier(vec)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Barrier failed'
        stop 1
    end if

    ! Fill with a constant value
    fill_value = 42.0_c_double
    status = dtl_vector_fill_local(vec, c_loc(fill_value))
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Fill failed'
        stop 1
    end if

    ! Verify fill
    if (all(abs(data - fill_value) < 1.0e-10_c_double)) then
        print '(A,I0,A)', 'Rank ', rank, ': fill verification PASSED'
    else
        print '(A,I0,A)', 'Rank ', rank, ': fill verification FAILED'
    end if

    ! Cleanup
    call dtl_vector_destroy(vec)
    call dtl_context_destroy(ctx)

    if (rank == 0) then
        print '(A)', ''
        print '(A)', 'Done!'
    end if

end program distributed_vector_example
