! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file fortran_vector_demo.f90
!> @brief DTL Fortran Vector Demo
!>
!> This example demonstrates:
!> - Creating a distributed vector
!> - Accessing local data via Fortran pointers
!> - Computing a distributed sum
!> - Proper resource cleanup
!>
!> Build: gfortran -o vector_demo fortran_vector_demo.f90 -ldtl_fortran -ldtl_c
!> Run:   mpirun -np 4 ./vector_demo

program fortran_vector_demo
    use dtl
    implicit none
    
    ! Handle variables
    type(c_ptr) :: ctx, vec
    
    ! Status and rank info
    integer(c_int) :: status, rank, nprocs
    
    ! Vector size info
    integer(c_int64_t) :: global_size, local_size, local_offset
    integer(c_int64_t) :: i
    
    ! Data pointer for local access
    real(c_double), pointer :: data(:)
    type(c_ptr) :: data_ptr
    
    ! For computing sum
    real(c_double) :: local_sum, expected_sum
    
    ! =========================================================================
    ! Step 1: Create context
    ! =========================================================================
    
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) then
        print *, 'ERROR: Failed to create context'
        print *, '  Message: ', dtl_get_error_message(status)
        stop 1
    end if
    
    rank = dtl_context_rank(ctx)
    nprocs = dtl_context_size(ctx)
    
    if (rank == 0) then
        print *, '================================================'
        print *, ' DTL Fortran Vector Demo'
        print *, '================================================'
        print '(A,I0)', '  Number of ranks: ', nprocs
        print *, ''
    end if
    
    ! Barrier to ensure header prints first
    status = dtl_context_barrier(ctx)
    
    ! =========================================================================
    ! Step 2: Create distributed vector
    ! =========================================================================
    
    global_size = 10000_c_int64_t
    
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
    if (status /= DTL_SUCCESS) then
        print *, 'ERROR: Failed to create vector'
        print *, '  Message: ', dtl_get_error_message(status)
        call dtl_context_destroy(ctx)
        stop 1
    end if
    
    ! Query distribution info
    local_size = dtl_vector_local_size(vec)
    local_offset = dtl_vector_local_offset(vec)
    
    print '(A,I0,A,I0,A,I0)', &
        '  Rank ', rank, ': local_size=', local_size, ', offset=', local_offset
    
    ! =========================================================================
    ! Step 3: Access local data and fill with values
    ! =========================================================================
    
    ! Get mutable pointer to local data
    data_ptr = dtl_vector_local_data_mut(vec)
    
    if (.not. c_associated(data_ptr)) then
        print *, 'ERROR: Failed to get local data pointer'
        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        stop 1
    end if
    
    ! Convert C pointer to Fortran array pointer
    call c_f_pointer(data_ptr, data, [local_size])
    
    ! Fill with global indices (0-based for consistency with C)
    do i = 1, local_size
        data(i) = real(local_offset + i - 1, kind=c_double)
    end do
    
    ! =========================================================================
    ! Step 4: Compute local sum
    ! =========================================================================
    
    local_sum = sum(data)
    
    print '(A,I0,A,F15.1)', '  Rank ', rank, ': local_sum = ', local_sum
    
    ! Barrier before final output
    status = dtl_context_barrier(ctx)
    
    ! =========================================================================
    ! Step 5: Verify result (on root)
    ! =========================================================================
    
    if (rank == 0) then
        print *, ''
        print *, 'Verification:'
        
        ! Expected: sum of 0..(N-1) = N*(N-1)/2
        expected_sum = real(global_size, c_double) * &
                       real(global_size - 1, c_double) / 2.0d0
        
        print '(A,I0)', '  Global size: ', global_size
        print '(A,F15.1)', '  Expected global sum: ', expected_sum
        print *, ''
        print *, 'Note: To get global sum, use DTL collective operations'
        print *, '      (not wrapped in this minimal example)'
        print *, ''
    end if
    
    ! =========================================================================
    ! Step 6: Cleanup
    ! =========================================================================
    
    ! Nullify Fortran pointer (important: DTL owns the memory)
    nullify(data)
    
    ! Destroy vector first (before context)
    call dtl_vector_destroy(vec)
    
    ! Destroy context
    call dtl_context_destroy(ctx)
    
    if (rank == 0) then
        print *, '================================================'
        print *, ' Demo completed successfully!'
        print *, '================================================'
    end if
    
end program fortran_vector_demo
