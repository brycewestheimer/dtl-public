! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file fortran_hello_dtl.f90
!> @brief Minimal DTL Fortran example - Hello World
!>
!> This example demonstrates:
!> - Creating a DTL context
!> - Querying rank and size
!> - Performing a barrier synchronization
!> - Proper cleanup

program fortran_hello_dtl
    use dtl
    implicit none
    
    type(c_ptr) :: ctx
    integer(c_int) :: status, rank, nprocs
    
    ! Create DTL context with default options
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) then
        print *, 'ERROR: Failed to create context - ', dtl_get_error_message(status)
        stop 1
    end if
    
    ! Get rank and size
    rank = dtl_context_rank(ctx)
    nprocs = dtl_context_size(ctx)
    
    ! Print greeting from each rank
    print '(A,I0,A,I0,A)', 'Hello from rank ', rank, ' of ', nprocs, ' (DTL Fortran)'
    
    ! Check device support
    if (dtl_context_has_device(ctx) == 1) then
        print '(A,I0,A,I0)', '  Rank ', rank, ' has GPU device: ', dtl_context_device_id(ctx)
    else
        print '(A,I0,A)', '  Rank ', rank, ' is CPU-only'
    end if
    
    ! Barrier to synchronize output
    status = dtl_context_barrier(ctx)
    if (status /= DTL_SUCCESS) then
        print *, 'ERROR: Barrier failed - ', dtl_get_error_message(status)
        stop 1
    end if
    
    ! Root prints completion message
    if (dtl_context_is_root(ctx) == 1) then
        print *, ''
        print *, 'DTL Fortran bindings working correctly!'
    end if
    
    ! Cleanup
    call dtl_context_destroy(ctx)
    
end program fortran_hello_dtl
