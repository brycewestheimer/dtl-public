! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file hello_dtl.f90
!> @brief Basic DTL Fortran example
!>
!> Demonstrates:
!> - Creating a DTL context
!> - Querying rank and size
!> - Feature detection
!>
!> Build with CMake:
!>   cmake -B build -DDTL_BUILD_C_BINDINGS=ON -DDTL_BUILD_FORTRAN=ON -DDTL_BUILD_EXAMPLES=ON
!>   cmake --build build
!>
!> Run with:
!>   ./build/examples/fortran/hello_dtl
!>   mpirun -np 4 ./build/examples/fortran/hello_dtl

program hello_dtl
    use, intrinsic :: iso_c_binding
    use dtl
    implicit none

    type(c_ptr) :: ctx
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, size

    ! Print version info
    print '(A)', 'DTL Fortran Bindings Example'
    print '(A)', '============================'
    print '(A,I0,A,I0,A,I0)', 'Version: ', dtl_version_major(), '.', &
        dtl_version_minor(), '.', dtl_version_patch()
    print '(A)', ''

    ! Check available backends
    print '(A)', 'Available backends:'
    print '(A,L1)', '  MPI:  ', dtl_has_mpi() /= 0
    print '(A,L1)', '  CUDA: ', dtl_has_cuda() /= 0
    print '(A)', ''

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create context'
        stop 1
    end if

    ! Query context properties
    rank = dtl_context_rank(ctx)
    size = dtl_context_size(ctx)

    print '(A)', 'Context created:'
    print '(A,I0)', '  Rank: ', rank
    print '(A,I0)', '  Size: ', size
    print '(A,L1)', '  Is root: ', dtl_context_is_root(ctx) /= 0

    ! Cleanup
    call dtl_context_destroy(ctx)

    print '(A)', ''
    print '(A)', 'Done!'

end program hello_dtl
