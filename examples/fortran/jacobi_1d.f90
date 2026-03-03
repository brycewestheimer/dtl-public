! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file jacobi_1d.f90
!> @brief 1D Jacobi iterative solver with DTL Fortran bindings
!>
!> Solves u''(x) = 0 with boundary conditions u(0) = 1, u(L) = 0.
!> Uses Jacobi iteration with halo exchange via dtl_send / dtl_recv.
!>
!> Demonstrates:
!> - Halo exchange using dtl_send / dtl_recv
!> - Convergence checking via dtl_allreduce with DTL_OP_MAX
!> - Distributed iterative solver pattern
!>
!> Run:
!>   mpirun -np 4 ./fortran_jacobi_1d

program jacobi_1d
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    integer :: global_n, max_iter, local_n, remainder_n
    real(c_double) :: tol
    real(c_double), allocatable, target :: u(:), u_new(:)
    real(c_double), target :: local_diff, global_diff
    integer :: iter, i
    integer :: halo_tag

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Failed to create context'
        stop 1
    end if

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (rank == 0) then
        print '(A)', 'DTL 1D Jacobi Solver (Fortran)'
        print '(A)', '==============================='
        print '(A,I0)', 'Ranks: ', num_ranks
    end if
    status = dtl_barrier(ctx)

    ! Problem setup
    global_n = 100       ! Interior points
    max_iter = 10000
    tol = 1.0e-8_c_double
    halo_tag = 10

    ! Partition interior points among ranks
    local_n = global_n / num_ranks
    remainder_n = mod(global_n, num_ranks)
    if (rank < remainder_n) local_n = local_n + 1

    ! Allocate arrays with halo cells: index 0 = left halo, local_n+1 = right halo
    allocate(u(0:local_n + 1))
    allocate(u_new(0:local_n + 1))
    u = 0.0_c_double
    u_new = 0.0_c_double

    ! Boundary conditions: u(0) = 1.0 on leftmost rank
    if (rank == 0) then
        u(0) = 1.0_c_double
        u_new(0) = 1.0_c_double
    end if
    ! u(L) = 0.0 on rightmost rank (already zero)

    if (rank == 0) then
        print '(A,I0,A)', 'Grid: ', global_n, ' interior points'
        print '(A)', 'BCs: u(0)=1, u(L)=0'
        print '(A,ES8.0)', 'Tolerance: ', tol
        print '(A)', ''
    end if
    status = dtl_barrier(ctx)

    global_diff = 0.0_c_double

    do iter = 1, max_iter
        ! --- Halo exchange ---
        ! Send right boundary to right neighbor
        if (rank < num_ranks - 1) then
            status = dtl_send(ctx, c_loc(u(local_n)), 1_c_int64_t, &
                              DTL_DTYPE_FLOAT64, rank + 1, halo_tag)
        end if
        if (rank > 0) then
            status = dtl_recv(ctx, c_loc(u(0)), 1_c_int64_t, &
                              DTL_DTYPE_FLOAT64, rank - 1, halo_tag)
        end if

        ! Send left boundary to left neighbor
        if (rank > 0) then
            status = dtl_send(ctx, c_loc(u(1)), 1_c_int64_t, &
                              DTL_DTYPE_FLOAT64, rank - 1, halo_tag + 1)
        end if
        if (rank < num_ranks - 1) then
            status = dtl_recv(ctx, c_loc(u(local_n + 1)), 1_c_int64_t, &
                              DTL_DTYPE_FLOAT64, rank + 1, halo_tag + 1)
        end if

        ! --- Jacobi update ---
        local_diff = 0.0_c_double
        do i = 1, local_n
            u_new(i) = 0.5_c_double * (u(i - 1) + u(i + 1))
            local_diff = max(local_diff, abs(u_new(i) - u(i)))
        end do

        ! Copy new to old
        u(1:local_n) = u_new(1:local_n)

        ! Check convergence: global max diff
        status = dtl_allreduce(ctx, c_loc(local_diff), c_loc(global_diff), &
                               1_c_int64_t, DTL_DTYPE_FLOAT64, DTL_OP_MAX)

        if (global_diff < tol) exit
    end do

    status = dtl_barrier(ctx)

    if (rank == 0) then
        print '(A,I0,A)', 'Converged after ', iter - 1, ' iterations'
        print '(A,ES12.4)', 'Final max diff: ', global_diff
        print '(A)', ''
    end if

    ! Print solution samples
    block
        integer(c_int32_t) :: r
        do r = 0, num_ranks - 1
            if (rank == r) then
                print '(A,I0,A,F10.6,A,F10.6)', &
                    '  Rank ', rank, ': u[first]=', u(1), ', u[last]=', u(local_n)
            end if
            status = dtl_barrier(ctx)
        end do
    end block

    if (rank == 0) then
        print '(A)', ''
        print '(A)', 'Expected: linear from 1.0 to 0.0'
        if (global_diff < tol) then
            print '(A)', 'SUCCESS: Solver converged'
        else
            print '(A)', 'FAILURE: Solver did not converge'
        end if
    end if

    deallocate(u)
    deallocate(u_new)
    call dtl_context_destroy(ctx)

end program jacobi_1d
