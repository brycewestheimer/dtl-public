! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file monte_carlo_pi.f90
!> @brief Monte Carlo Pi estimation with DTL Fortran bindings
!>
!> Demonstrates:
!> - dtl_allreduce for global summation
!> - Fortran random_number for sampling
!> - Distributed parallel computation
!>
!> Run:
!>   mpirun -np 4 ./fortran_monte_carlo_pi

program monte_carlo_pi
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    integer(c_int64_t) :: samples_per_rank, total_samples
    integer(c_int64_t) :: i, local_hits, global_hits
    real(c_double) :: x, y, local_pi, pi_estimate, error
    integer(c_int64_t), target :: send_hits, recv_hits
    real(c_double), parameter :: PI_EXACT = 3.14159265358979323846_c_double
    integer :: seed_array(8)

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) stop 1

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (rank == 0) then
        print '(A)', 'DTL Monte Carlo Pi Estimation (Fortran)'
        print '(A)', '========================================='
        print '(A,I0,A)', 'Running with ', num_ranks, ' ranks'
        print '(A)', ''
    end if
    status = dtl_barrier(ctx)

    ! Setup
    samples_per_rank = 1000000
    total_samples = samples_per_rank * int(num_ranks, c_int64_t)

    ! Initialize random seed (rank-specific)
    seed_array = 42 + rank * 12345
    call random_seed(put=seed_array)

    ! Sample random points
    local_hits = 0
    do i = 1, samples_per_rank
        call random_number(x)
        call random_number(y)
        if (x * x + y * y <= 1.0_c_double) then
            local_hits = local_hits + 1
        end if
    end do

    local_pi = 4.0_c_double * real(local_hits, c_double) / real(samples_per_rank, c_double)
    print '(A,I0,A,I0,A,I0,A,F10.6,A)', '  Rank ', rank, ': ', local_hits, &
        ' / ', samples_per_rank, ' hits (local pi ~ ', local_pi, ')'
    status = dtl_barrier(ctx)

    ! Global reduction
    send_hits = local_hits
    recv_hits = 0
    status = dtl_allreduce(ctx, c_loc(send_hits), c_loc(recv_hits), &
                           1_c_int64_t, DTL_DTYPE_INT64, DTL_OP_SUM)
    if (.not. is_success(status)) then
        print '(A)', 'ERROR: Allreduce failed'
        stop 1
    end if

    global_hits = recv_hits
    pi_estimate = 4.0_c_double * real(global_hits, c_double) / real(total_samples, c_double)
    error = abs(pi_estimate - PI_EXACT)

    if (rank == 0) then
        print '(A)', ''
        print '(A,I0)', 'Total samples: ', total_samples
        print '(A,I0)', 'Total hits:    ', global_hits
        print '(A,F12.8)', 'Pi estimate:   ', pi_estimate
        print '(A,F12.8)', 'Actual pi:     ', PI_EXACT
        print '(A,ES12.4)', 'Error:         ', error
        if (error < 0.01_c_double) then
            print '(A)', 'SUCCESS: Estimate within 0.01 tolerance'
        else
            print '(A)', 'WARNING: Estimate outside 0.01 tolerance'
        end if
    end if

    call dtl_context_destroy(ctx)

end program monte_carlo_pi
