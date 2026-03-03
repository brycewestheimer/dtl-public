! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file bench_fortran_api.f90
!> @brief DTL Fortran API benchmark
!>
!> Micro-benchmarks for the DTL Fortran bindings measuring:
!>   1. Context creation / destruction
!>   2. Vector creation / destruction (various sizes)
!>   3. Vector element access (write + read)
!>   4. Vector fill_local
!>   5. Barrier overhead (context barrier)
!>   6. Communicator barrier (dtl_barrier)
!>   7. Array creation / destruction
!>
!> Uses cpu_time for wall-clock measurement.  All timings are per-operation
!> averages reported in microseconds.
!>
!> Run:
!>   ./bench_fortran_api                      (single rank)
!>   mpirun -np 4 ./bench_fortran_api         (multi-rank)

program bench_fortran_api
    use dtl
    implicit none

    integer, parameter :: WARMUP  = 5
    integer, parameter :: N_ITER  = 100
    integer, parameter :: N_SIZES = 5

    integer(c_int64_t) :: sizes(N_SIZES)
    type(c_ptr) :: ctx
    integer(c_int) :: status, my_rank, my_size
    real(c_double) :: t0, t1
    integer :: it, si

    sizes = [100_c_int64_t, 1000_c_int64_t, 10000_c_int64_t, &
             100000_c_int64_t, 1000000_c_int64_t]

    ! ---- Create shared context ----
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) then
        print '(A)', 'FATAL: dtl_context_create_default failed'
        stop 1
    end if
    my_rank = dtl_context_rank(ctx)
    my_size = dtl_context_size(ctx)

    if (my_rank == 0) then
        print '(A)',    '================================================'
        print '(A)',    ' DTL Fortran API Benchmarks'
        print '(A,I0)', '   Ranks : ', my_size
        print '(A,I0)', '   Iters : ', N_ITER
        print '(A)',    '================================================'
        print *, ''
    end if

    ! ---- 1. Context create/destroy ----
    call bench_context_lifecycle()

    ! ---- 2. Vector create/destroy (various sizes) ----
    do si = 1, N_SIZES
        call bench_vector_lifecycle(sizes(si))
    end do

    ! ---- 3. Vector element access ----
    call bench_vector_access(10000_c_int64_t)

    ! ---- 4. Vector fill_local ----
    call bench_vector_fill(10000_c_int64_t)

    ! ---- 5. Context barrier ----
    call bench_context_barrier()

    ! ---- 6. Communicator barrier (dtl_barrier) ----
    call bench_comm_barrier()

    ! ---- 7. Array create/destroy ----
    call bench_array_lifecycle(10000_c_int64_t)

    ! ---- Cleanup ----
    call dtl_context_destroy(ctx)

    if (my_rank == 0) then
        print *, ''
        print '(A)', 'Benchmarks complete.'
    end if

contains

    ! ------------------------------------------------------------------
    ! Helper: report a single benchmark result (rank 0 only)
    ! ------------------------------------------------------------------
    subroutine report(label, elapsed_sec, iters)
        character(len=*), intent(in) :: label
        real(c_double), intent(in)   :: elapsed_sec
        integer, intent(in)          :: iters
        real(c_double) :: avg_us

        if (my_rank /= 0) return
        avg_us = elapsed_sec / real(iters, c_double) * 1.0d6
        print '(A,A,A,F12.2,A)', '  ', label, ':  avg = ', avg_us, ' us'
    end subroutine

    ! ------------------------------------------------------------------
    ! 1. Context creation + destruction
    ! ------------------------------------------------------------------
    subroutine bench_context_lifecycle()
        type(c_ptr) :: tmp_ctx
        integer(c_int) :: st

        ! Warmup
        do it = 1, WARMUP
            st = dtl_context_create_default(tmp_ctx)
            call dtl_context_destroy(tmp_ctx)
        end do

        call cpu_time(t0)
        do it = 1, N_ITER
            st = dtl_context_create_default(tmp_ctx)
            call dtl_context_destroy(tmp_ctx)
        end do
        call cpu_time(t1)
        call report('context create+destroy', t1 - t0, N_ITER)
    end subroutine

    ! ------------------------------------------------------------------
    ! 2. Vector creation + destruction for a given global_size
    ! ------------------------------------------------------------------
    subroutine bench_vector_lifecycle(gsize)
        integer(c_int64_t), intent(in) :: gsize
        type(c_ptr) :: vec
        integer(c_int) :: st
        character(len=60) :: label

        write(label, '(A,I0)') 'vector create+destroy (n=', gsize
        label = trim(label) // ')'

        ! Warmup
        do it = 1, WARMUP
            st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, gsize, vec)
            call dtl_vector_destroy(vec)
        end do

        call cpu_time(t0)
        do it = 1, N_ITER
            st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, gsize, vec)
            call dtl_vector_destroy(vec)
        end do
        call cpu_time(t1)
        call report(trim(label), t1 - t0, N_ITER)
    end subroutine

    ! ------------------------------------------------------------------
    ! 3. Vector element access — write + read local elements
    ! ------------------------------------------------------------------
    subroutine bench_vector_access(gsize)
        integer(c_int64_t), intent(in) :: gsize
        type(c_ptr) :: vec, dptr
        integer(c_int) :: st
        integer(c_int64_t) :: lsize, j
        real(c_double), pointer :: data(:)
        real(c_double) :: dummy

        st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, gsize, vec)
        if (st /= DTL_SUCCESS) return

        lsize = dtl_vector_local_size(vec)
        dptr = dtl_vector_local_data_mut(vec)
        call c_f_pointer(dptr, data, [lsize])

        ! Warmup
        do it = 1, WARMUP
            do j = 1, lsize
                data(j) = real(j, c_double)
            end do
        end do

        ! Timed: write
        call cpu_time(t0)
        do it = 1, N_ITER
            do j = 1, lsize
                data(j) = real(j, c_double)
            end do
        end do
        call cpu_time(t1)
        call report('vector write (n=10000)', t1 - t0, N_ITER)

        ! Timed: read (accumulate to prevent optimisation)
        dummy = 0.0_c_double
        call cpu_time(t0)
        do it = 1, N_ITER
            do j = 1, lsize
                dummy = dummy + data(j)
            end do
        end do
        call cpu_time(t1)
        call report('vector read  (n=10000)', t1 - t0, N_ITER)

        ! Use dummy to prevent dead-code elimination
        if (dummy < -1.0d30) print *, dummy

        nullify(data)
        call dtl_vector_destroy(vec)
    end subroutine

    ! ------------------------------------------------------------------
    ! 4. Vector fill_local
    ! ------------------------------------------------------------------
    subroutine bench_vector_fill(gsize)
        integer(c_int64_t), intent(in) :: gsize
        type(c_ptr) :: vec
        integer(c_int) :: st
        real(c_double), target :: fill_val

        st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, gsize, vec)
        if (st /= DTL_SUCCESS) return

        fill_val = 42.0_c_double

        ! Warmup
        do it = 1, WARMUP
            st = dtl_vector_fill_local(vec, c_loc(fill_val))
        end do

        call cpu_time(t0)
        do it = 1, N_ITER
            st = dtl_vector_fill_local(vec, c_loc(fill_val))
        end do
        call cpu_time(t1)
        call report('vector fill_local (n=10000)', t1 - t0, N_ITER)

        call dtl_vector_destroy(vec)
    end subroutine

    ! ------------------------------------------------------------------
    ! 5. Context barrier
    ! ------------------------------------------------------------------
    subroutine bench_context_barrier()
        integer(c_int) :: st

        ! Warmup
        do it = 1, WARMUP
            st = dtl_context_barrier(ctx)
        end do

        call cpu_time(t0)
        do it = 1, N_ITER
            st = dtl_context_barrier(ctx)
        end do
        call cpu_time(t1)
        call report('context_barrier', t1 - t0, N_ITER)
    end subroutine

    ! ------------------------------------------------------------------
    ! 6. Communicator barrier (dtl_barrier)
    ! ------------------------------------------------------------------
    subroutine bench_comm_barrier()
        integer(c_int) :: st

        ! Warmup
        do it = 1, WARMUP
            st = dtl_barrier(ctx)
        end do

        call cpu_time(t0)
        do it = 1, N_ITER
            st = dtl_barrier(ctx)
        end do
        call cpu_time(t1)
        call report('dtl_barrier', t1 - t0, N_ITER)
    end subroutine

    ! ------------------------------------------------------------------
    ! 7. Array create/destroy
    ! ------------------------------------------------------------------
    subroutine bench_array_lifecycle(asize)
        integer(c_int64_t), intent(in) :: asize
        type(c_ptr) :: arr
        integer(c_int) :: st

        ! Warmup
        do it = 1, WARMUP
            st = dtl_array_create(ctx, DTL_DTYPE_FLOAT64, asize, arr)
            call dtl_array_destroy(arr)
        end do

        call cpu_time(t0)
        do it = 1, N_ITER
            st = dtl_array_create(ctx, DTL_DTYPE_FLOAT64, asize, arr)
            call dtl_array_destroy(arr)
        end do
        call cpu_time(t1)
        call report('array create+destroy (n=10000)', t1 - t0, N_ITER)
    end subroutine

end program bench_fortran_api
