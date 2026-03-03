! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_mpi.f90
!> @brief DTL Fortran MPI integration test
!>
!> Self-contained test program exercising the DTL Fortran bindings under MPI.
!> Uses simple pass/fail counting — no external test framework required.
!>
!> Tests:
!>   1. Context creation and validity
!>   2. Context rank/size queries and consistency
!>   3. Context is_root correctness
!>   4. Context device queries
!>   5. Context barrier synchronization
!>   6. Vector creation and validity
!>   7. Vector size queries (global, local, offset)
!>   8. Vector data access (write/read via c_f_pointer)
!>   9. Vector fill_local
!>  10. Vector dtype and empty queries
!>  11. Vector barrier
!>  12. Vector create_fill
!>  13. Communicator barrier (dtl_barrier)
!>  14. Dtype utility functions
!>  15. Error message retrieval
!>  16. Array creation and queries
!>  17. Allreduce sum across ranks
!>  18. Broadcast from root
!>
!> Run:
!>   ./test_fortran_mpi                     (single rank)
!>   mpirun -np 2 ./test_fortran_mpi        (multi-rank)
!>   mpirun -np 4 ./test_fortran_mpi        (multi-rank)

program test_fortran_mpi
    use dtl
    implicit none

    integer :: tests_passed, tests_failed, tests_total
    logical :: result
    type(c_ptr) :: ctx
    integer(c_int) :: status, my_rank, my_size

    tests_passed = 0
    tests_failed = 0
    tests_total  = 0

    ! ---- Create a shared context for all tests ----
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) then
        print '(A)', 'FATAL: Failed to create DTL context — aborting'
        stop 1
    end if
    my_rank = dtl_context_rank(ctx)
    my_size = dtl_context_size(ctx)

    if (my_rank == 0) then
        print '(A)',    '================================================'
        print '(A)',    ' DTL Fortran MPI Integration Tests'
        print '(A,I0)', '   Ranks: ', my_size
        print '(A)',    '================================================'
        print *, ''
    end if

    ! Synchronize before starting tests
    status = dtl_context_barrier(ctx)

    ! ---- Run tests ----
    call test_context_creation(result);       call tally('context_creation',       result)
    call test_context_queries(result);        call tally('context_queries',        result)
    call test_context_is_root(result);        call tally('context_is_root',        result)
    call test_context_device_queries(result); call tally('context_device_queries', result)
    call test_context_barrier_sync(result);   call tally('context_barrier_sync',   result)
    call test_vector_create(result);          call tally('vector_create',          result)
    call test_vector_size_queries(result);    call tally('vector_size_queries',    result)
    call test_vector_data_access(result);     call tally('vector_data_access',     result)
    call test_vector_fill_local(result);      call tally('vector_fill_local',      result)
    call test_vector_dtype_empty(result);     call tally('vector_dtype_empty',     result)
    call test_vector_barrier(result);         call tally('vector_barrier',         result)
    call test_vector_create_fill(result);     call tally('vector_create_fill',     result)
    call test_comm_barrier(result);           call tally('comm_barrier',           result)
    call test_dtype_utilities(result);        call tally('dtype_utilities',        result)
    call test_error_messages(result);         call tally('error_messages',         result)
    call test_array_create(result);           call tally('array_create',           result)
    call test_allreduce_sum(result);          call tally('allreduce_sum',          result)
    call test_broadcast(result);              call tally('broadcast',              result)

    ! ---- Summary ----
    status = dtl_context_barrier(ctx)

    if (my_rank == 0) then
        print *, ''
        print '(A)',          '================================================'
        print '(A,I0,A,I0,A,I0)', '  Total: ', tests_total, &
              '  Passed: ', tests_passed, '  Failed: ', tests_failed
        print '(A)',          '================================================'
        if (tests_failed > 0) then
            print '(A)', '  SOME TESTS FAILED'
        else
            print '(A)', '  ALL TESTS PASSED'
        end if
    end if

    call dtl_context_destroy(ctx)

    if (tests_failed > 0) stop 1

contains

    ! ------------------------------------------------------------------
    ! Helper: record result
    ! ------------------------------------------------------------------
    subroutine tally(name, passed)
        character(len=*), intent(in) :: name
        logical, intent(in) :: passed

        tests_total = tests_total + 1
        if (passed) then
            tests_passed = tests_passed + 1
            if (my_rank == 0) print '(A,A)', '  [PASS] ', name
        else
            tests_failed = tests_failed + 1
            if (my_rank == 0) print '(A,A)', '  [FAIL] ', name
        end if
    end subroutine

    ! ==================================================================
    ! Test 1: Context creation and validity
    ! ==================================================================
    subroutine test_context_creation(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: local_ctx
        integer(c_int) :: st

        passed = .false.
        st = dtl_context_create_default(local_ctx)
        if (st /= DTL_SUCCESS) return
        if (dtl_context_is_valid(local_ctx) /= 1) then
            call dtl_context_destroy(local_ctx)
            return
        end if
        call dtl_context_destroy(local_ctx)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 2: Rank/size queries — rank in [0,size), size >= 1
    ! ==================================================================
    subroutine test_context_queries(passed)
        logical, intent(out) :: passed
        integer(c_int) :: r, s

        passed = .false.
        r = dtl_context_rank(ctx)
        s = dtl_context_size(ctx)
        if (r < 0)  return
        if (r >= s)  return
        if (s < 1)  return
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 3: is_root matches rank == 0
    ! ==================================================================
    subroutine test_context_is_root(passed)
        logical, intent(out) :: passed
        integer(c_int) :: r, root_flag

        passed = .false.
        r = dtl_context_rank(ctx)
        root_flag = dtl_context_is_root(ctx)
        if (r == 0) then
            if (root_flag /= 1) return
        else
            if (root_flag /= 0) return
        end if
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 4: Device queries (should not crash)
    ! ==================================================================
    subroutine test_context_device_queries(passed)
        logical, intent(out) :: passed
        integer(c_int) :: has_dev, dev_id

        passed = .false.
        has_dev = dtl_context_has_device(ctx)
        dev_id  = dtl_context_device_id(ctx)
        ! has_dev is 0 or 1
        if (has_dev /= 0 .and. has_dev /= 1) return
        ! If no device, device_id should be -1
        if (has_dev == 0 .and. dev_id /= -1) return
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 5: Context barrier returns success
    ! ==================================================================
    subroutine test_context_barrier_sync(passed)
        logical, intent(out) :: passed
        integer(c_int) :: st

        passed = .false.
        st = dtl_context_barrier(ctx)
        if (st /= DTL_SUCCESS) return
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 6: Vector creation/destruction
    ! ==================================================================
    subroutine test_vector_create(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec
        integer(c_int) :: st
        integer(c_int64_t) :: gsize

        passed = .false.
        gsize = 1000_c_int64_t

        st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, gsize, vec)
        if (st /= DTL_SUCCESS) return
        if (dtl_vector_is_valid(vec) /= 1) then
            call dtl_vector_destroy(vec)
            return
        end if
        call dtl_vector_destroy(vec)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 7: Vector size queries — global, local, offset
    ! ==================================================================
    subroutine test_vector_size_queries(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec
        integer(c_int) :: st
        integer(c_int64_t) :: gsize, lsize, loff

        passed = .false.
        gsize = 500_c_int64_t

        st = dtl_vector_create(ctx, DTL_DTYPE_INT32, gsize, vec)
        if (st /= DTL_SUCCESS) return

        ! Global size must match
        if (dtl_vector_global_size(vec) /= gsize) then
            call dtl_vector_destroy(vec)
            return
        end if

        ! Local size > 0
        lsize = dtl_vector_local_size(vec)
        if (lsize <= 0) then
            call dtl_vector_destroy(vec)
            return
        end if

        ! Offset >= 0
        loff = dtl_vector_local_offset(vec)
        if (loff < 0) then
            call dtl_vector_destroy(vec)
            return
        end if

        ! Offset + local_size <= global_size
        if (loff + lsize > gsize) then
            call dtl_vector_destroy(vec)
            return
        end if

        call dtl_vector_destroy(vec)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 8: Vector data access — write then read back via c_f_pointer
    ! ==================================================================
    subroutine test_vector_data_access(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec, dptr
        integer(c_int) :: st
        integer(c_int64_t) :: gsize, lsize, i
        real(c_double), pointer :: data(:)

        passed = .false.
        gsize = 200_c_int64_t

        st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, gsize, vec)
        if (st /= DTL_SUCCESS) return
        lsize = dtl_vector_local_size(vec)

        dptr = dtl_vector_local_data_mut(vec)
        if (.not. c_associated(dptr)) then
            call dtl_vector_destroy(vec)
            return
        end if
        call c_f_pointer(dptr, data, [lsize])

        ! Write pattern
        do i = 1, lsize
            data(i) = real(i * 3, c_double)
        end do

        ! Read back and verify
        do i = 1, lsize
            if (abs(data(i) - real(i * 3, c_double)) > 1.0d-12) then
                nullify(data)
                call dtl_vector_destroy(vec)
                return
            end if
        end do

        ! Also verify read-only pointer works
        dptr = dtl_vector_local_data(vec)
        if (.not. c_associated(dptr)) then
            nullify(data)
            call dtl_vector_destroy(vec)
            return
        end if

        nullify(data)
        call dtl_vector_destroy(vec)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 9: Vector fill_local
    ! ==================================================================
    subroutine test_vector_fill_local(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec, dptr
        integer(c_int) :: st
        integer(c_int64_t) :: gsize, lsize, i
        real(c_double), pointer :: data(:)
        real(c_double), target :: fill_val

        passed = .false.
        gsize = 100_c_int64_t

        st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, gsize, vec)
        if (st /= DTL_SUCCESS) return

        fill_val = 7.5_c_double
        st = dtl_vector_fill_local(vec, c_loc(fill_val))
        if (st /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            return
        end if

        lsize = dtl_vector_local_size(vec)
        dptr = dtl_vector_local_data(vec)
        call c_f_pointer(dptr, data, [lsize])

        do i = 1, lsize
            if (abs(data(i) - 7.5_c_double) > 1.0d-12) then
                nullify(data)
                call dtl_vector_destroy(vec)
                return
            end if
        end do

        nullify(data)
        call dtl_vector_destroy(vec)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 10: Vector dtype and empty queries
    ! ==================================================================
    subroutine test_vector_dtype_empty(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec
        integer(c_int) :: st
        integer(c_int64_t) :: gsize

        passed = .false.
        gsize = 64_c_int64_t

        st = dtl_vector_create(ctx, DTL_DTYPE_INT64, gsize, vec)
        if (st /= DTL_SUCCESS) return

        if (dtl_vector_dtype(vec) /= DTL_DTYPE_INT64) then
            call dtl_vector_destroy(vec)
            return
        end if

        ! Non-empty vector
        if (dtl_vector_empty(vec) /= 0) then
            call dtl_vector_destroy(vec)
            return
        end if

        call dtl_vector_destroy(vec)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 11: Vector barrier returns success
    ! ==================================================================
    subroutine test_vector_barrier(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec
        integer(c_int) :: st
        integer(c_int64_t) :: gsize

        passed = .false.
        gsize = 128_c_int64_t

        st = dtl_vector_create(ctx, DTL_DTYPE_FLOAT32, gsize, vec)
        if (st /= DTL_SUCCESS) return

        st = dtl_vector_barrier(vec)
        if (st /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            return
        end if

        call dtl_vector_destroy(vec)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 12: Vector create_fill
    ! ==================================================================
    subroutine test_vector_create_fill(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec, dptr
        integer(c_int) :: st
        integer(c_int64_t) :: gsize, lsize, i
        real(c_double), target :: init_val
        real(c_double), pointer :: data(:)

        passed = .false.
        gsize = 80_c_int64_t
        init_val = 3.14_c_double

        st = dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, gsize, &
                                    c_loc(init_val), vec)
        if (st /= DTL_SUCCESS) return

        lsize = dtl_vector_local_size(vec)
        dptr = dtl_vector_local_data(vec)
        if (.not. c_associated(dptr)) then
            call dtl_vector_destroy(vec)
            return
        end if
        call c_f_pointer(dptr, data, [lsize])

        do i = 1, lsize
            if (abs(data(i) - 3.14_c_double) > 1.0d-12) then
                nullify(data)
                call dtl_vector_destroy(vec)
                return
            end if
        end do

        nullify(data)
        call dtl_vector_destroy(vec)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 13: Communicator-level barrier (dtl_barrier)
    ! ==================================================================
    subroutine test_comm_barrier(passed)
        logical, intent(out) :: passed
        integer(c_int) :: st

        passed = .false.
        st = dtl_barrier(ctx)
        if (st /= DTL_SUCCESS) return
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 14: Dtype utility functions
    ! ==================================================================
    subroutine test_dtype_utilities(passed)
        logical, intent(out) :: passed

        passed = .false.

        if (dtl_dtype_size(DTL_DTYPE_FLOAT64) /= 8) return
        if (dtl_dtype_size(DTL_DTYPE_FLOAT32) /= 4) return
        if (dtl_dtype_size(DTL_DTYPE_INT32)   /= 4) return
        if (dtl_dtype_size(DTL_DTYPE_INT64)   /= 8) return
        if (dtl_dtype_size(DTL_DTYPE_INT8)    /= 1) return
        if (dtl_dtype_size(DTL_DTYPE_INT16)   /= 2) return

        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 15: Error message retrieval
    ! ==================================================================
    subroutine test_error_messages(passed)
        logical, intent(out) :: passed
        character(len=:), allocatable :: msg

        passed = .false.

        msg = dtl_get_error_message(DTL_SUCCESS)
        if (len(msg) == 0) return

        msg = dtl_get_error_message(DTL_ERROR_MEMORY)
        if (len(msg) == 0) return

        msg = dtl_get_error_message(DTL_ERROR_COMMUNICATION)
        if (len(msg) == 0) return

        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 16: Array creation and queries
    ! ==================================================================
    subroutine test_array_create(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: arr
        integer(c_int) :: st
        integer(c_int64_t) :: asize, lsize, loff

        passed = .false.
        asize = 256_c_int64_t

        st = dtl_array_create(ctx, DTL_DTYPE_FLOAT64, asize, arr)
        if (st /= DTL_SUCCESS) return

        if (dtl_array_is_valid(arr) /= 1) then
            call dtl_array_destroy(arr)
            return
        end if

        if (dtl_array_global_size(arr) /= asize) then
            call dtl_array_destroy(arr)
            return
        end if

        lsize = dtl_array_local_size(arr)
        if (lsize <= 0) then
            call dtl_array_destroy(arr)
            return
        end if

        loff = dtl_array_local_offset(arr)
        if (loff < 0) then
            call dtl_array_destroy(arr)
            return
        end if

        if (dtl_array_dtype(arr) /= DTL_DTYPE_FLOAT64) then
            call dtl_array_destroy(arr)
            return
        end if

        if (dtl_array_empty(arr) /= 0) then
            call dtl_array_destroy(arr)
            return
        end if

        call dtl_array_destroy(arr)
        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 17: Allreduce sum across ranks
    ! ==================================================================
    subroutine test_allreduce_sum(passed)
        logical, intent(out) :: passed
        real(c_double), target :: send_val, recv_val
        integer(c_int) :: st
        real(c_double) :: expected

        passed = .false.

        ! Each rank contributes its rank+1
        send_val = real(my_rank + 1, c_double)
        recv_val = 0.0_c_double

        st = dtl_allreduce(ctx, c_loc(send_val), c_loc(recv_val), &
                           1_c_int64_t, DTL_DTYPE_FLOAT64, DTL_OP_SUM)
        if (st /= DTL_SUCCESS) return

        ! Expected = sum(1..size) = size*(size+1)/2
        expected = real(my_size, c_double) * real(my_size + 1, c_double) / 2.0_c_double
        if (abs(recv_val - expected) > 1.0d-10) return

        passed = .true.
    end subroutine

    ! ==================================================================
    ! Test 18: Broadcast from root
    ! ==================================================================
    subroutine test_broadcast(passed)
        logical, intent(out) :: passed
        integer(c_int), target :: bval
        integer(c_int) :: st

        passed = .false.

        if (my_rank == 0) then
            bval = 12345
        else
            bval = 0
        end if

        st = dtl_broadcast(ctx, c_loc(bval), 1_c_int64_t, DTL_DTYPE_INT32, 0)
        if (st /= DTL_SUCCESS) return

        if (bval /= 12345) return

        passed = .true.
    end subroutine

end program test_fortran_mpi
