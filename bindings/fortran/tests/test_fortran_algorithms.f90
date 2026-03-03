! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_algorithms.f90
!> @brief Integration tests for DTL Fortran built-in algorithm bindings
!>
!> Tests:
!> 1. Sort vector ascending
!> 2. Copy and fill
!> 3. Find and count by value
!> 4. Local reduction with built-in op
!> 5. MinMax query
!> 6. Inclusive scan on vector
!>
!> Exit code 0 on success, non-zero on failure.

program test_fortran_algorithms
    use dtl
    implicit none

    integer :: num_passed, num_failed
    logical :: test_result

    num_passed = 0
    num_failed = 0

    print *, '================================================'
    print *, ' DTL Fortran Algorithms Integration Tests'
    print *, '================================================'
    print *, ''

    ! Run all tests
    call test_sort_vector(test_result)
    call report_test('sort_vector', test_result, num_passed, num_failed)

    call test_copy_fill(test_result)
    call report_test('copy_fill', test_result, num_passed, num_failed)

    call test_find_count(test_result)
    call report_test('find_count', test_result, num_passed, num_failed)

    call test_reduce_local(test_result)
    call report_test('reduce_local', test_result, num_passed, num_failed)

    call test_minmax(test_result)
    call report_test('minmax', test_result, num_passed, num_failed)

    call test_scan_vector(test_result)
    call report_test('scan_vector', test_result, num_passed, num_failed)

    ! Summary
    print *, ''
    print *, '================================================'
    print '(A,I0,A,I0)', '  Passed: ', num_passed, '  Failed: ', num_failed
    print *, '================================================'

    if (num_failed > 0) then
        print *, 'TESTS FAILED'
        stop 1
    else
        print *, 'ALL TESTS PASSED'
    end if

contains

    subroutine report_test(name, passed, num_passed, num_failed)
        character(len=*), intent(in) :: name
        logical, intent(in) :: passed
        integer, intent(inout) :: num_passed, num_failed

        if (passed) then
            print '(A,A,A)', '  [PASS] ', name, ''
            num_passed = num_passed + 1
        else
            print '(A,A,A)', '  [FAIL] ', name, ''
            num_failed = num_failed + 1
        end if
    end subroutine

    !> Test sort: create vector with descending values, sort ascending, verify
    subroutine test_sort_vector(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, i
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 100_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        local_size = dtl_vector_local_size(vec)

        ! Fill with descending values
        data_ptr = dtl_vector_local_data_mut(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 1, local_size
            data(i) = real(local_size - i + 1, c_double)
        end do
        nullify(data)

        ! Sort ascending
        status = dtl_sort_vector(vec)
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify sorted order
        data_ptr = dtl_vector_local_data(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 2, local_size
            if (data(i) < data(i - 1)) then
                nullify(data)
                call dtl_vector_destroy(vec)
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        nullify(data)
        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test copy and fill: create two vectors, fill src, copy to dst, verify match
    subroutine test_copy_fill(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, src, dst
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, i
        real(c_double), target :: fill_val
        real(c_double), pointer :: src_data(:), dst_data(:)
        type(c_ptr) :: src_ptr, dst_ptr

        passed = .false.
        ctx = c_null_ptr
        src = c_null_ptr
        dst = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 50_c_int64_t

        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, src)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, dst)
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(src)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Fill src with 7.77
        fill_val = 7.77_c_double
        status = dtl_fill_vector(src, c_loc(fill_val))
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(dst)
            call dtl_vector_destroy(src)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Copy src to dst
        status = dtl_copy_vector(src, dst)
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(dst)
            call dtl_vector_destroy(src)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify dst matches src
        local_size = dtl_vector_local_size(dst)
        src_ptr = dtl_vector_local_data(src)
        dst_ptr = dtl_vector_local_data(dst)
        if (.not. c_associated(src_ptr) .or. .not. c_associated(dst_ptr)) then
            call dtl_vector_destroy(dst)
            call dtl_vector_destroy(src)
            call dtl_context_destroy(ctx)
            return
        end if

        call c_f_pointer(src_ptr, src_data, [local_size])
        call c_f_pointer(dst_ptr, dst_data, [local_size])

        do i = 1, local_size
            if (abs(src_data(i) - dst_data(i)) > 1.0d-10) then
                nullify(src_data)
                nullify(dst_data)
                call dtl_vector_destroy(dst)
                call dtl_vector_destroy(src)
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        nullify(src_data)
        nullify(dst_data)
        call dtl_vector_destroy(dst)
        call dtl_vector_destroy(src)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test find and count: create vector with known values, find and count a target
    subroutine test_find_count(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, i, found_idx, cnt
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr
        real(c_double), target :: search_val

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 20_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        local_size = dtl_vector_local_size(vec)

        ! Fill with values: 1.0, 2.0, 1.0, 2.0, ...
        data_ptr = dtl_vector_local_data_mut(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 1, local_size
            if (mod(i, 2_c_int64_t) == 1) then
                data(i) = 1.0_c_double
            else
                data(i) = 2.0_c_double
            end if
        end do
        nullify(data)

        ! Find value 2.0 -- should return a valid index (not -1/NPOS)
        search_val = 2.0_c_double
        found_idx = dtl_find_vector(vec, c_loc(search_val))
        ! For a vector with alternating 1.0 and 2.0, first 2.0 is at index 1 (0-based)
        if (local_size >= 2 .and. found_idx < 0) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Count occurrences of 1.0
        search_val = 1.0_c_double
        cnt = dtl_count_vector(vec, c_loc(search_val))
        ! Half the elements (rounded up) should be 1.0
        if (cnt /= (local_size + 1) / 2) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test local reduction: fill with 1.0, reduce sum, verify result = local_size
    subroutine test_reduce_local(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size
        real(c_double), target :: fill_val, result_val

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 100_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Fill with 1.0
        fill_val = 1.0_c_double
        status = dtl_fill_vector(vec, c_loc(fill_val))
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Reduce with SUM
        result_val = 0.0_c_double
        status = dtl_reduce_local_vector(vec, DTL_OP_SUM, c_loc(result_val))
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify result equals local_size
        local_size = dtl_vector_local_size(vec)
        if (abs(result_val - real(local_size, c_double)) > 1.0d-10) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test minmax: create vector with known range, verify min and max
    subroutine test_minmax(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, i
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr
        real(c_double), target :: min_val, max_val

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 50_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        local_size = dtl_vector_local_size(vec)

        ! Fill with values 10.0, 20.0, 30.0, ... so min=10.0, max=local_size*10.0
        data_ptr = dtl_vector_local_data_mut(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 1, local_size
            data(i) = real(i * 10, c_double)
        end do
        nullify(data)

        ! Query minmax
        min_val = 0.0_c_double
        max_val = 0.0_c_double
        status = dtl_minmax_vector(vec, c_loc(min_val), c_loc(max_val))
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify min = 10.0, max = local_size * 10
        if (abs(min_val - 10.0_c_double) > 1.0d-10) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        if (abs(max_val - real(local_size * 10, c_double)) > 1.0d-10) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test inclusive scan: fill with 1.0, scan sum, verify last element = local_size
    subroutine test_scan_vector(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size
        real(c_double), target :: fill_val
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 100_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Fill with 1.0
        fill_val = 1.0_c_double
        status = dtl_fill_vector(vec, c_loc(fill_val))
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Inclusive scan with SUM
        status = dtl_inclusive_scan_vector(vec, DTL_OP_SUM)
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify: last local element should equal local_size
        local_size = dtl_vector_local_size(vec)
        data_ptr = dtl_vector_local_data(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        if (abs(data(local_size) - real(local_size, c_double)) > 1.0d-10) then
            nullify(data)
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Also verify first element = 1.0
        if (abs(data(1) - 1.0_c_double) > 1.0d-10) then
            nullify(data)
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        nullify(data)
        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

end program test_fortran_algorithms
