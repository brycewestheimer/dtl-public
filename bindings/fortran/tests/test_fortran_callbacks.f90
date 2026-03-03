! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_callbacks.f90
!> @brief Integration tests for DTL Fortran callback-based algorithm bindings
!>
!> Tests:
!> 1. for_each with a doubling callback
!> 2. count_if with a predicate (> 50)
!> 3. Custom sort with descending comparator
!>
!> Exit code 0 on success, non-zero on failure.

!> Helper module defining bind(c) callback functions at module scope.
!> These cannot be internal procedures due to Fortran bind(c) rules.
module test_callbacks_mod
    use, intrinsic :: iso_c_binding
    implicit none
contains

    !> Mutable for-each callback: doubles each element in-place.
    !> Signature: void(void* element, uint64_t index, void* user_data)
    subroutine double_element(element, index, user_data) bind(c)
        type(c_ptr), value :: element
        integer(c_int64_t), value :: index
        type(c_ptr), value :: user_data
        real(c_double), pointer :: val

        call c_f_pointer(element, val)
        val = val * 2.0_c_double
    end subroutine

    !> Predicate callback: returns 1 if element > 50.0, else 0.
    !> Signature: int(const void* element, void* user_data)
    function greater_than_50(element, user_data) bind(c) result(res)
        type(c_ptr), value :: element
        type(c_ptr), value :: user_data
        integer(c_int) :: res
        real(c_double), pointer :: val

        call c_f_pointer(element, val)
        if (val > 50.0_c_double) then
            res = 1
        else
            res = 0
        end if
    end function

    !> Comparator callback: descending order for FLOAT64.
    !> Returns 1 if a > b (meaning a should come first in descending).
    !> Signature: int(const void* a, const void* b, void* user_data)
    function descending_compare(a, b, user_data) bind(c) result(res)
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: user_data
        integer(c_int) :: res
        real(c_double), pointer :: va, vb

        call c_f_pointer(a, va)
        call c_f_pointer(b, vb)
        if (va > vb) then
            res = 1
        else
            res = 0
        end if
    end function

end module test_callbacks_mod


program test_fortran_callbacks
    use dtl
    use test_callbacks_mod
    implicit none

    integer :: num_passed, num_failed
    logical :: test_result

    num_passed = 0
    num_failed = 0

    print *, '================================================'
    print *, ' DTL Fortran Callbacks Integration Tests'
    print *, '================================================'
    print *, ''

    ! Run all tests
    call test_for_each(test_result)
    call report_test('for_each', test_result, num_passed, num_failed)

    call test_count_if(test_result)
    call report_test('count_if', test_result, num_passed, num_failed)

    call test_custom_sort(test_result)
    call report_test('custom_sort', test_result, num_passed, num_failed)

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

    !> Test for_each: fill with 5.0, apply double_element callback, verify 10.0
    subroutine test_for_each(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, i
        real(c_double), target :: fill_val
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 30_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Fill with 5.0
        fill_val = 5.0_c_double
        status = dtl_fill_vector(vec, c_loc(fill_val))
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Apply for_each with the doubling callback
        status = dtl_for_each_vector(vec, c_funloc(double_element), c_null_ptr)
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify all elements are now 10.0
        local_size = dtl_vector_local_size(vec)
        data_ptr = dtl_vector_local_data(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 1, local_size
            if (abs(data(i) - 10.0_c_double) > 1.0d-10) then
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

    !> Test count_if: fill vector with 1..N, count elements > 50
    subroutine test_count_if(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, i, cnt
        integer(c_int64_t) :: expected_count
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr
        integer(c_int64_t) :: offset

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
        offset = dtl_vector_local_offset(vec)

        ! Fill with global index values: 1.0, 2.0, ..., 100.0
        data_ptr = dtl_vector_local_data_mut(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 1, local_size
            data(i) = real(offset + i, c_double)
        end do

        ! Count how many local values are > 50
        expected_count = 0
        do i = 1, local_size
            if (data(i) > 50.0_c_double) then
                expected_count = expected_count + 1
            end if
        end do
        nullify(data)

        ! Use count_if with predicate
        cnt = dtl_count_if_vector(vec, c_funloc(greater_than_50), c_null_ptr)

        if (cnt /= expected_count) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test custom sort: sort in descending order using comparator callback
    subroutine test_custom_sort(passed)
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

        global_size = 50_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        local_size = dtl_vector_local_size(vec)

        ! Fill with ascending values: 1.0, 2.0, 3.0, ...
        data_ptr = dtl_vector_local_data_mut(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 1, local_size
            data(i) = real(i, c_double)
        end do
        nullify(data)

        ! Sort with descending comparator
        status = dtl_sort_vector_func(vec, c_funloc(descending_compare), &
                                      c_null_ptr)
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify descending order
        data_ptr = dtl_vector_local_data(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        call c_f_pointer(data_ptr, data, [local_size])

        do i = 2, local_size
            if (data(i) > data(i - 1)) then
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

end program test_fortran_callbacks
