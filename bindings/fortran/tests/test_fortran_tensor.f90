! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_tensor.f90
!> @brief Integration tests for DTL Fortran tensor bindings
!>
!> Tests:
!> 1. Tensor lifecycle (create, query, destroy)
!> 2. Tensor data access via fill and local_data pointer
!> 3. Tensor element access via set_local/get_local
!>
!> Exit code 0 on success, non-zero on failure.

program test_fortran_tensor
    use dtl
    implicit none

    integer :: num_passed, num_failed
    logical :: test_result

    num_passed = 0
    num_failed = 0

    print *, '================================================'
    print *, ' DTL Fortran Tensor Integration Tests'
    print *, '================================================'
    print *, ''

    ! Run all tests
    call test_tensor_lifecycle(test_result)
    call report_test('tensor_lifecycle', test_result, num_passed, num_failed)

    call test_tensor_data_access(test_result)
    call report_test('tensor_data_access', test_result, num_passed, num_failed)

    call test_tensor_element_access(test_result)
    call report_test('tensor_element_access', test_result, num_passed, num_failed)

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

    !> Test tensor lifecycle: create 2D tensor, query shape/ndim/dims/global_size, destroy
    subroutine test_tensor_lifecycle(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, tensor
        integer(c_int) :: status
        type(dtl_shape) :: shp
        integer(c_int64_t) :: gsize

        passed = .false.
        ctx = c_null_ptr
        tensor = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Create a 2D tensor of shape (10, 20)
        shp = dtl_shape_2d(10_c_int64_t, 20_c_int64_t)
        status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shp, tensor)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify validity
        if (dtl_tensor_is_valid(tensor) /= 1) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify ndim = 2
        if (dtl_tensor_ndim(tensor) /= 2) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify dimensions
        if (dtl_tensor_dim(tensor, 0) /= 10_c_int64_t) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if
        if (dtl_tensor_dim(tensor, 1) /= 20_c_int64_t) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify global size = 10 * 20 = 200
        gsize = dtl_tensor_global_size(tensor)
        if (gsize /= 200_c_int64_t) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Verify dtype
        if (dtl_tensor_dtype(tensor) /= DTL_DTYPE_FLOAT64) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_tensor_destroy(tensor)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test tensor data access: create 1D tensor, fill, read back via local_data
    subroutine test_tensor_data_access(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, tensor
        integer(c_int) :: status
        type(dtl_shape) :: shp
        integer(c_int64_t) :: local_size, i
        real(c_double), target :: fill_val
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr

        passed = .false.
        ctx = c_null_ptr
        tensor = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Create a 1D tensor of size 50
        shp = dtl_shape_1d(50_c_int64_t)
        status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shp, tensor)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        ! Fill with a known value
        fill_val = 3.14_c_double
        status = dtl_tensor_fill_local(tensor, c_loc(fill_val))
        if (status /= DTL_SUCCESS) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        ! Read back via local_data pointer
        local_size = dtl_tensor_local_size(tensor)
        if (local_size <= 0) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        data_ptr = dtl_tensor_local_data(tensor)
        if (.not. c_associated(data_ptr)) then
            call dtl_tensor_destroy(tensor)
            call dtl_context_destroy(ctx)
            return
        end if

        call c_f_pointer(data_ptr, data, [local_size])

        ! Verify all elements are 3.14
        do i = 1, local_size
            if (abs(data(i) - 3.14_c_double) > 1.0d-10) then
                nullify(data)
                call dtl_tensor_destroy(tensor)
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        nullify(data)
        call dtl_tensor_destroy(tensor)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test tensor element access via set_local/get_local (linear indexing)
    subroutine test_tensor_element_access(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, tensor
        integer(c_int) :: status
        type(dtl_shape) :: shp
        integer(c_int64_t) :: local_size, i
        real(c_double), target :: set_val, get_val

        passed = .false.
        ctx = c_null_ptr
        tensor = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Create a small 1D tensor
        shp = dtl_shape_1d(8_c_int64_t)
        status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shp, tensor)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        local_size = dtl_tensor_local_size(tensor)

        ! Set each local element to its index * 10
        do i = 1, local_size
            set_val = real(i * 10, c_double)
            ! C API uses 0-based indexing
            status = dtl_tensor_set_local(tensor, i - 1_c_int64_t, c_loc(set_val))
            if (status /= DTL_SUCCESS) then
                call dtl_tensor_destroy(tensor)
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        ! Read back and verify
        do i = 1, local_size
            get_val = 0.0_c_double
            status = dtl_tensor_get_local(tensor, i - 1_c_int64_t, c_loc(get_val))
            if (status /= DTL_SUCCESS) then
                call dtl_tensor_destroy(tensor)
                call dtl_context_destroy(ctx)
                return
            end if
            if (abs(get_val - real(i * 10, c_double)) > 1.0d-10) then
                call dtl_tensor_destroy(tensor)
                call dtl_context_destroy(ctx)
                return
            end if
        end do

        call dtl_tensor_destroy(tensor)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

end program test_fortran_tensor
