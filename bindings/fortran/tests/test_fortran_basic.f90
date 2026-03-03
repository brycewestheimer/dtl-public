! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file test_fortran_basic.f90
!> @brief Basic integration test for DTL Fortran bindings
!>
!> Tests:
!> 1. Context creation and destruction
!> 2. Rank and size queries
!> 3. Vector creation and destruction
!> 4. Local data access
!> 5. Size queries
!>
!> Exit code 0 on success, non-zero on failure.

program test_fortran_basic
    use dtl
    implicit none
    
    integer :: num_passed, num_failed
    logical :: test_result
    
    num_passed = 0
    num_failed = 0
    
    print *, '================================================'
    print *, ' DTL Fortran Basic Integration Tests'
    print *, '================================================'
    print *, ''
    
    ! Run all tests
    call test_context_lifecycle(test_result)
    call report_test('context_lifecycle', test_result, num_passed, num_failed)
    
    call test_context_queries(test_result)
    call report_test('context_queries', test_result, num_passed, num_failed)
    
    call test_vector_lifecycle(test_result)
    call report_test('vector_lifecycle', test_result, num_passed, num_failed)
    
    call test_vector_data_access(test_result)
    call report_test('vector_data_access', test_result, num_passed, num_failed)
    
    call test_vector_size_queries(test_result)
    call report_test('vector_size_queries', test_result, num_passed, num_failed)

    call test_span_from_vector(test_result)
    call report_test('span_from_vector', test_result, num_passed, num_failed)
    
    call test_error_message(test_result)
    call report_test('error_message', test_result, num_passed, num_failed)
    
    call test_dtype_utilities(test_result)
    call report_test('dtype_utilities', test_result, num_passed, num_failed)

    call test_array_lifecycle(test_result)
    call report_test('array_lifecycle', test_result, num_passed, num_failed)

    call test_error_invalid_context(test_result)
    call report_test('error_invalid_context', test_result, num_passed, num_failed)

    call test_broadcast_zero_count(test_result)
    call report_test('broadcast_zero_count', test_result, num_passed, num_failed)

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
    
    !> Test context creation and destruction
    subroutine test_context_lifecycle(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status
        
        passed = .false.
        
        ! Create context
        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return
        
        ! Check validity
        if (dtl_context_is_valid(ctx) /= 1) return
        
        ! Destroy context
        call dtl_context_destroy(ctx)
        
        passed = .true.
    end subroutine
    
    !> Test context rank and size queries
    subroutine test_context_queries(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status, rank, nprocs
        
        passed = .false.
        
        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return
        
        rank = dtl_context_rank(ctx)
        nprocs = dtl_context_size(ctx)
        
        ! Rank should be in [0, size-1]
        if (rank < 0) return
        if (rank >= nprocs) return
        if (nprocs < 1) return
        
        ! is_root should match rank == 0
        if (rank == 0) then
            if (dtl_context_is_root(ctx) /= 1) return
        else
            if (dtl_context_is_root(ctx) /= 0) return
        end if
        
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine
    
    !> Test vector creation and destruction
    subroutine test_vector_lifecycle(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size
        
        passed = .false.
        
        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return
        
        global_size = 1000_c_int64_t
        
        ! Create vector
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Check validity
        if (dtl_vector_is_valid(vec) /= 1) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Destroy vector
        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        
        passed = .true.
    end subroutine
    
    !> Test vector local data access
    subroutine test_vector_data_access(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, i
        real(c_double), pointer :: data(:)
        type(c_ptr) :: data_ptr
        
        passed = .false.
        
        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return
        
        global_size = 100_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if
        
        local_size = dtl_vector_local_size(vec)
        
        ! Get data pointer
        data_ptr = dtl_vector_local_data_mut(vec)
        if (.not. c_associated(data_ptr)) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Convert to Fortran pointer
        call c_f_pointer(data_ptr, data, [local_size])
        
        ! Write and read back
        do i = 1, local_size
            data(i) = real(i * 2, c_double)
        end do
        
        ! Verify
        do i = 1, local_size
            if (abs(data(i) - real(i * 2, c_double)) > 1.0d-10) then
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
    
    !> Test vector size queries
    subroutine test_vector_size_queries(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, reported_global, local_size, offset
        
        passed = .false.
        
        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return
        
        global_size = 500_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_INT32, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Check global size
        reported_global = dtl_vector_global_size(vec)
        if (reported_global /= global_size) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Check local size is positive (for single rank, equals global)
        local_size = dtl_vector_local_size(vec)
        if (local_size <= 0) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Check offset is non-negative
        offset = dtl_vector_local_offset(vec)
        if (offset < 0) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Check dtype
        if (dtl_vector_dtype(vec) /= DTL_DTYPE_INT32) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        
        ! Check empty
        if (dtl_vector_empty(vec) /= 0) then
            call dtl_vector_destroy(vec)
            call dtl_context_destroy(ctx)
            return
        end if
        
        call dtl_vector_destroy(vec)
        call dtl_context_destroy(ctx)
        
        passed = .true.
    end subroutine
    
    !> Test error message retrieval
    subroutine test_error_message(passed)
        logical, intent(out) :: passed
        character(len=:), allocatable :: msg
        
        passed = .false.
        
        ! Get message for success
        msg = dtl_get_error_message(DTL_SUCCESS)
        if (len(msg) == 0) return
        
        ! Get message for an error code
        msg = dtl_get_error_message(DTL_ERROR_MEMORY)
        if (len(msg) == 0) return
        
        passed = .true.
    end subroutine

    !> Test span creation from vector and local access
    subroutine test_span_from_vector(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, vec, span, sub
        type(c_ptr) :: vec_data_ptr, span_data_ptr
        real(c_double), pointer :: vec_data(:), span_data(:)
        integer(c_int) :: status
        integer(c_int64_t) :: global_size, local_size, sub_count

        passed = .false.
        ctx = c_null_ptr
        vec = c_null_ptr
        span = c_null_ptr
        sub = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        global_size = 64_c_int64_t
        status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            ctx = c_null_ptr
            return
        end if

        status = dtl_span_from_vector(vec, span)
        if (status /= DTL_SUCCESS) then
            call dtl_vector_destroy(vec)
            vec = c_null_ptr
            call dtl_context_destroy(ctx)
            ctx = c_null_ptr
            return
        end if

        if (dtl_span_is_valid(span) /= 1) goto 100
        if (dtl_span_dtype(span) /= DTL_DTYPE_FLOAT64) goto 100
        if (dtl_span_size(span) /= dtl_vector_global_size(vec)) goto 100

        local_size = dtl_span_local_size(span)
        if (local_size /= dtl_vector_local_size(vec)) goto 100

        if (local_size > 0_c_int64_t) then
            vec_data_ptr = dtl_vector_local_data_mut(vec)
            span_data_ptr = dtl_span_data_mut(span)
            if (.not. c_associated(vec_data_ptr)) goto 100
            if (.not. c_associated(span_data_ptr)) goto 100

            call c_f_pointer(vec_data_ptr, vec_data, [local_size])
            call c_f_pointer(span_data_ptr, span_data, [local_size])

            span_data(1) = 123.5_c_double
            if (abs(vec_data(1) - 123.5_c_double) > 1.0d-10) then
                nullify(span_data)
                nullify(vec_data)
                goto 100
            end if

            nullify(span_data)
            nullify(vec_data)
        end if

        sub_count = min(local_size, 2_c_int64_t)
        status = dtl_span_first(span, sub_count, sub)
        if (status /= DTL_SUCCESS) goto 100
        if (dtl_span_local_size(sub) /= sub_count) then
            call dtl_span_destroy(sub)
            sub = c_null_ptr
            goto 100
        end if
        call dtl_span_destroy(sub)
        sub = c_null_ptr

        call dtl_span_destroy(span)
        span = c_null_ptr
        call dtl_vector_destroy(vec)
        vec = c_null_ptr
        call dtl_context_destroy(ctx)
        ctx = c_null_ptr
        passed = .true.
        return

100     continue
        if (c_associated(sub)) then
            call dtl_span_destroy(sub)
            sub = c_null_ptr
        end if
        if (c_associated(span)) then
            call dtl_span_destroy(span)
            span = c_null_ptr
        end if
        if (c_associated(vec)) then
            call dtl_vector_destroy(vec)
            vec = c_null_ptr
        end if
        if (c_associated(ctx)) then
            call dtl_context_destroy(ctx)
            ctx = c_null_ptr
        end if
    end subroutine

    !> Test dtype utilities
    subroutine test_dtype_utilities(passed)
        logical, intent(out) :: passed
        integer(c_int64_t) :: size
        
        passed = .false.
        
        ! Check sizes
        size = dtl_dtype_size(DTL_DTYPE_FLOAT64)
        if (size /= 8) return
        
        size = dtl_dtype_size(DTL_DTYPE_FLOAT32)
        if (size /= 4) return
        
        size = dtl_dtype_size(DTL_DTYPE_INT32)
        if (size /= 4) return
        
        size = dtl_dtype_size(DTL_DTYPE_INT64)
        if (size /= 8) return
        
        passed = .true.
    end subroutine

    !> Test array lifecycle
    subroutine test_array_lifecycle(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx, arr
        integer(c_int) :: status
        integer(c_int64_t) :: global_sz, local_sz

        passed = .false.
        ctx = c_null_ptr
        arr = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        status = dtl_array_create(ctx, DTL_DTYPE_FLOAT64, 100_c_int64_t, arr)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        global_sz = dtl_array_global_size(arr)
        if (global_sz /= 100) then
            call dtl_array_destroy(arr)
            call dtl_context_destroy(ctx)
            return
        end if

        local_sz = dtl_array_local_size(arr)
        if (local_sz <= 0) then
            call dtl_array_destroy(arr)
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_array_destroy(arr)
        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

    !> Test error handling - invalid context
    subroutine test_error_invalid_context(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: vec
        integer(c_int) :: status

        passed = .false.
        vec = c_null_ptr

        ! Creating a vector with a null context should fail
        status = dtl_vector_create(c_null_ptr, DTL_DTYPE_FLOAT64, 10_c_int64_t, vec)
        if (status == DTL_SUCCESS) return  ! Should have failed

        ! If we got an error status, test passes
        passed = .true.
    end subroutine

    !> Test broadcast zero-count guard
    subroutine test_broadcast_zero_count(passed)
        logical, intent(out) :: passed
        type(c_ptr) :: ctx
        integer(c_int) :: status
        real(c_double), target :: data(1)

        passed = .false.
        ctx = c_null_ptr

        status = dtl_context_create_default(ctx)
        if (status /= DTL_SUCCESS) return

        ! Zero-count broadcast should succeed immediately
        status = dtl_broadcast_double(ctx, data, 0_c_int64_t, 0)
        if (status /= DTL_SUCCESS) then
            call dtl_context_destroy(ctx)
            return
        end if

        call dtl_context_destroy(ctx)
        passed = .true.
    end subroutine

end program test_fortran_basic
