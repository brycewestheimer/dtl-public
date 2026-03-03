! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_algorithms.f90
!> @brief DTL Algorithms Module - Built-in and callback-based algorithms
!> @since 0.1.0

module dtl_algorithms
    use, intrinsic :: iso_c_binding
    use dtl_core, only: dtl_f_unary_func, dtl_f_const_unary_func, &
                        dtl_f_predicate, dtl_f_comparator, &
                        dtl_f_transform_func, dtl_f_binary_func
    implicit none

    private

    ! Built-in sort
    public :: dtl_sort_vector, dtl_sort_vector_descending
    public :: dtl_sort_array, dtl_sort_array_descending

    ! Copy / Fill
    public :: dtl_copy_vector, dtl_copy_array
    public :: dtl_fill_vector, dtl_fill_array

    ! Find by value
    public :: dtl_find_vector, dtl_find_array

    ! Count by value
    public :: dtl_count_vector, dtl_count_array

    ! Local reduction (built-in op)
    public :: dtl_reduce_local_vector, dtl_reduce_local_array

    ! MinMax
    public :: dtl_minmax_vector, dtl_minmax_array

    ! Extrema element
    public :: dtl_min_element_vector, dtl_max_element_vector
    public :: dtl_min_element_array, dtl_max_element_array

    ! Scan (local)
    public :: dtl_inclusive_scan_vector, dtl_exclusive_scan_vector
    public :: dtl_inclusive_scan_array, dtl_exclusive_scan_array

    ! Callback: for-each
    public :: dtl_for_each_vector, dtl_for_each_vector_const
    public :: dtl_for_each_array, dtl_for_each_array_const

    ! Callback: transform
    public :: dtl_transform_vector, dtl_transform_array

    ! Callback: find-if
    public :: dtl_find_if_vector, dtl_find_if_array

    ! Callback: count-if
    public :: dtl_count_if_vector, dtl_count_if_array

    ! Callback: predicates
    public :: dtl_all_of_vector, dtl_any_of_vector, dtl_none_of_vector
    public :: dtl_all_of_array, dtl_any_of_array, dtl_none_of_array

    ! Callback: custom sort
    public :: dtl_sort_vector_func, dtl_sort_array_func

    ! Callback: custom reduce
    public :: dtl_reduce_local_vector_func, dtl_reduce_local_array_func

    ! Async algorithms (experimental)
    public :: dtl_async_for_each_vector, dtl_async_transform_vector
    public :: dtl_async_reduce_vector, dtl_async_sort_vector
    public :: dtl_async_for_each_array, dtl_async_reduce_array

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        ! ==================================================================
        ! Built-in Sort
        ! ==================================================================

        function dtl_sort_vector(vec) bind(c, name='dtl_sort_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_sort_vector
        end function

        function dtl_sort_vector_descending(vec) &
                bind(c, name='dtl_sort_vector_descending')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_sort_vector_descending
        end function

        function dtl_sort_array(arr) bind(c, name='dtl_sort_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_sort_array
        end function

        function dtl_sort_array_descending(arr) &
                bind(c, name='dtl_sort_array_descending')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_sort_array_descending
        end function

        ! ==================================================================
        ! Copy / Fill
        ! ==================================================================

        function dtl_copy_vector(src, dst) bind(c, name='dtl_copy_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: src
            type(c_ptr), value :: dst
            integer(c_int) :: dtl_copy_vector
        end function

        function dtl_copy_array(src, dst) bind(c, name='dtl_copy_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: src
            type(c_ptr), value :: dst
            integer(c_int) :: dtl_copy_array
        end function

        function dtl_fill_vector(vec, value) bind(c, name='dtl_fill_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            type(c_ptr), value :: value
            integer(c_int) :: dtl_fill_vector
        end function

        function dtl_fill_array(arr, value) bind(c, name='dtl_fill_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            type(c_ptr), value :: value
            integer(c_int) :: dtl_fill_array
        end function

        ! ==================================================================
        ! Find by value
        ! ==================================================================

        function dtl_find_vector(vec, value) bind(c, name='dtl_find_vector')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            type(c_ptr), value :: value
            integer(c_int64_t) :: dtl_find_vector
        end function

        function dtl_find_array(arr, value) bind(c, name='dtl_find_array')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: arr
            type(c_ptr), value :: value
            integer(c_int64_t) :: dtl_find_array
        end function

        ! ==================================================================
        ! Count by value
        ! ==================================================================

        function dtl_count_vector(vec, value) &
                bind(c, name='dtl_count_vector')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            type(c_ptr), value :: value
            integer(c_int64_t) :: dtl_count_vector
        end function

        function dtl_count_array(arr, value) bind(c, name='dtl_count_array')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: arr
            type(c_ptr), value :: value
            integer(c_int64_t) :: dtl_count_array
        end function

        ! ==================================================================
        ! Local Reduction (built-in op)
        ! ==================================================================

        function dtl_reduce_local_vector(vec, op, result) &
                bind(c, name='dtl_reduce_local_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int), value :: op
            type(c_ptr), value :: result
            integer(c_int) :: dtl_reduce_local_vector
        end function

        function dtl_reduce_local_array(arr, op, result) &
                bind(c, name='dtl_reduce_local_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int), value :: op
            type(c_ptr), value :: result
            integer(c_int) :: dtl_reduce_local_array
        end function

        ! ==================================================================
        ! MinMax
        ! ==================================================================

        function dtl_minmax_vector(vec, min_val, max_val) &
                bind(c, name='dtl_minmax_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            type(c_ptr), value :: min_val
            type(c_ptr), value :: max_val
            integer(c_int) :: dtl_minmax_vector
        end function

        function dtl_minmax_array(arr, min_val, max_val) &
                bind(c, name='dtl_minmax_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            type(c_ptr), value :: min_val
            type(c_ptr), value :: max_val
            integer(c_int) :: dtl_minmax_array
        end function

        ! ==================================================================
        ! Extrema Element
        ! ==================================================================

        function dtl_min_element_vector(vec) &
                bind(c, name='dtl_min_element_vector')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_min_element_vector
        end function

        function dtl_max_element_vector(vec) &
                bind(c, name='dtl_max_element_vector')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_max_element_vector
        end function

        function dtl_min_element_array(arr) &
                bind(c, name='dtl_min_element_array')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: arr
            integer(c_int64_t) :: dtl_min_element_array
        end function

        function dtl_max_element_array(arr) &
                bind(c, name='dtl_max_element_array')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: arr
            integer(c_int64_t) :: dtl_max_element_array
        end function

        ! ==================================================================
        ! Scan (local)
        ! ==================================================================

        function dtl_inclusive_scan_vector(vec, op) &
                bind(c, name='dtl_inclusive_scan_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int), value :: op
            integer(c_int) :: dtl_inclusive_scan_vector
        end function

        function dtl_exclusive_scan_vector(vec, op) &
                bind(c, name='dtl_exclusive_scan_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int), value :: op
            integer(c_int) :: dtl_exclusive_scan_vector
        end function

        function dtl_inclusive_scan_array(arr, op) &
                bind(c, name='dtl_inclusive_scan_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int), value :: op
            integer(c_int) :: dtl_inclusive_scan_array
        end function

        function dtl_exclusive_scan_array(arr, op) &
                bind(c, name='dtl_exclusive_scan_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int), value :: op
            integer(c_int) :: dtl_exclusive_scan_array
        end function

        ! ==================================================================
        ! Callback: for-each
        ! ==================================================================

        function dtl_for_each_vector(vec, func, user_data) &
                bind(c, name='dtl_for_each_vector')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_for_each_vector
        end function

        function dtl_for_each_vector_const(vec, func, user_data) &
                bind(c, name='dtl_for_each_vector_const')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_for_each_vector_const
        end function

        function dtl_for_each_array(arr, func, user_data) &
                bind(c, name='dtl_for_each_array')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_for_each_array
        end function

        function dtl_for_each_array_const(arr, func, user_data) &
                bind(c, name='dtl_for_each_array_const')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_for_each_array_const
        end function

        ! ==================================================================
        ! Callback: transform
        ! ==================================================================

        function dtl_transform_vector(src, dst, func, user_data) &
                bind(c, name='dtl_transform_vector')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: src
            type(c_ptr), value :: dst
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_transform_vector
        end function

        function dtl_transform_array(src, dst, func, user_data) &
                bind(c, name='dtl_transform_array')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: src
            type(c_ptr), value :: dst
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_transform_array
        end function

        ! ==================================================================
        ! Callback: find-if
        ! ==================================================================

        function dtl_find_if_vector(vec, pred, user_data) &
                bind(c, name='dtl_find_if_vector')
            import :: c_ptr, c_int64_t, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int64_t) :: dtl_find_if_vector
        end function

        function dtl_find_if_array(arr, pred, user_data) &
                bind(c, name='dtl_find_if_array')
            import :: c_ptr, c_int64_t, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int64_t) :: dtl_find_if_array
        end function

        ! ==================================================================
        ! Callback: count-if
        ! ==================================================================

        function dtl_count_if_vector(vec, pred, user_data) &
                bind(c, name='dtl_count_if_vector')
            import :: c_ptr, c_int64_t, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int64_t) :: dtl_count_if_vector
        end function

        function dtl_count_if_array(arr, pred, user_data) &
                bind(c, name='dtl_count_if_array')
            import :: c_ptr, c_int64_t, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int64_t) :: dtl_count_if_array
        end function

        ! ==================================================================
        ! Callback: predicates
        ! ==================================================================

        function dtl_all_of_vector(vec, pred, user_data) &
                bind(c, name='dtl_all_of_vector')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_all_of_vector
        end function

        function dtl_any_of_vector(vec, pred, user_data) &
                bind(c, name='dtl_any_of_vector')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_any_of_vector
        end function

        function dtl_none_of_vector(vec, pred, user_data) &
                bind(c, name='dtl_none_of_vector')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_none_of_vector
        end function

        function dtl_all_of_array(arr, pred, user_data) &
                bind(c, name='dtl_all_of_array')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_all_of_array
        end function

        function dtl_any_of_array(arr, pred, user_data) &
                bind(c, name='dtl_any_of_array')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_any_of_array
        end function

        function dtl_none_of_array(arr, pred, user_data) &
                bind(c, name='dtl_none_of_array')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: pred
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_none_of_array
        end function

        ! ==================================================================
        ! Callback: custom sort
        ! ==================================================================

        function dtl_sort_vector_func(vec, cmp, user_data) &
                bind(c, name='dtl_sort_vector_func')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: cmp
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_sort_vector_func
        end function

        function dtl_sort_array_func(arr, cmp, user_data) &
                bind(c, name='dtl_sort_array_func')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: cmp
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_sort_array_func
        end function

        ! ==================================================================
        ! Callback: custom reduce
        ! ==================================================================

        function dtl_reduce_local_vector_func(vec, func, identity, result, &
                                              user_data) &
                bind(c, name='dtl_reduce_local_vector_func')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: func
            type(c_ptr), value :: identity
            type(c_ptr), value :: result
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_reduce_local_vector_func
        end function

        function dtl_reduce_local_array_func(arr, func, identity, result, &
                                             user_data) &
                bind(c, name='dtl_reduce_local_array_func')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: func
            type(c_ptr), value :: identity
            type(c_ptr), value :: result
            type(c_ptr), value :: user_data
            integer(c_int) :: dtl_reduce_local_array_func
        end function

        ! ==================================================================
        ! Async Algorithms (Experimental)
        ! ==================================================================

        !> @warning Experimental — may hang due to progress engine issues
        function dtl_async_for_each_vector(vec, func, user_data, req) &
                bind(c, name='dtl_async_for_each_vector')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: vec
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_async_for_each_vector
        end function

        !> @warning Experimental — may hang due to progress engine issues
        function dtl_async_transform_vector(src, dst, func, user_data, req) &
                bind(c, name='dtl_async_transform_vector')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: src
            type(c_ptr), value :: dst
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_async_transform_vector
        end function

        !> @warning Experimental — may hang due to progress engine issues
        function dtl_async_reduce_vector(vec, op, result, req) &
                bind(c, name='dtl_async_reduce_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int), value :: op
            type(c_ptr), value :: result
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_async_reduce_vector
        end function

        !> @warning Experimental — may hang due to progress engine issues
        function dtl_async_sort_vector(vec, req) &
                bind(c, name='dtl_async_sort_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_async_sort_vector
        end function

        !> @warning Experimental — may hang due to progress engine issues
        function dtl_async_for_each_array(arr, func, user_data, req) &
                bind(c, name='dtl_async_for_each_array')
            import :: c_ptr, c_int, c_funptr
            type(c_ptr), value :: arr
            type(c_funptr), value :: func
            type(c_ptr), value :: user_data
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_async_for_each_array
        end function

        !> @warning Experimental — may hang due to progress engine issues
        function dtl_async_reduce_array(arr, op, result, req) &
                bind(c, name='dtl_async_reduce_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int), value :: op
            type(c_ptr), value :: result
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_async_reduce_array
        end function

    end interface

end module dtl_algorithms
