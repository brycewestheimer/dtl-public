! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_futures.f90
!> @brief DTL Futures Module - Asynchronous completion tokens
!> @since 0.1.0
!> @warning Experimental — DTL progress engine has known stability issues

module dtl_futures
    use, intrinsic :: iso_c_binding
    implicit none

    private

    ! Future lifecycle
    public :: dtl_future_create, dtl_future_destroy

    ! Synchronization
    public :: dtl_future_wait, dtl_future_test

    ! Value access
    public :: dtl_future_get, dtl_future_set

    ! Combinators
    public :: dtl_when_all, dtl_when_any

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        !> @warning Experimental
        function dtl_future_create(fut) bind(c, name='dtl_future_create')
            import :: c_ptr, c_int
            type(c_ptr), intent(out) :: fut
            integer(c_int) :: dtl_future_create
        end function

        !> @warning Experimental
        subroutine dtl_future_destroy(fut) &
                bind(c, name='dtl_future_destroy')
            import :: c_ptr
            type(c_ptr), value :: fut
        end subroutine

        !> @warning Experimental
        function dtl_future_wait(fut) bind(c, name='dtl_future_wait')
            import :: c_ptr, c_int
            type(c_ptr), value :: fut
            integer(c_int) :: dtl_future_wait
        end function

        !> @warning Experimental
        function dtl_future_test(fut, completed) &
                bind(c, name='dtl_future_test')
            import :: c_ptr, c_int
            type(c_ptr), value :: fut
            integer(c_int), intent(out) :: completed
            integer(c_int) :: dtl_future_test
        end function

        !> @warning Experimental
        function dtl_future_get(fut, buffer, size) &
                bind(c, name='dtl_future_get')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: fut
            type(c_ptr), value :: buffer
            integer(c_int64_t), value :: size
            integer(c_int) :: dtl_future_get
        end function

        !> @warning Experimental
        function dtl_future_set(fut, value, size) &
                bind(c, name='dtl_future_set')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: fut
            type(c_ptr), value :: value
            integer(c_int64_t), value :: size
            integer(c_int) :: dtl_future_set
        end function

        !> @warning Experimental
        function dtl_when_all(futures, count, result) &
                bind(c, name='dtl_when_all')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), intent(in) :: futures(*)
            integer(c_int64_t), value :: count
            type(c_ptr), intent(out) :: result
            integer(c_int) :: dtl_when_all
        end function

        !> @warning Experimental
        function dtl_when_any(futures, count, result, completed_index) &
                bind(c, name='dtl_when_any')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), intent(in) :: futures(*)
            integer(c_int64_t), value :: count
            type(c_ptr), intent(out) :: result
            integer(c_int64_t), intent(out) :: completed_index
            integer(c_int) :: dtl_when_any
        end function

    end interface

end module dtl_futures
