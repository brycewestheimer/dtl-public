! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_rma.f90
!> @brief DTL RMA Module - Remote memory access operations
!> @since 0.1.0

module dtl_rma
    use, intrinsic :: iso_c_binding
    implicit none

    private

    ! Lock mode constants
    public :: DTL_LOCK_EXCLUSIVE, DTL_LOCK_SHARED

    ! Window lifecycle
    public :: dtl_window_create, dtl_window_allocate, dtl_window_destroy
    public :: dtl_window_base, dtl_window_size, dtl_window_is_valid

    ! Active-target sync
    public :: dtl_window_fence

    ! Passive-target sync
    public :: dtl_window_lock, dtl_window_unlock
    public :: dtl_window_lock_all, dtl_window_unlock_all
    public :: dtl_window_flush, dtl_window_flush_all
    public :: dtl_window_flush_local, dtl_window_flush_local_all

    ! Data transfer
    public :: dtl_rma_put, dtl_rma_get
    public :: dtl_rma_put_async, dtl_rma_get_async

    ! Atomics
    public :: dtl_rma_accumulate, dtl_rma_fetch_and_op
    public :: dtl_rma_compare_and_swap

    ! ======================================================================
    ! Constants
    ! ======================================================================

    integer(c_int), parameter :: DTL_LOCK_EXCLUSIVE = 0
    integer(c_int), parameter :: DTL_LOCK_SHARED = 1

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        ! ------------------------------------------------------------------
        ! Window Lifecycle
        ! ------------------------------------------------------------------

        function dtl_window_create(ctx, base, size, win) &
                bind(c, name='dtl_window_create')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: base
            integer(c_int64_t), value :: size
            type(c_ptr), intent(out) :: win
            integer(c_int) :: dtl_window_create
        end function

        function dtl_window_allocate(ctx, size, win) &
                bind(c, name='dtl_window_allocate')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            integer(c_int64_t), value :: size
            type(c_ptr), intent(out) :: win
            integer(c_int) :: dtl_window_allocate
        end function

        subroutine dtl_window_destroy(win) &
                bind(c, name='dtl_window_destroy')
            import :: c_ptr
            type(c_ptr), value :: win
        end subroutine

        function dtl_window_base(win) bind(c, name='dtl_window_base')
            import :: c_ptr
            type(c_ptr), value :: win
            type(c_ptr) :: dtl_window_base
        end function

        function dtl_window_size(win) bind(c, name='dtl_window_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: win
            integer(c_int64_t) :: dtl_window_size
        end function

        function dtl_window_is_valid(win) &
                bind(c, name='dtl_window_is_valid')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int) :: dtl_window_is_valid
        end function

        ! ------------------------------------------------------------------
        ! Active-Target Synchronization
        ! ------------------------------------------------------------------

        function dtl_window_fence(win) bind(c, name='dtl_window_fence')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int) :: dtl_window_fence
        end function

        ! ------------------------------------------------------------------
        ! Passive-Target Synchronization
        ! ------------------------------------------------------------------

        function dtl_window_lock(win, target, mode) &
                bind(c, name='dtl_window_lock')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int), value :: mode
            integer(c_int) :: dtl_window_lock
        end function

        function dtl_window_unlock(win, target) &
                bind(c, name='dtl_window_unlock')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int) :: dtl_window_unlock
        end function

        function dtl_window_lock_all(win) &
                bind(c, name='dtl_window_lock_all')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int) :: dtl_window_lock_all
        end function

        function dtl_window_unlock_all(win) &
                bind(c, name='dtl_window_unlock_all')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int) :: dtl_window_unlock_all
        end function

        function dtl_window_flush(win, target) &
                bind(c, name='dtl_window_flush')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int) :: dtl_window_flush
        end function

        function dtl_window_flush_all(win) &
                bind(c, name='dtl_window_flush_all')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int) :: dtl_window_flush_all
        end function

        function dtl_window_flush_local(win, target) &
                bind(c, name='dtl_window_flush_local')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int) :: dtl_window_flush_local
        end function

        function dtl_window_flush_local_all(win) &
                bind(c, name='dtl_window_flush_local_all')
            import :: c_ptr, c_int
            type(c_ptr), value :: win
            integer(c_int) :: dtl_window_flush_local_all
        end function

        ! ------------------------------------------------------------------
        ! Data Transfer
        ! ------------------------------------------------------------------

        function dtl_rma_put(win, target, target_offset, origin, size) &
                bind(c, name='dtl_rma_put')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: origin
            integer(c_int64_t), value :: size
            integer(c_int) :: dtl_rma_put
        end function

        function dtl_rma_get(win, target, target_offset, buffer, size) &
                bind(c, name='dtl_rma_get')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: buffer
            integer(c_int64_t), value :: size
            integer(c_int) :: dtl_rma_get
        end function

        function dtl_rma_put_async(win, target, target_offset, origin, &
                                   size, req) &
                bind(c, name='dtl_rma_put_async')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: origin
            integer(c_int64_t), value :: size
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_rma_put_async
        end function

        function dtl_rma_get_async(win, target, target_offset, buffer, &
                                   size, req) &
                bind(c, name='dtl_rma_get_async')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: buffer
            integer(c_int64_t), value :: size
            type(c_ptr), intent(out) :: req
            integer(c_int) :: dtl_rma_get_async
        end function

        ! ------------------------------------------------------------------
        ! Atomics
        ! ------------------------------------------------------------------

        function dtl_rma_accumulate(win, target, target_offset, origin, &
                                    size, dtype, op) &
                bind(c, name='dtl_rma_accumulate')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: origin
            integer(c_int64_t), value :: size
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_rma_accumulate
        end function

        function dtl_rma_fetch_and_op(win, target, target_offset, origin, &
                                      result, dtype, op) &
                bind(c, name='dtl_rma_fetch_and_op')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: origin
            type(c_ptr), value :: result
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_rma_fetch_and_op
        end function

        function dtl_rma_compare_and_swap(win, target, target_offset, &
                                          compare, swap, result, dtype) &
                bind(c, name='dtl_rma_compare_and_swap')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: win
            integer(c_int), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: compare
            type(c_ptr), value :: swap
            type(c_ptr), value :: result
            integer(c_int), value :: dtype
            integer(c_int) :: dtl_rma_compare_and_swap
        end function

    end interface

end module dtl_rma
