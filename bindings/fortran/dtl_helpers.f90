! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_helpers.f90
!> @brief DTL Helpers Module - Fortran convenience wrappers and utilities
!> @since 0.1.0
!>
!> Provides Fortran-idiomatic wrappers that simplify common operations:
!> - String conversion (C to Fortran)
!> - Typed broadcast/allreduce wrappers that avoid void* casting
!> - Shape construction helpers
!> - Pre-built convenience callbacks for common patterns

module dtl_helpers
    use, intrinsic :: iso_c_binding
    use dtl_core
    use dtl_containers, only: dtl_vector_local_size
    implicit none

    private

    ! String helpers
    public :: dtl_c_to_f_string, dtl_get_error_message

    ! Typed broadcast wrappers
    public :: dtl_broadcast_double, dtl_broadcast_int32
    public :: dtl_broadcast_float, dtl_broadcast_int64

    ! Typed allreduce wrappers
    public :: dtl_allreduce_sum_double, dtl_allreduce_sum_int32
    public :: dtl_allreduce_sum_int64, dtl_allreduce_sum_float
    public :: dtl_allreduce_max_double, dtl_allreduce_min_double

    ! Typed send/recv wrappers
    public :: dtl_send_double, dtl_recv_double
    public :: dtl_send_int32, dtl_recv_int32

    ! Typed gather/scatter wrappers
    public :: dtl_gather_double, dtl_scatter_double
    public :: dtl_reduce_double

    ! Typed scan wrappers
    public :: dtl_scan_double

    ! Fortran shape convenience constructors
    public :: dtl_make_shape_1d, dtl_make_shape_2d, dtl_make_shape_3d

    ! ======================================================================
    ! C API interfaces needed for wrappers
    ! ======================================================================

    interface
        function dtl_broadcast_c(ctx, buf, count, dtype, root) &
                bind(c, name='dtl_broadcast')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_broadcast_c
        end function

        function dtl_allreduce_c(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_allreduce')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_allreduce_c
        end function

        function dtl_send_c(ctx, buf, count, dtype, dest, tag) &
                bind(c, name='dtl_send')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: dest
            integer(c_int), value :: tag
            integer(c_int) :: dtl_send_c
        end function

        function dtl_recv_c(ctx, buf, count, dtype, source, tag) &
                bind(c, name='dtl_recv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: source
            integer(c_int), value :: tag
            integer(c_int) :: dtl_recv_c
        end function

        function dtl_gather_c(ctx, sendbuf, sendcount, senddtype, &
                              recvbuf, recvcount, recvdtype, root) &
                bind(c, name='dtl_gather')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_gather_c
        end function

        function dtl_scatter_c(ctx, sendbuf, sendcount, senddtype, &
                               recvbuf, recvcount, recvdtype, root) &
                bind(c, name='dtl_scatter')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_scatter_c
        end function

        function dtl_reduce_c(ctx, sendbuf, recvbuf, count, dtype, op, root) &
                bind(c, name='dtl_reduce')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int), value :: root
            integer(c_int) :: dtl_reduce_c
        end function

        function dtl_scan_c(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_scan')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_scan_c
        end function
    end interface

contains

    ! ======================================================================
    ! String Helpers
    ! ======================================================================

    !> Convert a C string pointer to a Fortran allocatable string
    function dtl_c_to_f_string(c_str) result(f_str)
        type(c_ptr), intent(in) :: c_str
        character(len=:), allocatable :: f_str

        character(kind=c_char), pointer :: chars(:)
        integer :: i, length
        integer, parameter :: MAX_LEN = 1024

        if (.not. c_associated(c_str)) then
            f_str = ""
            return
        end if

        call c_f_pointer(c_str, chars, [MAX_LEN])
        length = 0
        do i = 1, MAX_LEN
            if (chars(i) == c_null_char) exit
            length = length + 1
        end do

        allocate(character(len=length) :: f_str)
        do i = 1, length
            f_str(i:i) = chars(i)
        end do
    end function dtl_c_to_f_string

    !> Get error message as a Fortran string
    function dtl_get_error_message(status) result(msg)
        integer(c_int), intent(in) :: status
        character(len=:), allocatable :: msg

        msg = dtl_c_to_f_string(dtl_status_message(status))
    end function dtl_get_error_message

    ! ======================================================================
    ! Typed Broadcast Wrappers
    ! ======================================================================

    function dtl_broadcast_double(ctx, data, count, root) result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(inout), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: root
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_broadcast_c(ctx, c_loc(data(1)), count, &
                                 DTL_DTYPE_FLOAT64, root)
    end function

    function dtl_broadcast_float(ctx, data, count, root) result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_float), intent(inout), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: root
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_broadcast_c(ctx, c_loc(data(1)), count, &
                                 DTL_DTYPE_FLOAT32, root)
    end function

    function dtl_broadcast_int32(ctx, data, count, root) result(status)
        type(c_ptr), intent(in) :: ctx
        integer(c_int32_t), intent(inout), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: root
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_broadcast_c(ctx, c_loc(data(1)), count, &
                                 DTL_DTYPE_INT32, root)
    end function

    function dtl_broadcast_int64(ctx, data, count, root) result(status)
        type(c_ptr), intent(in) :: ctx
        integer(c_int64_t), intent(inout), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: root
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_broadcast_c(ctx, c_loc(data(1)), count, &
                                 DTL_DTYPE_INT64, root)
    end function

    ! ======================================================================
    ! Typed Allreduce Wrappers
    ! ======================================================================

    function dtl_allreduce_sum_double(ctx, sendbuf, recvbuf, count) &
            result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: sendbuf(*)
        real(c_double), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_allreduce_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                                 count, DTL_DTYPE_FLOAT64, DTL_OP_SUM)
    end function

    function dtl_allreduce_sum_float(ctx, sendbuf, recvbuf, count) &
            result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_float), intent(in), target :: sendbuf(*)
        real(c_float), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_allreduce_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                                 count, DTL_DTYPE_FLOAT32, DTL_OP_SUM)
    end function

    function dtl_allreduce_sum_int32(ctx, sendbuf, recvbuf, count) &
            result(status)
        type(c_ptr), intent(in) :: ctx
        integer(c_int32_t), intent(in), target :: sendbuf(*)
        integer(c_int32_t), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_allreduce_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                                 count, DTL_DTYPE_INT32, DTL_OP_SUM)
    end function

    function dtl_allreduce_sum_int64(ctx, sendbuf, recvbuf, count) &
            result(status)
        type(c_ptr), intent(in) :: ctx
        integer(c_int64_t), intent(in), target :: sendbuf(*)
        integer(c_int64_t), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_allreduce_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                                 count, DTL_DTYPE_INT64, DTL_OP_SUM)
    end function

    function dtl_allreduce_max_double(ctx, sendbuf, recvbuf, count) &
            result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: sendbuf(*)
        real(c_double), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_allreduce_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                                 count, DTL_DTYPE_FLOAT64, DTL_OP_MAX)
    end function

    function dtl_allreduce_min_double(ctx, sendbuf, recvbuf, count) &
            result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: sendbuf(*)
        real(c_double), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int) :: status

        if (count <= 0) then
            status = DTL_SUCCESS
            return
        end if
        status = dtl_allreduce_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                                 count, DTL_DTYPE_FLOAT64, DTL_OP_MIN)
    end function

    ! ======================================================================
    ! Typed Send/Recv Wrappers
    ! ======================================================================

    function dtl_send_double(ctx, data, count, dest, tag) result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: dest, tag
        integer(c_int) :: status

        status = dtl_send_c(ctx, c_loc(data(1)), count, &
                            DTL_DTYPE_FLOAT64, dest, tag)
    end function

    function dtl_recv_double(ctx, data, count, source, tag) result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(out), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: source, tag
        integer(c_int) :: status

        status = dtl_recv_c(ctx, c_loc(data(1)), count, &
                            DTL_DTYPE_FLOAT64, source, tag)
    end function

    function dtl_send_int32(ctx, data, count, dest, tag) result(status)
        type(c_ptr), intent(in) :: ctx
        integer(c_int32_t), intent(in), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: dest, tag
        integer(c_int) :: status

        status = dtl_send_c(ctx, c_loc(data(1)), count, &
                            DTL_DTYPE_INT32, dest, tag)
    end function

    function dtl_recv_int32(ctx, data, count, source, tag) result(status)
        type(c_ptr), intent(in) :: ctx
        integer(c_int32_t), intent(out), target :: data(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: source, tag
        integer(c_int) :: status

        status = dtl_recv_c(ctx, c_loc(data(1)), count, &
                            DTL_DTYPE_INT32, source, tag)
    end function

    ! ======================================================================
    ! Typed Gather/Scatter/Reduce Wrappers
    ! ======================================================================

    function dtl_gather_double(ctx, sendbuf, sendcount, &
                               recvbuf, recvcount, root) result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: sendbuf(*)
        integer(c_int64_t), intent(in) :: sendcount
        real(c_double), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: recvcount
        integer(c_int), intent(in) :: root
        integer(c_int) :: status

        status = dtl_gather_c(ctx, c_loc(sendbuf(1)), sendcount, &
                              DTL_DTYPE_FLOAT64, c_loc(recvbuf(1)), &
                              recvcount, DTL_DTYPE_FLOAT64, root)
    end function

    function dtl_scatter_double(ctx, sendbuf, sendcount, &
                                recvbuf, recvcount, root) result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: sendbuf(*)
        integer(c_int64_t), intent(in) :: sendcount
        real(c_double), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: recvcount
        integer(c_int), intent(in) :: root
        integer(c_int) :: status

        status = dtl_scatter_c(ctx, c_loc(sendbuf(1)), sendcount, &
                               DTL_DTYPE_FLOAT64, c_loc(recvbuf(1)), &
                               recvcount, DTL_DTYPE_FLOAT64, root)
    end function

    function dtl_reduce_double(ctx, sendbuf, recvbuf, count, op, root) &
            result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: sendbuf(*)
        real(c_double), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: op, root
        integer(c_int) :: status

        status = dtl_reduce_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                              count, DTL_DTYPE_FLOAT64, op, root)
    end function

    function dtl_scan_double(ctx, sendbuf, recvbuf, count, op) result(status)
        type(c_ptr), intent(in) :: ctx
        real(c_double), intent(in), target :: sendbuf(*)
        real(c_double), intent(out), target :: recvbuf(*)
        integer(c_int64_t), intent(in) :: count
        integer(c_int), intent(in) :: op
        integer(c_int) :: status

        status = dtl_scan_c(ctx, c_loc(sendbuf(1)), c_loc(recvbuf(1)), &
                            count, DTL_DTYPE_FLOAT64, op)
    end function

    ! ======================================================================
    ! Fortran Shape Convenience Constructors
    ! ======================================================================

    function dtl_make_shape_1d(d0) result(s)
        integer(c_int64_t), intent(in) :: d0
        type(dtl_shape) :: s

        s%ndim = 1
        s%dims = 0
        s%dims(1) = d0
    end function

    function dtl_make_shape_2d(d0, d1) result(s)
        integer(c_int64_t), intent(in) :: d0, d1
        type(dtl_shape) :: s

        s%ndim = 2
        s%dims = 0
        s%dims(1) = d0
        s%dims(2) = d1
    end function

    function dtl_make_shape_3d(d0, d1, d2) result(s)
        integer(c_int64_t), intent(in) :: d0, d1, d2
        type(dtl_shape) :: s

        s%ndim = 3
        s%dims = 0
        s%dims(1) = d0
        s%dims(2) = d1
        s%dims(3) = d2
    end function

end module dtl_helpers
