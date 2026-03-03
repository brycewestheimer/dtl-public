! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_fortran.f90
!> @brief DTL Fortran bindings module
!>
!> Provides Fortran interfaces to DTL C bindings using ISO_C_BINDING.
!> This module allows Fortran programs to use DTL distributed containers.

module dtl_fortran
    use, intrinsic :: iso_c_binding
    implicit none

    !--------------------------------------------------------------------------
    ! Constants
    !--------------------------------------------------------------------------

    ! Data types
    integer(c_int), parameter :: DTL_DTYPE_INT8    = 0
    integer(c_int), parameter :: DTL_DTYPE_INT16   = 1
    integer(c_int), parameter :: DTL_DTYPE_INT32   = 2
    integer(c_int), parameter :: DTL_DTYPE_INT64   = 3
    integer(c_int), parameter :: DTL_DTYPE_UINT8   = 4
    integer(c_int), parameter :: DTL_DTYPE_UINT16  = 5
    integer(c_int), parameter :: DTL_DTYPE_UINT32  = 6
    integer(c_int), parameter :: DTL_DTYPE_UINT64  = 7
    integer(c_int), parameter :: DTL_DTYPE_FLOAT32 = 8
    integer(c_int), parameter :: DTL_DTYPE_FLOAT64 = 9
    integer(c_int), parameter :: DTL_DTYPE_BYTE    = 10
    integer(c_int), parameter :: DTL_DTYPE_BOOL    = 11

    ! Reduce operations
    integer(c_int), parameter :: DTL_OP_SUM  = 0
    integer(c_int), parameter :: DTL_OP_PROD = 1
    integer(c_int), parameter :: DTL_OP_MIN  = 2
    integer(c_int), parameter :: DTL_OP_MAX  = 3
    integer(c_int), parameter :: DTL_OP_LAND = 4
    integer(c_int), parameter :: DTL_OP_LOR  = 5
    integer(c_int), parameter :: DTL_OP_BAND = 6
    integer(c_int), parameter :: DTL_OP_BOR  = 7

    ! Status codes
    integer(c_int32_t), parameter :: DTL_SUCCESS = 0

    ! Max tensor rank
    integer, parameter :: DTL_MAX_TENSOR_RANK = 8

    !--------------------------------------------------------------------------
    ! Types
    !--------------------------------------------------------------------------

    !> DTL shape type
    type, bind(c) :: dtl_shape
        integer(c_int) :: ndim
        integer(c_int64_t) :: dims(DTL_MAX_TENSOR_RANK)
    end type dtl_shape

    !--------------------------------------------------------------------------
    ! C Function Interfaces
    !--------------------------------------------------------------------------

    interface

        !----------------------------------------------------------------------
        ! Version and Feature Detection
        !----------------------------------------------------------------------

        function dtl_version_major() bind(c, name='dtl_version_major')
            import :: c_int
            integer(c_int) :: dtl_version_major
        end function

        function dtl_version_minor() bind(c, name='dtl_version_minor')
            import :: c_int
            integer(c_int) :: dtl_version_minor
        end function

        function dtl_version_patch() bind(c, name='dtl_version_patch')
            import :: c_int
            integer(c_int) :: dtl_version_patch
        end function

        function dtl_has_mpi() bind(c, name='dtl_has_mpi')
            import :: c_int
            integer(c_int) :: dtl_has_mpi
        end function

        function dtl_has_cuda() bind(c, name='dtl_has_cuda')
            import :: c_int
            integer(c_int) :: dtl_has_cuda
        end function

        !----------------------------------------------------------------------
        ! Status
        !----------------------------------------------------------------------

        function dtl_status_ok(status) bind(c, name='dtl_status_ok')
            import :: c_int, c_int32_t
            integer(c_int32_t), value :: status
            integer(c_int) :: dtl_status_ok
        end function

        function dtl_status_message(status) bind(c, name='dtl_status_message')
            import :: c_ptr, c_int32_t
            integer(c_int32_t), value :: status
            type(c_ptr) :: dtl_status_message
        end function

        !----------------------------------------------------------------------
        ! Context
        !----------------------------------------------------------------------

        function dtl_context_create_default(ctx) bind(c, name='dtl_context_create_default')
            import :: c_ptr, c_int32_t
            type(c_ptr), intent(out) :: ctx
            integer(c_int32_t) :: dtl_context_create_default
        end function

        subroutine dtl_context_destroy(ctx) bind(c, name='dtl_context_destroy')
            import :: c_ptr
            type(c_ptr), value :: ctx
        end subroutine

        function dtl_context_rank(ctx) bind(c, name='dtl_context_rank')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: ctx
            integer(c_int32_t) :: dtl_context_rank
        end function

        function dtl_context_size(ctx) bind(c, name='dtl_context_size')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: ctx
            integer(c_int32_t) :: dtl_context_size
        end function

        function dtl_context_is_root(ctx) bind(c, name='dtl_context_is_root')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_is_root
        end function

        !----------------------------------------------------------------------
        ! Collective Operations
        !----------------------------------------------------------------------

        function dtl_barrier(ctx) bind(c, name='dtl_barrier')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: ctx
            integer(c_int32_t) :: dtl_barrier
        end function

        function dtl_broadcast(ctx, data, count, dtype, root) bind(c, name='dtl_broadcast')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: data
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int32_t), value :: root
            integer(c_int32_t) :: dtl_broadcast
        end function

        function dtl_reduce(ctx, sendbuf, recvbuf, count, dtype, op, root) &
                bind(c, name='dtl_reduce')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int32_t), value :: root
            integer(c_int32_t) :: dtl_reduce
        end function

        function dtl_allreduce(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_allreduce')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int32_t) :: dtl_allreduce
        end function

        !----------------------------------------------------------------------
        ! Distributed Vector
        !----------------------------------------------------------------------

        function dtl_vector_create(ctx, dtype, size, vec) bind(c, name='dtl_vector_create')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: size
            type(c_ptr), intent(out) :: vec
            integer(c_int32_t) :: dtl_vector_create
        end function

        subroutine dtl_vector_destroy(vec) bind(c, name='dtl_vector_destroy')
            import :: c_ptr
            type(c_ptr), value :: vec
        end subroutine

        function dtl_vector_global_size(vec) bind(c, name='dtl_vector_global_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_global_size
        end function

        function dtl_vector_local_size(vec) bind(c, name='dtl_vector_local_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_local_size
        end function

        function dtl_vector_local_offset(vec) bind(c, name='dtl_vector_local_offset')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_local_offset
        end function

        function dtl_vector_local_data_mut(vec) bind(c, name='dtl_vector_local_data_mut')
            import :: c_ptr
            type(c_ptr), value :: vec
            type(c_ptr) :: dtl_vector_local_data_mut
        end function

        function dtl_vector_fill_local(vec, value) bind(c, name='dtl_vector_fill_local')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: vec
            type(c_ptr), value :: value
            integer(c_int32_t) :: dtl_vector_fill_local
        end function

        function dtl_vector_barrier(vec) bind(c, name='dtl_vector_barrier')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: vec
            integer(c_int32_t) :: dtl_vector_barrier
        end function

        !----------------------------------------------------------------------
        ! Distributed Tensor
        !----------------------------------------------------------------------

        function dtl_tensor_create(ctx, dtype, shape, tensor) bind(c, name='dtl_tensor_create')
            import :: c_ptr, c_int32_t, c_int, dtl_shape
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            type(dtl_shape), value :: shape
            type(c_ptr), intent(out) :: tensor
            integer(c_int32_t) :: dtl_tensor_create
        end function

        subroutine dtl_tensor_destroy(tensor) bind(c, name='dtl_tensor_destroy')
            import :: c_ptr
            type(c_ptr), value :: tensor
        end subroutine

        subroutine dtl_tensor_shape(tensor, shape) bind(c, name='dtl_tensor_shape')
            import :: c_ptr, dtl_shape
            type(c_ptr), value :: tensor
            type(dtl_shape), intent(out) :: shape
        end subroutine

        subroutine dtl_tensor_local_shape(tensor, shape) bind(c, name='dtl_tensor_local_shape')
            import :: c_ptr, dtl_shape
            type(c_ptr), value :: tensor
            type(dtl_shape), intent(out) :: shape
        end subroutine

        function dtl_tensor_global_size(tensor) bind(c, name='dtl_tensor_global_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t) :: dtl_tensor_global_size
        end function

        function dtl_tensor_local_size(tensor) bind(c, name='dtl_tensor_local_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t) :: dtl_tensor_local_size
        end function

        function dtl_tensor_local_data_mut(tensor) bind(c, name='dtl_tensor_local_data_mut')
            import :: c_ptr
            type(c_ptr), value :: tensor
            type(c_ptr) :: dtl_tensor_local_data_mut
        end function

        function dtl_tensor_fill_local(tensor, value) bind(c, name='dtl_tensor_fill_local')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: tensor
            type(c_ptr), value :: value
            integer(c_int32_t) :: dtl_tensor_fill_local
        end function

        !----------------------------------------------------------------------
        ! Point-to-Point Communication
        !----------------------------------------------------------------------

        function dtl_send(ctx, buf, count, dtype, dest, tag) &
                bind(c, name='dtl_send')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int32_t), value :: dest
            integer(c_int32_t), value :: tag
            integer(c_int32_t) :: dtl_send
        end function

        function dtl_recv(ctx, buf, count, dtype, source, tag) &
                bind(c, name='dtl_recv')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int32_t), value :: source
            integer(c_int32_t), value :: tag
            integer(c_int32_t) :: dtl_recv
        end function

        function dtl_sendrecv(ctx, sendbuf, sendcount, senddtype, dest, sendtag, &
                              recvbuf, recvcount, recvdtype, source, recvtag) &
                bind(c, name='dtl_sendrecv')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            integer(c_int32_t), value :: dest
            integer(c_int32_t), value :: sendtag
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int32_t), value :: source
            integer(c_int32_t), value :: recvtag
            integer(c_int32_t) :: dtl_sendrecv
        end function

        !----------------------------------------------------------------------
        ! Gather / Scatter / Allgather
        !----------------------------------------------------------------------

        function dtl_gather(ctx, sendbuf, sendcount, senddtype, &
                            recvbuf, recvcount, recvdtype, root) &
                bind(c, name='dtl_gather')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int32_t), value :: root
            integer(c_int32_t) :: dtl_gather
        end function

        function dtl_scatter(ctx, sendbuf, sendcount, senddtype, &
                             recvbuf, recvcount, recvdtype, root) &
                bind(c, name='dtl_scatter')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int32_t), value :: root
            integer(c_int32_t) :: dtl_scatter
        end function

        function dtl_allgather(ctx, sendbuf, sendcount, senddtype, &
                               recvbuf, recvcount, recvdtype) &
                bind(c, name='dtl_allgather')
            import :: c_ptr, c_int32_t, c_int64_t, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int32_t) :: dtl_allgather
        end function

        !----------------------------------------------------------------------
        ! Topology
        !----------------------------------------------------------------------

        function dtl_topology_num_cpus(count) bind(c, name='dtl_topology_num_cpus')
            import :: c_int, c_int32_t
            integer(c_int), intent(out) :: count
            integer(c_int32_t) :: dtl_topology_num_cpus
        end function

        function dtl_topology_num_gpus(count) bind(c, name='dtl_topology_num_gpus')
            import :: c_int, c_int32_t
            integer(c_int), intent(out) :: count
            integer(c_int32_t) :: dtl_topology_num_gpus
        end function

        function dtl_topology_cpu_affinity(rank, cpu_id) &
                bind(c, name='dtl_topology_cpu_affinity')
            import :: c_int, c_int32_t
            integer(c_int32_t), value :: rank
            integer(c_int), intent(out) :: cpu_id
            integer(c_int32_t) :: dtl_topology_cpu_affinity
        end function

        function dtl_topology_gpu_id(rank, gpu_id) &
                bind(c, name='dtl_topology_gpu_id')
            import :: c_int, c_int32_t
            integer(c_int32_t), value :: rank
            integer(c_int), intent(out) :: gpu_id
            integer(c_int32_t) :: dtl_topology_gpu_id
        end function

        function dtl_topology_is_local(rank_a, rank_b, is_local) &
                bind(c, name='dtl_topology_is_local')
            import :: c_int, c_int32_t
            integer(c_int32_t), value :: rank_a
            integer(c_int32_t), value :: rank_b
            integer(c_int), intent(out) :: is_local
            integer(c_int32_t) :: dtl_topology_is_local
        end function

        function dtl_topology_node_id(rank, node_id) &
                bind(c, name='dtl_topology_node_id')
            import :: c_int, c_int32_t
            integer(c_int32_t), value :: rank
            integer(c_int), intent(out) :: node_id
            integer(c_int32_t) :: dtl_topology_node_id
        end function

        !----------------------------------------------------------------------
        ! RMA (Remote Memory Access)
        !----------------------------------------------------------------------

        function dtl_window_create(ctx, base, size, win) &
                bind(c, name='dtl_window_create')
            import :: c_ptr, c_int32_t, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: base
            integer(c_int64_t), value :: size
            type(c_ptr), intent(out) :: win
            integer(c_int32_t) :: dtl_window_create
        end function

        subroutine dtl_window_destroy(win) bind(c, name='dtl_window_destroy')
            import :: c_ptr
            type(c_ptr), value :: win
        end subroutine

        function dtl_window_fence(win) bind(c, name='dtl_window_fence')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: win
            integer(c_int32_t) :: dtl_window_fence
        end function

        function dtl_window_lock(win, target, mode) &
                bind(c, name='dtl_window_lock')
            import :: c_ptr, c_int32_t, c_int
            type(c_ptr), value :: win
            integer(c_int32_t), value :: target
            integer(c_int), value :: mode
            integer(c_int32_t) :: dtl_window_lock
        end function

        function dtl_window_unlock(win, target) &
                bind(c, name='dtl_window_unlock')
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: win
            integer(c_int32_t), value :: target
            integer(c_int32_t) :: dtl_window_unlock
        end function

        function dtl_rma_put(win, target, target_offset, origin, size) &
                bind(c, name='dtl_rma_put')
            import :: c_ptr, c_int32_t, c_int64_t
            type(c_ptr), value :: win
            integer(c_int32_t), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: origin
            integer(c_int64_t), value :: size
            integer(c_int32_t) :: dtl_rma_put
        end function

        function dtl_rma_get(win, target, target_offset, buffer, size) &
                bind(c, name='dtl_rma_get')
            import :: c_ptr, c_int32_t, c_int64_t
            type(c_ptr), value :: win
            integer(c_int32_t), value :: target
            integer(c_int64_t), value :: target_offset
            type(c_ptr), value :: buffer
            integer(c_int64_t), value :: size
            integer(c_int32_t) :: dtl_rma_get
        end function

    end interface

contains

    !--------------------------------------------------------------------------
    ! Helper Functions
    !--------------------------------------------------------------------------

    !> Create a 1D shape
    function shape_1d(n1) result(s)
        integer(c_int64_t), intent(in) :: n1
        type(dtl_shape) :: s

        s%ndim = 1
        s%dims(1) = n1
        s%dims(2:DTL_MAX_TENSOR_RANK) = 0
    end function

    !> Create a 2D shape
    function shape_2d(n1, n2) result(s)
        integer(c_int64_t), intent(in) :: n1, n2
        type(dtl_shape) :: s

        s%ndim = 2
        s%dims(1) = n1
        s%dims(2) = n2
        s%dims(3:DTL_MAX_TENSOR_RANK) = 0
    end function

    !> Create a 3D shape
    function shape_3d(n1, n2, n3) result(s)
        integer(c_int64_t), intent(in) :: n1, n2, n3
        type(dtl_shape) :: s

        s%ndim = 3
        s%dims(1) = n1
        s%dims(2) = n2
        s%dims(3) = n3
        s%dims(4:DTL_MAX_TENSOR_RANK) = 0
    end function

    !> Check if status is success
    function is_success(status) result(ok)
        integer(c_int32_t), intent(in) :: status
        logical :: ok
        ok = dtl_status_ok(status) /= 0
    end function

end module dtl_fortran
