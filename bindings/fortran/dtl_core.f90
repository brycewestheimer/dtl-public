! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_core.f90
!> @brief DTL Core Module - Status codes, data types, reduce ops, constants, types
!> @since 0.1.0
!>
!> This module defines all core constants, derived types, and callback abstract
!> interfaces used throughout the DTL Fortran bindings.

module dtl_core
    use, intrinsic :: iso_c_binding
    implicit none

    private

    ! Re-export commonly used ISO_C_BINDING types for convenience
    public :: c_ptr, c_int, c_int8_t, c_int16_t, c_int32_t, c_int64_t
    public :: c_double, c_float, c_null_ptr, c_null_char
    public :: c_f_pointer, c_associated, c_loc, c_funloc, c_funptr

    ! ==========================================================================
    ! Status Codes
    ! ==========================================================================
    public :: DTL_SUCCESS, DTL_NOT_FOUND, DTL_END

    ! Communication errors (100-199)
    public :: DTL_ERROR_COMMUNICATION, DTL_ERROR_SEND_FAILED
    public :: DTL_ERROR_RECV_FAILED, DTL_ERROR_BROADCAST_FAILED
    public :: DTL_ERROR_REDUCE_FAILED, DTL_ERROR_BARRIER_FAILED
    public :: DTL_ERROR_TIMEOUT, DTL_ERROR_CANCELED
    public :: DTL_ERROR_CONNECTION_LOST, DTL_ERROR_RANK_FAILURE
    public :: DTL_ERROR_COLLECTIVE_FAILED, DTL_ERROR_COLLECTIVE_PARTICIPATION

    ! Memory errors (200-299)
    public :: DTL_ERROR_MEMORY, DTL_ERROR_ALLOCATION_FAILED
    public :: DTL_ERROR_OUT_OF_MEMORY, DTL_ERROR_INVALID_POINTER
    public :: DTL_ERROR_TRANSFER_FAILED, DTL_ERROR_DEVICE_MEMORY

    ! Serialization errors (300-399)
    public :: DTL_ERROR_SERIALIZATION, DTL_ERROR_SERIALIZE_FAILED
    public :: DTL_ERROR_DESERIALIZE_FAILED, DTL_ERROR_BUFFER_TOO_SMALL
    public :: DTL_ERROR_INVALID_FORMAT

    ! Bounds/argument errors (400-499)
    public :: DTL_ERROR_BOUNDS, DTL_ERROR_OUT_OF_BOUNDS
    public :: DTL_ERROR_INVALID_INDEX, DTL_ERROR_INVALID_RANK
    public :: DTL_ERROR_DIMENSION_MISMATCH, DTL_ERROR_EXTENT_MISMATCH
    public :: DTL_ERROR_KEY_NOT_FOUND, DTL_ERROR_OUT_OF_RANGE
    public :: DTL_ERROR_INVALID_ARGUMENT, DTL_ERROR_NULL_POINTER
    public :: DTL_ERROR_NOT_SUPPORTED

    ! Backend errors (500-599)
    public :: DTL_ERROR_BACKEND, DTL_ERROR_BACKEND_UNAVAILABLE
    public :: DTL_ERROR_BACKEND_INIT_FAILED, DTL_ERROR_BACKEND_INVALID
    public :: DTL_ERROR_CUDA, DTL_ERROR_HIP
    public :: DTL_ERROR_MPI, DTL_ERROR_NCCL, DTL_ERROR_SHMEM

    ! Algorithm errors (600-699)
    public :: DTL_ERROR_ALGORITHM, DTL_ERROR_PRECONDITION_FAILED
    public :: DTL_ERROR_POSTCONDITION_FAILED, DTL_ERROR_CONVERGENCE_FAILED

    ! Consistency errors (700-799)
    public :: DTL_ERROR_CONSISTENCY, DTL_ERROR_CONSISTENCY_VIOLATION
    public :: DTL_ERROR_STRUCTURAL_INVALIDATION

    ! Internal errors (900-999)
    public :: DTL_ERROR_INTERNAL, DTL_ERROR_NOT_IMPLEMENTED
    public :: DTL_ERROR_INVALID_STATE, DTL_ERROR_UNKNOWN

    ! Status categories
    public :: DTL_CATEGORY_SUCCESS, DTL_CATEGORY_COMMUNICATION
    public :: DTL_CATEGORY_MEMORY, DTL_CATEGORY_SERIALIZATION
    public :: DTL_CATEGORY_BOUNDS, DTL_CATEGORY_BACKEND
    public :: DTL_CATEGORY_ALGORITHM, DTL_CATEGORY_CONSISTENCY
    public :: DTL_CATEGORY_INTERNAL

    ! ==========================================================================
    ! Data types
    ! ==========================================================================
    public :: DTL_DTYPE_INT8, DTL_DTYPE_INT16, DTL_DTYPE_INT32, DTL_DTYPE_INT64
    public :: DTL_DTYPE_UINT8, DTL_DTYPE_UINT16, DTL_DTYPE_UINT32, DTL_DTYPE_UINT64
    public :: DTL_DTYPE_FLOAT32, DTL_DTYPE_FLOAT64
    public :: DTL_DTYPE_BYTE, DTL_DTYPE_BOOL, DTL_DTYPE_COUNT

    ! ==========================================================================
    ! Reduction operations
    ! ==========================================================================
    public :: DTL_OP_SUM, DTL_OP_PROD, DTL_OP_MIN, DTL_OP_MAX
    public :: DTL_OP_LAND, DTL_OP_LOR, DTL_OP_BAND, DTL_OP_BOR
    public :: DTL_OP_LXOR, DTL_OP_BXOR, DTL_OP_MINLOC, DTL_OP_MAXLOC
    public :: DTL_OP_COUNT

    ! NCCL mode / operation enums
    public :: DTL_NCCL_MODE_NATIVE_ONLY, DTL_NCCL_MODE_HYBRID_PARITY
    public :: DTL_NCCL_OP_POINT_TO_POINT, DTL_NCCL_OP_BARRIER
    public :: DTL_NCCL_OP_BROADCAST, DTL_NCCL_OP_REDUCE
    public :: DTL_NCCL_OP_ALLREDUCE, DTL_NCCL_OP_GATHER
    public :: DTL_NCCL_OP_SCATTER, DTL_NCCL_OP_ALLGATHER
    public :: DTL_NCCL_OP_ALLTOALL, DTL_NCCL_OP_GATHERV
    public :: DTL_NCCL_OP_SCATTERV, DTL_NCCL_OP_ALLGATHERV
    public :: DTL_NCCL_OP_ALLTOALLV, DTL_NCCL_OP_SCAN
    public :: DTL_NCCL_OP_EXSCAN, DTL_NCCL_OP_LOGICAL_REDUCTION

    ! ==========================================================================
    ! Determinism/scheduling policies
    ! ==========================================================================
    public :: DTL_DETERMINISM_THROUGHPUT, DTL_DETERMINISM_DETERMINISTIC
    public :: DTL_REDUCTION_SCHEDULE_IMPLEMENTATION_DEFINED
    public :: DTL_REDUCTION_SCHEDULE_FIXED_TREE
    public :: DTL_PROGRESS_ORDERING_IMPLEMENTATION_DEFINED
    public :: DTL_PROGRESS_ORDERING_RANK_ORDERED

    ! ==========================================================================
    ! Span constants
    ! ==========================================================================
    public :: DTL_SPAN_NPOS

    ! ==========================================================================
    ! Communication constants
    ! ==========================================================================
    public :: DTL_ANY_SOURCE, DTL_ANY_TAG, DTL_NO_RANK

    ! ==========================================================================
    ! Tensor constants
    ! ==========================================================================
    public :: DTL_MAX_TENSOR_RANK

    ! ==========================================================================
    ! Derived types
    ! ==========================================================================
    public :: dtl_context_options, dtl_shape, dtl_message_info

    ! ==========================================================================
    ! Utility function interfaces
    ! ==========================================================================
    public :: dtl_status_message, dtl_status_name, dtl_status_category
    public :: dtl_status_category_code, dtl_status_ok, dtl_status_is_error
    public :: dtl_status_is_category
    public :: dtl_dtype_size, dtl_dtype_name, dtl_reduce_op_name
    public :: dtl_context_options_init
    public :: dtl_shape_1d, dtl_shape_2d, dtl_shape_3d, dtl_shape_nd
    public :: dtl_shape_size

    ! ==========================================================================
    ! Callback abstract interfaces (for algorithm callbacks)
    ! ==========================================================================
    public :: dtl_f_unary_func, dtl_f_const_unary_func
    public :: dtl_f_predicate, dtl_f_comparator
    public :: dtl_f_transform_func, dtl_f_binary_func
    public :: dtl_f_action_func

    ! ======================================================================
    ! Status Codes
    ! ======================================================================

    ! Success codes
    integer(c_int), parameter :: DTL_SUCCESS = 0
    integer(c_int), parameter :: DTL_NOT_FOUND = 1
    integer(c_int), parameter :: DTL_END = 2

    ! Communication errors (100-199)
    integer(c_int), parameter :: DTL_ERROR_COMMUNICATION = 100
    integer(c_int), parameter :: DTL_ERROR_SEND_FAILED = 101
    integer(c_int), parameter :: DTL_ERROR_RECV_FAILED = 102
    integer(c_int), parameter :: DTL_ERROR_BROADCAST_FAILED = 103
    integer(c_int), parameter :: DTL_ERROR_REDUCE_FAILED = 104
    integer(c_int), parameter :: DTL_ERROR_BARRIER_FAILED = 105
    integer(c_int), parameter :: DTL_ERROR_TIMEOUT = 106
    integer(c_int), parameter :: DTL_ERROR_CANCELED = 107
    integer(c_int), parameter :: DTL_ERROR_CONNECTION_LOST = 108
    integer(c_int), parameter :: DTL_ERROR_RANK_FAILURE = 109
    integer(c_int), parameter :: DTL_ERROR_COLLECTIVE_FAILED = 110
    integer(c_int), parameter :: DTL_ERROR_COLLECTIVE_PARTICIPATION = 111

    ! Memory errors (200-299)
    integer(c_int), parameter :: DTL_ERROR_MEMORY = 200
    integer(c_int), parameter :: DTL_ERROR_ALLOCATION_FAILED = 201
    integer(c_int), parameter :: DTL_ERROR_OUT_OF_MEMORY = 202
    integer(c_int), parameter :: DTL_ERROR_INVALID_POINTER = 203
    integer(c_int), parameter :: DTL_ERROR_TRANSFER_FAILED = 204
    integer(c_int), parameter :: DTL_ERROR_DEVICE_MEMORY = 205

    ! Serialization errors (300-399)
    integer(c_int), parameter :: DTL_ERROR_SERIALIZATION = 300
    integer(c_int), parameter :: DTL_ERROR_SERIALIZE_FAILED = 301
    integer(c_int), parameter :: DTL_ERROR_DESERIALIZE_FAILED = 302
    integer(c_int), parameter :: DTL_ERROR_BUFFER_TOO_SMALL = 303
    integer(c_int), parameter :: DTL_ERROR_INVALID_FORMAT = 304

    ! Bounds/argument errors (400-499)
    integer(c_int), parameter :: DTL_ERROR_BOUNDS = 400
    integer(c_int), parameter :: DTL_ERROR_OUT_OF_BOUNDS = 401
    integer(c_int), parameter :: DTL_ERROR_INVALID_INDEX = 402
    integer(c_int), parameter :: DTL_ERROR_INVALID_RANK = 403
    integer(c_int), parameter :: DTL_ERROR_DIMENSION_MISMATCH = 404
    integer(c_int), parameter :: DTL_ERROR_EXTENT_MISMATCH = 405
    integer(c_int), parameter :: DTL_ERROR_KEY_NOT_FOUND = 406
    integer(c_int), parameter :: DTL_ERROR_OUT_OF_RANGE = 407
    integer(c_int), parameter :: DTL_ERROR_INVALID_ARGUMENT = 410
    integer(c_int), parameter :: DTL_ERROR_NULL_POINTER = 411
    integer(c_int), parameter :: DTL_ERROR_NOT_SUPPORTED = 420

    ! Backend errors (500-599)
    integer(c_int), parameter :: DTL_ERROR_BACKEND = 500
    integer(c_int), parameter :: DTL_ERROR_BACKEND_UNAVAILABLE = 501
    integer(c_int), parameter :: DTL_ERROR_BACKEND_INIT_FAILED = 502
    integer(c_int), parameter :: DTL_ERROR_BACKEND_INVALID = 503
    integer(c_int), parameter :: DTL_ERROR_CUDA = 510
    integer(c_int), parameter :: DTL_ERROR_HIP = 520
    integer(c_int), parameter :: DTL_ERROR_MPI = 530
    integer(c_int), parameter :: DTL_ERROR_NCCL = 540
    integer(c_int), parameter :: DTL_ERROR_SHMEM = 550

    ! Algorithm errors (600-699)
    integer(c_int), parameter :: DTL_ERROR_ALGORITHM = 600
    integer(c_int), parameter :: DTL_ERROR_PRECONDITION_FAILED = 601
    integer(c_int), parameter :: DTL_ERROR_POSTCONDITION_FAILED = 602
    integer(c_int), parameter :: DTL_ERROR_CONVERGENCE_FAILED = 603

    ! Consistency errors (700-799)
    integer(c_int), parameter :: DTL_ERROR_CONSISTENCY = 700
    integer(c_int), parameter :: DTL_ERROR_CONSISTENCY_VIOLATION = 701
    integer(c_int), parameter :: DTL_ERROR_STRUCTURAL_INVALIDATION = 702

    ! Internal errors (900-999)
    integer(c_int), parameter :: DTL_ERROR_INTERNAL = 900
    integer(c_int), parameter :: DTL_ERROR_NOT_IMPLEMENTED = 901
    integer(c_int), parameter :: DTL_ERROR_INVALID_STATE = 902
    integer(c_int), parameter :: DTL_ERROR_UNKNOWN = 999

    ! Status category codes
    integer(c_int), parameter :: DTL_CATEGORY_SUCCESS = 0
    integer(c_int), parameter :: DTL_CATEGORY_COMMUNICATION = 1
    integer(c_int), parameter :: DTL_CATEGORY_MEMORY = 2
    integer(c_int), parameter :: DTL_CATEGORY_SERIALIZATION = 3
    integer(c_int), parameter :: DTL_CATEGORY_BOUNDS = 4
    integer(c_int), parameter :: DTL_CATEGORY_BACKEND = 5
    integer(c_int), parameter :: DTL_CATEGORY_ALGORITHM = 6
    integer(c_int), parameter :: DTL_CATEGORY_CONSISTENCY = 7
    integer(c_int), parameter :: DTL_CATEGORY_INTERNAL = 9

    ! ======================================================================
    ! Data Type Enumeration
    ! ======================================================================

    integer(c_int), parameter :: DTL_DTYPE_INT8 = 0
    integer(c_int), parameter :: DTL_DTYPE_INT16 = 1
    integer(c_int), parameter :: DTL_DTYPE_INT32 = 2
    integer(c_int), parameter :: DTL_DTYPE_INT64 = 3
    integer(c_int), parameter :: DTL_DTYPE_UINT8 = 4
    integer(c_int), parameter :: DTL_DTYPE_UINT16 = 5
    integer(c_int), parameter :: DTL_DTYPE_UINT32 = 6
    integer(c_int), parameter :: DTL_DTYPE_UINT64 = 7
    integer(c_int), parameter :: DTL_DTYPE_FLOAT32 = 8
    integer(c_int), parameter :: DTL_DTYPE_FLOAT64 = 9
    integer(c_int), parameter :: DTL_DTYPE_BYTE = 10
    integer(c_int), parameter :: DTL_DTYPE_BOOL = 11
    integer(c_int), parameter :: DTL_DTYPE_COUNT = 12

    ! ======================================================================
    ! Reduction Operation Enumeration
    ! ======================================================================

    integer(c_int), parameter :: DTL_OP_SUM = 0
    integer(c_int), parameter :: DTL_OP_PROD = 1
    integer(c_int), parameter :: DTL_OP_MIN = 2
    integer(c_int), parameter :: DTL_OP_MAX = 3
    integer(c_int), parameter :: DTL_OP_LAND = 4
    integer(c_int), parameter :: DTL_OP_LOR = 5
    integer(c_int), parameter :: DTL_OP_BAND = 6
    integer(c_int), parameter :: DTL_OP_BOR = 7
    integer(c_int), parameter :: DTL_OP_LXOR = 8
    integer(c_int), parameter :: DTL_OP_BXOR = 9
    integer(c_int), parameter :: DTL_OP_MINLOC = 10
    integer(c_int), parameter :: DTL_OP_MAXLOC = 11
    integer(c_int), parameter :: DTL_OP_COUNT = 12

    ! ======================================================================
    ! NCCL Mode / Operation Enumerations
    ! ======================================================================

    integer(c_int), parameter :: DTL_NCCL_MODE_NATIVE_ONLY = 0
    integer(c_int), parameter :: DTL_NCCL_MODE_HYBRID_PARITY = 1

    integer(c_int), parameter :: DTL_NCCL_OP_POINT_TO_POINT = 0
    integer(c_int), parameter :: DTL_NCCL_OP_BARRIER = 1
    integer(c_int), parameter :: DTL_NCCL_OP_BROADCAST = 2
    integer(c_int), parameter :: DTL_NCCL_OP_REDUCE = 3
    integer(c_int), parameter :: DTL_NCCL_OP_ALLREDUCE = 4
    integer(c_int), parameter :: DTL_NCCL_OP_GATHER = 5
    integer(c_int), parameter :: DTL_NCCL_OP_SCATTER = 6
    integer(c_int), parameter :: DTL_NCCL_OP_ALLGATHER = 7
    integer(c_int), parameter :: DTL_NCCL_OP_ALLTOALL = 8
    integer(c_int), parameter :: DTL_NCCL_OP_GATHERV = 9
    integer(c_int), parameter :: DTL_NCCL_OP_SCATTERV = 10
    integer(c_int), parameter :: DTL_NCCL_OP_ALLGATHERV = 11
    integer(c_int), parameter :: DTL_NCCL_OP_ALLTOALLV = 12
    integer(c_int), parameter :: DTL_NCCL_OP_SCAN = 13
    integer(c_int), parameter :: DTL_NCCL_OP_EXSCAN = 14
    integer(c_int), parameter :: DTL_NCCL_OP_LOGICAL_REDUCTION = 15

    ! ======================================================================
    ! Determinism / Scheduling Policies
    ! ======================================================================

    integer(c_int), parameter :: DTL_DETERMINISM_THROUGHPUT = 0
    integer(c_int), parameter :: DTL_DETERMINISM_DETERMINISTIC = 1

    integer(c_int), parameter :: DTL_REDUCTION_SCHEDULE_IMPLEMENTATION_DEFINED = 0
    integer(c_int), parameter :: DTL_REDUCTION_SCHEDULE_FIXED_TREE = 1

    integer(c_int), parameter :: DTL_PROGRESS_ORDERING_IMPLEMENTATION_DEFINED = 0
    integer(c_int), parameter :: DTL_PROGRESS_ORDERING_RANK_ORDERED = 1

    ! ======================================================================
    ! Span Constants
    ! ======================================================================

    integer(c_int64_t), parameter :: DTL_SPAN_NPOS = -1_c_int64_t

    ! ======================================================================
    ! Communication Constants
    ! ======================================================================

    !> Wildcard: match any source rank
    integer(c_int), parameter :: DTL_NO_RANK = -1
    !> Wildcard: match any source rank in recv/probe
    integer(c_int), parameter :: DTL_ANY_SOURCE = -2
    !> Wildcard: match any tag in recv/probe
    integer(c_int), parameter :: DTL_ANY_TAG = -1

    ! ======================================================================
    ! Tensor Constants
    ! ======================================================================

    integer, parameter :: DTL_MAX_TENSOR_RANK = 8

    ! ======================================================================
    ! Derived Types (bind(c) for C ABI compatibility)
    ! ======================================================================

    !> Context creation options (matches C dtl_context_options)
    type, bind(c) :: dtl_context_options
        integer(c_int) :: device_id = -1
        integer(c_int) :: init_mpi = 1
        integer(c_int) :: finalize_mpi = 0
        integer(c_int) :: reserved(4) = 0
    end type

    !> N-dimensional shape descriptor (matches C dtl_shape)
    type, bind(c) :: dtl_shape
        integer(c_int) :: ndim = 0
        integer(c_int64_t) :: dims(8) = 0
    end type

    !> Message info from probe operations (matches C dtl_message_info_t)
    type, bind(c) :: dtl_message_info
        integer(c_int) :: source = 0
        integer(c_int) :: tag = 0
        integer(c_int64_t) :: count = 0
    end type

    ! ======================================================================
    ! Callback Abstract Interfaces
    ! ======================================================================

    abstract interface
        !> Mutable unary function callback: void(void* elem, uint64_t idx, void* user)
        subroutine dtl_f_unary_func(element, index, user_data) bind(c)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: element
            integer(c_int64_t), value :: index
            type(c_ptr), value :: user_data
        end subroutine

        !> Const unary function callback: void(const void* elem, uint64_t idx, void* user)
        subroutine dtl_f_const_unary_func(element, index, user_data) bind(c)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: element
            integer(c_int64_t), value :: index
            type(c_ptr), value :: user_data
        end subroutine

        !> Predicate callback: int(const void* elem, void* user)
        function dtl_f_predicate(element, user_data) bind(c) result(res)
            import :: c_ptr, c_int
            type(c_ptr), value :: element
            type(c_ptr), value :: user_data
            integer(c_int) :: res
        end function

        !> Comparator callback: int(const void* a, const void* b, void* user)
        function dtl_f_comparator(a, b, user_data) bind(c) result(res)
            import :: c_ptr, c_int
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: user_data
            integer(c_int) :: res
        end function

        !> Transform callback: void(const void* in, void* out, uint64_t idx, void* user)
        subroutine dtl_f_transform_func(input, output, index, user_data) bind(c)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: input
            type(c_ptr), value :: output
            integer(c_int64_t), value :: index
            type(c_ptr), value :: user_data
        end subroutine

        !> Binary function callback: void(const void* a, const void* b, void* result, void* user)
        subroutine dtl_f_binary_func(a, b, result, user_data) bind(c)
            import :: c_ptr
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: result
            type(c_ptr), value :: user_data
        end subroutine

        !> Action function callback for remote invocation
        function dtl_f_action_func(args, args_size, result, result_size, &
                                   user_data) bind(c) result(status)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: args
            integer(c_int64_t), value :: args_size
            type(c_ptr), value :: result
            integer(c_int64_t), value :: result_size
            type(c_ptr), value :: user_data
            integer(c_int) :: status
        end function
    end interface

    ! ======================================================================
    ! C API Utility Interfaces
    ! ======================================================================

    interface

        function dtl_status_message(status) bind(c, name='dtl_status_message')
            import :: c_ptr, c_int
            integer(c_int), value :: status
            type(c_ptr) :: dtl_status_message
        end function

        function dtl_status_name(status) bind(c, name='dtl_status_name')
            import :: c_ptr, c_int
            integer(c_int), value :: status
            type(c_ptr) :: dtl_status_name
        end function

        function dtl_status_category(status) bind(c, name='dtl_status_category')
            import :: c_ptr, c_int
            integer(c_int), value :: status
            type(c_ptr) :: dtl_status_category
        end function

        function dtl_status_category_code(status) &
                bind(c, name='dtl_status_category_code')
            import :: c_int
            integer(c_int), value :: status
            integer(c_int) :: dtl_status_category_code
        end function

        function dtl_status_ok(status) bind(c, name='dtl_status_ok')
            import :: c_int
            integer(c_int), value :: status
            integer(c_int) :: dtl_status_ok
        end function

        function dtl_status_is_error(status) bind(c, name='dtl_status_is_error')
            import :: c_int
            integer(c_int), value :: status
            integer(c_int) :: dtl_status_is_error
        end function

        function dtl_status_is_category(status, category_code) &
                bind(c, name='dtl_status_is_category')
            import :: c_int
            integer(c_int), value :: status
            integer(c_int), value :: category_code
            integer(c_int) :: dtl_status_is_category
        end function

        function dtl_dtype_size(dtype) bind(c, name='dtl_dtype_size')
            import :: c_int, c_int64_t
            integer(c_int), value :: dtype
            integer(c_int64_t) :: dtl_dtype_size
        end function

        function dtl_dtype_name(dtype) bind(c, name='dtl_dtype_name')
            import :: c_ptr, c_int
            integer(c_int), value :: dtype
            type(c_ptr) :: dtl_dtype_name
        end function

        function dtl_reduce_op_name(op) bind(c, name='dtl_reduce_op_name')
            import :: c_ptr, c_int
            integer(c_int), value :: op
            type(c_ptr) :: dtl_reduce_op_name
        end function

        subroutine dtl_context_options_init(opts) &
                bind(c, name='dtl_context_options_init')
            import :: dtl_context_options
            type(dtl_context_options), intent(out) :: opts
        end subroutine

        function dtl_shape_1d(dim0) bind(c, name='dtl_shape_1d')
            import :: dtl_shape, c_int64_t
            integer(c_int64_t), value :: dim0
            type(dtl_shape) :: dtl_shape_1d
        end function

        function dtl_shape_2d(dim0, dim1) bind(c, name='dtl_shape_2d')
            import :: dtl_shape, c_int64_t
            integer(c_int64_t), value :: dim0
            integer(c_int64_t), value :: dim1
            type(dtl_shape) :: dtl_shape_2d
        end function

        function dtl_shape_3d(dim0, dim1, dim2) bind(c, name='dtl_shape_3d')
            import :: dtl_shape, c_int64_t
            integer(c_int64_t), value :: dim0
            integer(c_int64_t), value :: dim1
            integer(c_int64_t), value :: dim2
            type(dtl_shape) :: dtl_shape_3d
        end function

        function dtl_shape_nd(ndim, dims) bind(c, name='dtl_shape_nd')
            import :: dtl_shape, c_int, c_int64_t
            integer(c_int), value :: ndim
            integer(c_int64_t), intent(in) :: dims(*)
            type(dtl_shape) :: dtl_shape_nd
        end function

        function dtl_shape_size(shape) bind(c, name='dtl_shape_size')
            import :: dtl_shape, c_int64_t
            type(dtl_shape), intent(in) :: shape
            integer(c_int64_t) :: dtl_shape_size
        end function

    end interface

end module dtl_core
