! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_policies.f90
!> @brief DTL Policies Module - Partition, placement, execution policy enums
!> @since 0.1.0

module dtl_policies
    use, intrinsic :: iso_c_binding
    implicit none

    private

    ! Partition policies
    public :: DTL_PARTITION_BLOCK, DTL_PARTITION_CYCLIC
    public :: DTL_PARTITION_BLOCK_CYCLIC, DTL_PARTITION_HASH
    public :: DTL_PARTITION_REPLICATED

    ! Placement policies
    public :: DTL_PLACEMENT_HOST, DTL_PLACEMENT_DEVICE
    public :: DTL_PLACEMENT_UNIFIED, DTL_PLACEMENT_DEVICE_PREFERRED

    ! Execution policies
    public :: DTL_EXEC_SEQ, DTL_EXEC_PAR, DTL_EXEC_ASYNC

    ! Consistency policies
    public :: DTL_CONSISTENCY_BULK_SYNCHRONOUS, DTL_CONSISTENCY_RELAXED
    public :: DTL_CONSISTENCY_RELEASE_ACQUIRE, DTL_CONSISTENCY_SEQUENTIAL

    ! Error policies
    public :: DTL_ERROR_POLICY_RETURN_STATUS, DTL_ERROR_POLICY_CALLBACK
    public :: DTL_ERROR_POLICY_TERMINATE

    ! Container options type
    public :: dtl_container_options

    ! Policy functions
    public :: dtl_container_options_init
    public :: dtl_partition_policy_name, dtl_placement_policy_name
    public :: dtl_execution_policy_name, dtl_consistency_policy_name
    public :: dtl_error_policy_name, dtl_placement_available

    ! Policy-aware container creation
    public :: dtl_vector_create_with_options, dtl_array_create_with_options

    ! Policy queries (vector)
    public :: dtl_vector_partition_policy, dtl_vector_placement_policy
    public :: dtl_vector_execution_policy, dtl_vector_device_id
    public :: dtl_vector_consistency_policy, dtl_vector_error_policy

    ! Policy queries (array)
    public :: dtl_array_partition_policy, dtl_array_placement_policy
    public :: dtl_array_execution_policy, dtl_array_device_id
    public :: dtl_array_consistency_policy, dtl_array_error_policy

    ! Device memory copy helpers
    public :: dtl_vector_copy_to_host, dtl_vector_copy_from_host
    public :: dtl_array_copy_to_host, dtl_array_copy_from_host

    ! ======================================================================
    ! Policy Constants
    ! ======================================================================

    ! Partition policies
    integer(c_int), parameter :: DTL_PARTITION_BLOCK = 0
    integer(c_int), parameter :: DTL_PARTITION_CYCLIC = 1
    integer(c_int), parameter :: DTL_PARTITION_BLOCK_CYCLIC = 2
    integer(c_int), parameter :: DTL_PARTITION_HASH = 3
    integer(c_int), parameter :: DTL_PARTITION_REPLICATED = 4

    ! Placement policies
    integer(c_int), parameter :: DTL_PLACEMENT_HOST = 0
    integer(c_int), parameter :: DTL_PLACEMENT_DEVICE = 1
    integer(c_int), parameter :: DTL_PLACEMENT_UNIFIED = 2
    integer(c_int), parameter :: DTL_PLACEMENT_DEVICE_PREFERRED = 3

    ! Execution policies
    integer(c_int), parameter :: DTL_EXEC_SEQ = 0
    integer(c_int), parameter :: DTL_EXEC_PAR = 1
    integer(c_int), parameter :: DTL_EXEC_ASYNC = 2

    ! Consistency policies
    integer(c_int), parameter :: DTL_CONSISTENCY_BULK_SYNCHRONOUS = 0
    integer(c_int), parameter :: DTL_CONSISTENCY_RELAXED = 1
    integer(c_int), parameter :: DTL_CONSISTENCY_RELEASE_ACQUIRE = 2
    integer(c_int), parameter :: DTL_CONSISTENCY_SEQUENTIAL = 3

    ! Error policies
    integer(c_int), parameter :: DTL_ERROR_POLICY_RETURN_STATUS = 0
    integer(c_int), parameter :: DTL_ERROR_POLICY_CALLBACK = 1
    integer(c_int), parameter :: DTL_ERROR_POLICY_TERMINATE = 2

    ! ======================================================================
    ! Container Options Type
    ! ======================================================================

    type, bind(c) :: dtl_container_options
        integer(c_int) :: partition = 0     ! DTL_PARTITION_BLOCK
        integer(c_int) :: placement = 0     ! DTL_PLACEMENT_HOST
        integer(c_int) :: execution = 0     ! DTL_EXEC_SEQ
        integer(c_int) :: device_id = 0
        integer(c_int64_t) :: block_size = 1
        integer(c_int) :: reserved(3) = 0
    end type

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        subroutine dtl_container_options_init(opts) &
                bind(c, name='dtl_container_options_init')
            import :: dtl_container_options
            type(dtl_container_options), intent(out) :: opts
        end subroutine

        function dtl_partition_policy_name(policy) &
                bind(c, name='dtl_partition_policy_name')
            import :: c_ptr, c_int
            integer(c_int), value :: policy
            type(c_ptr) :: dtl_partition_policy_name
        end function

        function dtl_placement_policy_name(policy) &
                bind(c, name='dtl_placement_policy_name')
            import :: c_ptr, c_int
            integer(c_int), value :: policy
            type(c_ptr) :: dtl_placement_policy_name
        end function

        function dtl_execution_policy_name(policy) &
                bind(c, name='dtl_execution_policy_name')
            import :: c_ptr, c_int
            integer(c_int), value :: policy
            type(c_ptr) :: dtl_execution_policy_name
        end function

        function dtl_consistency_policy_name(policy) &
                bind(c, name='dtl_consistency_policy_name')
            import :: c_ptr, c_int
            integer(c_int), value :: policy
            type(c_ptr) :: dtl_consistency_policy_name
        end function

        function dtl_error_policy_name(policy) &
                bind(c, name='dtl_error_policy_name')
            import :: c_ptr, c_int
            integer(c_int), value :: policy
            type(c_ptr) :: dtl_error_policy_name
        end function

        function dtl_placement_available(policy) &
                bind(c, name='dtl_placement_available')
            import :: c_int
            integer(c_int), value :: policy
            integer(c_int) :: dtl_placement_available
        end function

        ! Policy-aware creation
        function dtl_vector_create_with_options(ctx, dtype, global_size, &
                                                opts, vec) &
                bind(c, name='dtl_vector_create_with_options')
            import :: c_ptr, c_int, c_int64_t, dtl_container_options
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: global_size
            type(dtl_container_options), intent(in) :: opts
            type(c_ptr), intent(out) :: vec
            integer(c_int) :: dtl_vector_create_with_options
        end function

        function dtl_array_create_with_options(ctx, dtype, size, opts, arr) &
                bind(c, name='dtl_array_create_with_options')
            import :: c_ptr, c_int, c_int64_t, dtl_container_options
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: size
            type(dtl_container_options), intent(in) :: opts
            type(c_ptr), intent(out) :: arr
            integer(c_int) :: dtl_array_create_with_options
        end function

        ! Policy queries (vector)
        function dtl_vector_partition_policy(vec) &
                bind(c, name='dtl_vector_partition_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_partition_policy
        end function

        function dtl_vector_placement_policy(vec) &
                bind(c, name='dtl_vector_placement_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_placement_policy
        end function

        function dtl_vector_execution_policy(vec) &
                bind(c, name='dtl_vector_execution_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_execution_policy
        end function

        function dtl_vector_device_id(vec) &
                bind(c, name='dtl_vector_device_id')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_device_id
        end function

        function dtl_vector_consistency_policy(vec) &
                bind(c, name='dtl_vector_consistency_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_consistency_policy
        end function

        function dtl_vector_error_policy(vec) &
                bind(c, name='dtl_vector_error_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_error_policy
        end function

        ! Policy queries (array)
        function dtl_array_partition_policy(arr) &
                bind(c, name='dtl_array_partition_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_partition_policy
        end function

        function dtl_array_placement_policy(arr) &
                bind(c, name='dtl_array_placement_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_placement_policy
        end function

        function dtl_array_execution_policy(arr) &
                bind(c, name='dtl_array_execution_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_execution_policy
        end function

        function dtl_array_device_id(arr) &
                bind(c, name='dtl_array_device_id')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_device_id
        end function

        function dtl_array_consistency_policy(arr) &
                bind(c, name='dtl_array_consistency_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_consistency_policy
        end function

        function dtl_array_error_policy(arr) &
                bind(c, name='dtl_array_error_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_error_policy
        end function

        ! Device memory copy helpers
        function dtl_vector_copy_to_host(vec, host_buffer, count) &
                bind(c, name='dtl_vector_copy_to_host')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: vec
            type(c_ptr), value :: host_buffer
            integer(c_int64_t), value :: count
            integer(c_int) :: dtl_vector_copy_to_host
        end function

        function dtl_vector_copy_from_host(vec, host_buffer, count) &
                bind(c, name='dtl_vector_copy_from_host')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: vec
            type(c_ptr), value :: host_buffer
            integer(c_int64_t), value :: count
            integer(c_int) :: dtl_vector_copy_from_host
        end function

        function dtl_array_copy_to_host(arr, host_buffer, count) &
                bind(c, name='dtl_array_copy_to_host')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: arr
            type(c_ptr), value :: host_buffer
            integer(c_int64_t), value :: count
            integer(c_int) :: dtl_array_copy_to_host
        end function

        function dtl_array_copy_from_host(arr, host_buffer, count) &
                bind(c, name='dtl_array_copy_from_host')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: arr
            type(c_ptr), value :: host_buffer
            integer(c_int64_t), value :: count
            integer(c_int) :: dtl_array_copy_from_host
        end function

    end interface

end module dtl_policies
