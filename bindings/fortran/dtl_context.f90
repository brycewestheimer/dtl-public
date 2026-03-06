! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_context.f90
!> @brief DTL Context Module - Context lifecycle, queries, and advanced operations
!> @since 0.1.0

module dtl_context
    use, intrinsic :: iso_c_binding
    use dtl_core, only: dtl_context_options
    implicit none

    private

    ! Context lifecycle
    public :: dtl_context_create, dtl_context_create_default, dtl_context_destroy

    ! Context queries
    public :: dtl_context_rank, dtl_context_size, dtl_context_is_root
    public :: dtl_context_device_id, dtl_context_has_device
    public :: dtl_context_is_valid

    ! Synchronization
    public :: dtl_context_barrier, dtl_context_fence

    ! Advanced operations
    public :: dtl_context_dup, dtl_context_split

    ! Domain queries (V1.3.0)
    public :: dtl_context_has_mpi, dtl_context_has_cuda
    public :: dtl_context_has_nccl, dtl_context_has_shmem
    public :: dtl_context_nccl_mode
    public :: dtl_context_nccl_supports_native
    public :: dtl_context_nccl_supports_hybrid

    ! Context transformations (V1.3.0)
    public :: dtl_context_with_cuda, dtl_context_with_nccl
    public :: dtl_context_split_nccl
    public :: dtl_context_with_nccl_ex, dtl_context_split_nccl_ex

    ! Policy queries
    public :: dtl_context_determinism_mode
    public :: dtl_context_reduction_schedule_policy
    public :: dtl_context_progress_ordering_policy

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        ! ------------------------------------------------------------------
        ! Context Lifecycle
        ! ------------------------------------------------------------------

        !> Create a context with options
        function dtl_context_create(ctx, opts) &
                bind(c, name='dtl_context_create')
            import :: c_ptr, c_int, dtl_context_options
            type(c_ptr), intent(out) :: ctx
            type(dtl_context_options), intent(in) :: opts
            integer(c_int) :: dtl_context_create
        end function

        !> Create a context with default options
        function dtl_context_create_default(ctx) &
                bind(c, name='dtl_context_create_default')
            import :: c_ptr, c_int
            type(c_ptr), intent(out) :: ctx
            integer(c_int) :: dtl_context_create_default
        end function

        !> Destroy a context
        subroutine dtl_context_destroy(ctx) bind(c, name='dtl_context_destroy')
            import :: c_ptr
            type(c_ptr), value :: ctx
        end subroutine

        ! ------------------------------------------------------------------
        ! Context Queries
        ! ------------------------------------------------------------------

        function dtl_context_rank(ctx) bind(c, name='dtl_context_rank')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_rank
        end function

        function dtl_context_size(ctx) bind(c, name='dtl_context_size')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_size
        end function

        function dtl_context_is_root(ctx) bind(c, name='dtl_context_is_root')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_is_root
        end function

        function dtl_context_device_id(ctx) &
                bind(c, name='dtl_context_device_id')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_device_id
        end function

        function dtl_context_has_device(ctx) &
                bind(c, name='dtl_context_has_device')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_has_device
        end function

        function dtl_context_is_valid(ctx) &
                bind(c, name='dtl_context_is_valid')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_is_valid
        end function

        ! ------------------------------------------------------------------
        ! Synchronization
        ! ------------------------------------------------------------------

        function dtl_context_barrier(ctx) &
                bind(c, name='dtl_context_barrier')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_barrier
        end function

        function dtl_context_fence(ctx) bind(c, name='dtl_context_fence')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_fence
        end function

        ! ------------------------------------------------------------------
        ! Advanced Operations
        ! ------------------------------------------------------------------

        !> Duplicate a context
        function dtl_context_dup(src, dst) bind(c, name='dtl_context_dup')
            import :: c_ptr, c_int
            type(c_ptr), value :: src
            type(c_ptr), intent(out) :: dst
            integer(c_int) :: dtl_context_dup
        end function

        !> Split a context by color and key
        function dtl_context_split(ctx, color, key, out) &
                bind(c, name='dtl_context_split')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: color
            integer(c_int), value :: key
            type(c_ptr), intent(out) :: out
            integer(c_int) :: dtl_context_split
        end function

        ! ------------------------------------------------------------------
        ! Domain Queries (V1.3.0)
        ! ------------------------------------------------------------------

        function dtl_context_has_mpi(ctx) &
                bind(c, name='dtl_context_has_mpi')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_has_mpi
        end function

        function dtl_context_has_cuda(ctx) &
                bind(c, name='dtl_context_has_cuda')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_has_cuda
        end function

        function dtl_context_has_nccl(ctx) &
                bind(c, name='dtl_context_has_nccl')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_has_nccl
        end function

        function dtl_context_nccl_mode(ctx) &
                bind(c, name='dtl_context_nccl_mode')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_nccl_mode
        end function

        function dtl_context_nccl_supports_native(ctx, op) &
                bind(c, name='dtl_context_nccl_supports_native')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: op
            integer(c_int) :: dtl_context_nccl_supports_native
        end function

        function dtl_context_nccl_supports_hybrid(ctx, op) &
                bind(c, name='dtl_context_nccl_supports_hybrid')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: op
            integer(c_int) :: dtl_context_nccl_supports_hybrid
        end function

        function dtl_context_has_shmem(ctx) &
                bind(c, name='dtl_context_has_shmem')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_has_shmem
        end function

        ! ------------------------------------------------------------------
        ! Context Transformations (V1.3.0)
        ! ------------------------------------------------------------------

        !> Create a new context with CUDA support
        function dtl_context_with_cuda(ctx, device_id, out) &
                bind(c, name='dtl_context_with_cuda')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: device_id
            type(c_ptr), intent(out) :: out
            integer(c_int) :: dtl_context_with_cuda
        end function

        !> Create a new context with NCCL support
        function dtl_context_with_nccl(ctx, device_id, out) &
                bind(c, name='dtl_context_with_nccl')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: device_id
            type(c_ptr), intent(out) :: out
            integer(c_int) :: dtl_context_with_nccl
        end function

        !> Create a new context with NCCL support and explicit mode
        function dtl_context_with_nccl_ex(ctx, device_id, mode, out) &
                bind(c, name='dtl_context_with_nccl_ex')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: device_id
            integer(c_int), value :: mode
            type(c_ptr), intent(out) :: out
            integer(c_int) :: dtl_context_with_nccl_ex
        end function

        !> Split context creating sub-groups with NCCL communicators
        function dtl_context_split_nccl(ctx, color, key, out) &
                bind(c, name='dtl_context_split_nccl')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: color
            integer(c_int), value :: key
            type(c_ptr), intent(out) :: out
            integer(c_int) :: dtl_context_split_nccl
        end function

        !> Split context creating sub-groups with NCCL communicators and explicit mode/device
        function dtl_context_split_nccl_ex(ctx, color, key, device_id, mode, out) &
                bind(c, name='dtl_context_split_nccl_ex')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int), value :: color
            integer(c_int), value :: key
            integer(c_int), value :: device_id
            integer(c_int), value :: mode
            type(c_ptr), intent(out) :: out
            integer(c_int) :: dtl_context_split_nccl_ex
        end function

        ! ------------------------------------------------------------------
        ! Policy Queries
        ! ------------------------------------------------------------------

        function dtl_context_determinism_mode(ctx) &
                bind(c, name='dtl_context_determinism_mode')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_determinism_mode
        end function

        function dtl_context_reduction_schedule_policy(ctx) &
                bind(c, name='dtl_context_reduction_schedule_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_reduction_schedule_policy
        end function

        function dtl_context_progress_ordering_policy(ctx) &
                bind(c, name='dtl_context_progress_ordering_policy')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_progress_ordering_policy
        end function

    end interface

end module dtl_context
