! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_environment.f90
!> @brief DTL Environment Module - Environment lifecycle and backend queries
!> @since 0.1.0

module dtl_environment
    use, intrinsic :: iso_c_binding
    implicit none

    private

    ! Environment lifecycle
    public :: dtl_environment_create, dtl_environment_create_with_args
    public :: dtl_environment_destroy

    ! Environment state queries
    public :: dtl_environment_is_initialized, dtl_environment_ref_count
    public :: dtl_environment_domain

    ! Backend availability (environment-level)
    public :: dtl_environment_has_mpi, dtl_environment_has_cuda
    public :: dtl_environment_has_hip, dtl_environment_has_nccl
    public :: dtl_environment_has_shmem, dtl_environment_mpi_thread_level

    ! Context factories from environment
    public :: dtl_environment_make_world_context
    public :: dtl_environment_make_world_context_gpu
    public :: dtl_environment_make_cpu_context

    ! Backend queries (global)
    public :: dtl_backend_name, dtl_backend_count, dtl_version

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        ! ------------------------------------------------------------------
        ! Environment Lifecycle
        ! ------------------------------------------------------------------

        !> Create a DTL environment with default options
        function dtl_environment_create(env) &
                bind(c, name='dtl_environment_create')
            import :: c_ptr, c_int
            type(c_ptr), intent(out) :: env
            integer(c_int) :: dtl_environment_create
        end function

        !> Create a DTL environment with command-line arguments
        function dtl_environment_create_with_args(env, argc, argv) &
                bind(c, name='dtl_environment_create_with_args')
            import :: c_ptr, c_int
            type(c_ptr), intent(out) :: env
            integer(c_int), intent(inout) :: argc
            type(c_ptr), intent(inout) :: argv
            integer(c_int) :: dtl_environment_create_with_args
        end function

        !> Destroy a DTL environment
        subroutine dtl_environment_destroy(env) &
                bind(c, name='dtl_environment_destroy')
            import :: c_ptr
            type(c_ptr), value :: env
        end subroutine

        ! ------------------------------------------------------------------
        ! Environment State Queries
        ! ------------------------------------------------------------------

        !> Check if the environment is initialized
        function dtl_environment_is_initialized() &
                bind(c, name='dtl_environment_is_initialized')
            import :: c_int
            integer(c_int) :: dtl_environment_is_initialized
        end function

        !> Get environment reference count
        function dtl_environment_ref_count() &
                bind(c, name='dtl_environment_ref_count')
            import :: c_int64_t
            integer(c_int64_t) :: dtl_environment_ref_count
        end function

        !> Get the environment domain string
        function dtl_environment_domain(env) &
                bind(c, name='dtl_environment_domain')
            import :: c_ptr
            type(c_ptr), value :: env
            type(c_ptr) :: dtl_environment_domain
        end function

        ! ------------------------------------------------------------------
        ! Backend Availability
        ! ------------------------------------------------------------------

        function dtl_environment_has_mpi() &
                bind(c, name='dtl_environment_has_mpi')
            import :: c_int
            integer(c_int) :: dtl_environment_has_mpi
        end function

        function dtl_environment_has_cuda() &
                bind(c, name='dtl_environment_has_cuda')
            import :: c_int
            integer(c_int) :: dtl_environment_has_cuda
        end function

        function dtl_environment_has_hip() &
                bind(c, name='dtl_environment_has_hip')
            import :: c_int
            integer(c_int) :: dtl_environment_has_hip
        end function

        function dtl_environment_has_nccl() &
                bind(c, name='dtl_environment_has_nccl')
            import :: c_int
            integer(c_int) :: dtl_environment_has_nccl
        end function

        function dtl_environment_has_shmem() &
                bind(c, name='dtl_environment_has_shmem')
            import :: c_int
            integer(c_int) :: dtl_environment_has_shmem
        end function

        function dtl_environment_mpi_thread_level() &
                bind(c, name='dtl_environment_mpi_thread_level')
            import :: c_int
            integer(c_int) :: dtl_environment_mpi_thread_level
        end function

        ! ------------------------------------------------------------------
        ! Context Factories
        ! ------------------------------------------------------------------

        !> Create a world context from the environment
        function dtl_environment_make_world_context(env, ctx) &
                bind(c, name='dtl_environment_make_world_context')
            import :: c_ptr, c_int
            type(c_ptr), value :: env
            type(c_ptr), intent(out) :: ctx
            integer(c_int) :: dtl_environment_make_world_context
        end function

        !> Create a world context with GPU from the environment
        function dtl_environment_make_world_context_gpu(env, device_id, ctx) &
                bind(c, name='dtl_environment_make_world_context_gpu')
            import :: c_ptr, c_int
            type(c_ptr), value :: env
            integer(c_int), value :: device_id
            type(c_ptr), intent(out) :: ctx
            integer(c_int) :: dtl_environment_make_world_context_gpu
        end function

        !> Create a CPU-only context from the environment
        function dtl_environment_make_cpu_context(env, ctx) &
                bind(c, name='dtl_environment_make_cpu_context')
            import :: c_ptr, c_int
            type(c_ptr), value :: env
            type(c_ptr), intent(out) :: ctx
            integer(c_int) :: dtl_environment_make_cpu_context
        end function

        ! ------------------------------------------------------------------
        ! Global Backend Queries
        ! ------------------------------------------------------------------

        function dtl_backend_name() bind(c, name='dtl_backend_name')
            import :: c_ptr
            type(c_ptr) :: dtl_backend_name
        end function

        function dtl_backend_count() bind(c, name='dtl_backend_count')
            import :: c_int
            integer(c_int) :: dtl_backend_count
        end function

        function dtl_version() bind(c, name='dtl_version')
            import :: c_ptr
            type(c_ptr) :: dtl_version
        end function

    end interface

end module dtl_environment
