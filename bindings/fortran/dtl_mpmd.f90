! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_mpmd.f90
!> @brief DTL MPMD Module - Role manager and intergroup communication
!> @since 0.1.0

module dtl_mpmd
    use, intrinsic :: iso_c_binding
    implicit none

    private

    ! Role manager lifecycle
    public :: dtl_role_manager_create, dtl_role_manager_destroy

    ! Role configuration
    public :: dtl_role_manager_add_role, dtl_role_manager_initialize

    ! Role queries
    public :: dtl_role_manager_has_role
    public :: dtl_role_manager_role_size, dtl_role_manager_role_rank

    ! Intergroup communication
    public :: dtl_intergroup_send, dtl_intergroup_recv

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        function dtl_role_manager_create(ctx, mgr) &
                bind(c, name='dtl_role_manager_create')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            type(c_ptr), intent(out) :: mgr
            integer(c_int) :: dtl_role_manager_create
        end function

        subroutine dtl_role_manager_destroy(mgr) &
                bind(c, name='dtl_role_manager_destroy')
            import :: c_ptr
            type(c_ptr), value :: mgr
        end subroutine

        function dtl_role_manager_add_role(mgr, role_name, num_ranks) &
                bind(c, name='dtl_role_manager_add_role')
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: mgr
            character(kind=c_char), intent(in) :: role_name(*)
            integer(c_int64_t), value :: num_ranks
            integer(c_int) :: dtl_role_manager_add_role
        end function

        function dtl_role_manager_initialize(mgr) &
                bind(c, name='dtl_role_manager_initialize')
            import :: c_ptr, c_int
            type(c_ptr), value :: mgr
            integer(c_int) :: dtl_role_manager_initialize
        end function

        function dtl_role_manager_has_role(mgr, role_name, has_role) &
                bind(c, name='dtl_role_manager_has_role')
            import :: c_ptr, c_int, c_char
            type(c_ptr), value :: mgr
            character(kind=c_char), intent(in) :: role_name(*)
            integer(c_int), intent(out) :: has_role
            integer(c_int) :: dtl_role_manager_has_role
        end function

        function dtl_role_manager_role_size(mgr, role_name, size) &
                bind(c, name='dtl_role_manager_role_size')
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: mgr
            character(kind=c_char), intent(in) :: role_name(*)
            integer(c_int64_t), intent(out) :: size
            integer(c_int) :: dtl_role_manager_role_size
        end function

        function dtl_role_manager_role_rank(mgr, role_name, rank) &
                bind(c, name='dtl_role_manager_role_rank')
            import :: c_ptr, c_int, c_char
            type(c_ptr), value :: mgr
            character(kind=c_char), intent(in) :: role_name(*)
            integer(c_int), intent(out) :: rank
            integer(c_int) :: dtl_role_manager_role_rank
        end function

        function dtl_intergroup_send(mgr, target_role, target_rank, &
                                     buf, count, dtype, tag) &
                bind(c, name='dtl_intergroup_send')
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: mgr
            character(kind=c_char), intent(in) :: target_role(*)
            integer(c_int), value :: target_rank
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: tag
            integer(c_int) :: dtl_intergroup_send
        end function

        function dtl_intergroup_recv(mgr, source_role, source_rank, &
                                     buf, count, dtype, tag) &
                bind(c, name='dtl_intergroup_recv')
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: mgr
            character(kind=c_char), intent(in) :: source_role(*)
            integer(c_int), value :: source_rank
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: tag
            integer(c_int) :: dtl_intergroup_recv
        end function

    end interface

end module dtl_mpmd
