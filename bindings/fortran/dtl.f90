! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl.f90
!> @brief DTL Parent Module - Re-exports all submodules
!> @since 0.1.0
!>
!> This module provides native Fortran bindings to the DTL (Distributed Template
!> Library) C API. Users can import this module directly without writing manual
!> interface blocks. All submodules are re-exported so `use dtl` gives access
!> to the entire API.
!>
!> @code
!> program example
!>     use dtl
!>     implicit none
!>
!>     type(c_ptr) :: ctx
!>     integer(c_int) :: status
!>
!>     status = dtl_context_create_default(ctx)
!>     if (status /= DTL_SUCCESS) stop 'Failed to create context'
!>
!>     print *, 'Rank', dtl_context_rank(ctx), 'of', dtl_context_size(ctx)
!>
!>     call dtl_context_destroy(ctx)
!> end program
!> @endcode

module dtl
    use dtl_core
    use dtl_environment
    use dtl_context
    use dtl_containers
    use dtl_communication
    use dtl_algorithms
    use dtl_helpers
    use dtl_rma
    use dtl_mpmd
    use dtl_futures
    use dtl_policies
    use dtl_topology
    implicit none
end module dtl
