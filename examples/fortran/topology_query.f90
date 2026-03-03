! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file topology_query.f90
!> @brief Hardware topology queries with DTL Fortran bindings
!>
!> Demonstrates:
!> - dtl_topology_num_cpus / dtl_topology_num_gpus
!> - dtl_topology_node_id for locality detection
!> - dtl_topology_is_local for co-location checking
!>
!> Run:
!>   mpirun -np 2 ./fortran_topology_query

program topology_query
    use, intrinsic :: iso_c_binding
    use dtl_fortran
    implicit none

    type(c_ptr) :: ctx
    integer(c_int32_t) :: status
    integer(c_int32_t) :: rank, num_ranks
    integer(c_int) :: num_cpus, num_gpus, cpu_id, gpu_id, node_id, is_local
    integer(c_int32_t) :: r

    ! Create context
    status = dtl_context_create_default(ctx)
    if (.not. is_success(status)) stop 1

    rank = dtl_context_rank(ctx)
    num_ranks = dtl_context_size(ctx)

    if (rank == 0) then
        print '(A)', 'DTL Topology Query (Fortran)'
        print '(A)', '=============================='
        print '(A,I0,A)', 'Running with ', num_ranks, ' ranks'
        print '(A)', ''
    end if
    status = dtl_barrier(ctx)

    ! Query CPU count
    status = dtl_topology_num_cpus(num_cpus)
    if (is_success(status)) then
        print '(A,I0,A,I0,A)', 'Rank ', rank, ': ', num_cpus, ' CPUs'
    else
        print '(A,I0,A)', 'Rank ', rank, ': CPU query failed'
    end if

    ! Query GPU count
    status = dtl_topology_num_gpus(num_gpus)
    if (is_success(status)) then
        print '(A,I0,A,I0,A)', 'Rank ', rank, ': ', num_gpus, ' GPUs'
    else
        print '(A,I0,A)', 'Rank ', rank, ': GPU query failed'
    end if

    ! CPU affinity
    status = dtl_topology_cpu_affinity(rank, cpu_id)
    if (is_success(status)) then
        print '(A,I0,A,I0)', 'Rank ', rank, ': CPU affinity = ', cpu_id
    end if

    ! GPU ID (if available)
    if (num_gpus > 0) then
        status = dtl_topology_gpu_id(rank, gpu_id)
        if (is_success(status)) then
            print '(A,I0,A,I0)', 'Rank ', rank, ': GPU ID = ', gpu_id
        end if
    end if

    ! Node ID
    status = dtl_topology_node_id(rank, node_id)
    if (is_success(status)) then
        print '(A,I0,A,I0)', 'Rank ', rank, ': node_id = ', node_id
    end if

    status = dtl_barrier(ctx)

    ! Locality checks from rank 0
    if (rank == 0 .and. num_ranks > 1) then
        print '(A)', ''
        print '(A)', 'Locality checks:'
        do r = 1, num_ranks - 1
            status = dtl_topology_is_local(0, r, is_local)
            if (is_success(status)) then
                if (is_local /= 0) then
                    print '(A,I0,A)', '  Rank 0 & Rank ', r, ': same node'
                else
                    print '(A,I0,A)', '  Rank 0 & Rank ', r, ': different nodes'
                end if
            end if
        end do
    end if
    status = dtl_barrier(ctx)

    call dtl_context_destroy(ctx)

    if (rank == 0) then
        print '(A)', ''
        print '(A)', 'Done!'
    end if

end program topology_query
