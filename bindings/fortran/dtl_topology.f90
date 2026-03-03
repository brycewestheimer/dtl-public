! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_topology.f90
!> @brief DTL Topology Module - Hardware topology queries
!> @since 0.1.0

module dtl_topology
    use, intrinsic :: iso_c_binding
    implicit none

    private

    public :: dtl_topology_num_cpus, dtl_topology_cpu_affinity
    public :: dtl_topology_num_gpus, dtl_topology_gpu_id
    public :: dtl_topology_is_local, dtl_topology_node_id

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        function dtl_topology_num_cpus(count) &
                bind(c, name='dtl_topology_num_cpus')
            import :: c_int
            integer(c_int), intent(out) :: count
            integer(c_int) :: dtl_topology_num_cpus
        end function

        function dtl_topology_cpu_affinity(rank, cpu_id) &
                bind(c, name='dtl_topology_cpu_affinity')
            import :: c_int
            integer(c_int), value :: rank
            integer(c_int), intent(out) :: cpu_id
            integer(c_int) :: dtl_topology_cpu_affinity
        end function

        function dtl_topology_num_gpus(count) &
                bind(c, name='dtl_topology_num_gpus')
            import :: c_int
            integer(c_int), intent(out) :: count
            integer(c_int) :: dtl_topology_num_gpus
        end function

        function dtl_topology_gpu_id(rank, gpu_id) &
                bind(c, name='dtl_topology_gpu_id')
            import :: c_int
            integer(c_int), value :: rank
            integer(c_int), intent(out) :: gpu_id
            integer(c_int) :: dtl_topology_gpu_id
        end function

        function dtl_topology_is_local(rank_a, rank_b, is_local) &
                bind(c, name='dtl_topology_is_local')
            import :: c_int
            integer(c_int), value :: rank_a
            integer(c_int), value :: rank_b
            integer(c_int), intent(out) :: is_local
            integer(c_int) :: dtl_topology_is_local
        end function

        function dtl_topology_node_id(rank, node_id) &
                bind(c, name='dtl_topology_node_id')
            import :: c_int
            integer(c_int), value :: rank
            integer(c_int), intent(out) :: node_id
            integer(c_int) :: dtl_topology_node_id
        end function

    end interface

end module dtl_topology
