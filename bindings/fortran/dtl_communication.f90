! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_communication.f90
!> @brief DTL Communication Module - P2P, collectives, probing
!> @since 0.1.0

module dtl_communication
    use, intrinsic :: iso_c_binding
    use dtl_core, only: dtl_message_info
    implicit none

    private

    ! Barrier
    public :: dtl_barrier

    ! Broadcast / Allreduce / Reduce
    public :: dtl_broadcast, dtl_allreduce, dtl_reduce

    ! P2P blocking
    public :: dtl_send, dtl_recv, dtl_sendrecv

    ! P2P non-blocking
    public :: dtl_isend, dtl_irecv

    ! Request management
    public :: dtl_wait, dtl_waitall, dtl_test, dtl_request_free

    ! Probe
    public :: dtl_probe, dtl_iprobe

    ! Collectives
    public :: dtl_gather, dtl_scatter, dtl_allgather, dtl_alltoall

    ! Variable-count collectives
    public :: dtl_gatherv, dtl_scatterv, dtl_allgatherv, dtl_alltoallv

    ! Scan/prefix
    public :: dtl_scan, dtl_exscan

    ! Explicit NCCL device collectives
    public :: dtl_nccl_allreduce_device, dtl_nccl_allreduce_device_ex
    public :: dtl_nccl_broadcast_device, dtl_nccl_broadcast_device_ex
    public :: dtl_nccl_barrier_device
    public :: dtl_nccl_gatherv_device_ex, dtl_nccl_scatterv_device_ex
    public :: dtl_nccl_allgatherv_device_ex, dtl_nccl_alltoallv_device_ex
    public :: dtl_nccl_scan_device_ex, dtl_nccl_exscan_device_ex

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        ! ------------------------------------------------------------------
        ! Barrier
        ! ------------------------------------------------------------------

        function dtl_barrier(ctx) bind(c, name='dtl_barrier')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_barrier
        end function

        ! ------------------------------------------------------------------
        ! Broadcast / Allreduce / Reduce
        ! ------------------------------------------------------------------

        function dtl_broadcast(ctx, buf, count, dtype, root) &
                bind(c, name='dtl_broadcast')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_broadcast
        end function

        function dtl_allreduce(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_allreduce')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_allreduce
        end function

        function dtl_reduce(ctx, sendbuf, recvbuf, count, dtype, op, root) &
                bind(c, name='dtl_reduce')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int), value :: root
            integer(c_int) :: dtl_reduce
        end function

        ! ------------------------------------------------------------------
        ! P2P Blocking
        ! ------------------------------------------------------------------

        function dtl_send(ctx, buf, count, dtype, dest, tag) &
                bind(c, name='dtl_send')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: dest
            integer(c_int), value :: tag
            integer(c_int) :: dtl_send
        end function

        function dtl_recv(ctx, buf, count, dtype, source, tag) &
                bind(c, name='dtl_recv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: source
            integer(c_int), value :: tag
            integer(c_int) :: dtl_recv
        end function

        function dtl_sendrecv(ctx, sendbuf, sendcount, senddtype, dest, &
                              sendtag, recvbuf, recvcount, recvdtype, &
                              source, recvtag) &
                bind(c, name='dtl_sendrecv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            integer(c_int), value :: dest
            integer(c_int), value :: sendtag
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int), value :: source
            integer(c_int), value :: recvtag
            integer(c_int) :: dtl_sendrecv
        end function

        ! ------------------------------------------------------------------
        ! P2P Non-blocking
        ! ------------------------------------------------------------------

        function dtl_isend(ctx, buf, count, dtype, dest, tag, request) &
                bind(c, name='dtl_isend')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: dest
            integer(c_int), value :: tag
            type(c_ptr), intent(out) :: request
            integer(c_int) :: dtl_isend
        end function

        function dtl_irecv(ctx, buf, count, dtype, source, tag, request) &
                bind(c, name='dtl_irecv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: source
            integer(c_int), value :: tag
            type(c_ptr), intent(out) :: request
            integer(c_int) :: dtl_irecv
        end function

        ! ------------------------------------------------------------------
        ! Request Management
        ! ------------------------------------------------------------------

        function dtl_wait(request) bind(c, name='dtl_wait')
            import :: c_ptr, c_int
            type(c_ptr), value :: request
            integer(c_int) :: dtl_wait
        end function

        function dtl_waitall(count, requests) bind(c, name='dtl_waitall')
            import :: c_ptr, c_int, c_int64_t
            integer(c_int64_t), value :: count
            type(c_ptr), intent(inout) :: requests(*)
            integer(c_int) :: dtl_waitall
        end function

        function dtl_test(request, completed) bind(c, name='dtl_test')
            import :: c_ptr, c_int
            type(c_ptr), value :: request
            integer(c_int), intent(out) :: completed
            integer(c_int) :: dtl_test
        end function

        subroutine dtl_request_free(request) &
                bind(c, name='dtl_request_free')
            import :: c_ptr
            type(c_ptr), value :: request
        end subroutine

        ! ------------------------------------------------------------------
        ! Probe
        ! ------------------------------------------------------------------

        function dtl_probe(ctx, source, tag, dtype, info) &
                bind(c, name='dtl_probe')
            import :: c_ptr, c_int, dtl_message_info
            type(c_ptr), value :: ctx
            integer(c_int), value :: source
            integer(c_int), value :: tag
            integer(c_int), value :: dtype
            type(dtl_message_info), intent(out) :: info
            integer(c_int) :: dtl_probe
        end function

        function dtl_iprobe(ctx, source, tag, dtype, flag, info) &
                bind(c, name='dtl_iprobe')
            import :: c_ptr, c_int, dtl_message_info
            type(c_ptr), value :: ctx
            integer(c_int), value :: source
            integer(c_int), value :: tag
            integer(c_int), value :: dtype
            integer(c_int), intent(out) :: flag
            type(dtl_message_info), intent(out) :: info
            integer(c_int) :: dtl_iprobe
        end function

        ! ------------------------------------------------------------------
        ! Collectives
        ! ------------------------------------------------------------------

        function dtl_gather(ctx, sendbuf, sendcount, senddtype, &
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
            integer(c_int) :: dtl_gather
        end function

        function dtl_scatter(ctx, sendbuf, sendcount, senddtype, &
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
            integer(c_int) :: dtl_scatter
        end function

        function dtl_allgather(ctx, sendbuf, sendcount, senddtype, &
                               recvbuf, recvcount, recvdtype) &
                bind(c, name='dtl_allgather')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int) :: dtl_allgather
        end function

        function dtl_alltoall(ctx, sendbuf, sendcount, senddtype, &
                              recvbuf, recvcount, recvdtype) &
                bind(c, name='dtl_alltoall')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int) :: dtl_alltoall
        end function

        ! ------------------------------------------------------------------
        ! Variable-Count Collectives
        ! ------------------------------------------------------------------

        function dtl_gatherv(ctx, sendbuf, sendcount, senddtype, &
                             recvbuf, recvcounts, displs, recvdtype, root) &
                bind(c, name='dtl_gatherv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), intent(in) :: recvcounts(*)
            integer(c_int64_t), intent(in) :: displs(*)
            integer(c_int), value :: recvdtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_gatherv
        end function

        function dtl_scatterv(ctx, sendbuf, sendcounts, displs, senddtype, &
                              recvbuf, recvcount, recvdtype, root) &
                bind(c, name='dtl_scatterv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), intent(in) :: sendcounts(*)
            integer(c_int64_t), intent(in) :: displs(*)
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_scatterv
        end function

        function dtl_allgatherv(ctx, sendbuf, sendcount, dtype, &
                                recvbuf, recvcounts, displs) &
                bind(c, name='dtl_allgatherv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: dtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), intent(in) :: recvcounts(*)
            integer(c_int64_t), intent(in) :: displs(*)
            integer(c_int) :: dtl_allgatherv
        end function

        function dtl_alltoallv(ctx, sendbuf, sendcounts, sdispls, senddtype, &
                               recvbuf, recvcounts, rdispls, recvdtype) &
                bind(c, name='dtl_alltoallv')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), intent(in) :: sendcounts(*)
            integer(c_int64_t), intent(in) :: sdispls(*)
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), intent(in) :: recvcounts(*)
            integer(c_int64_t), intent(in) :: rdispls(*)
            integer(c_int), value :: recvdtype
            integer(c_int) :: dtl_alltoallv
        end function

        ! ------------------------------------------------------------------
        ! Scan / Prefix
        ! ------------------------------------------------------------------

        function dtl_scan(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_scan')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_scan
        end function

        function dtl_exscan(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_exscan')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_exscan
        end function

        ! ------------------------------------------------------------------
        ! Explicit NCCL Device Collectives
        ! ------------------------------------------------------------------

        function dtl_nccl_allreduce_device(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_nccl_allreduce_device')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_nccl_allreduce_device
        end function

        function dtl_nccl_allreduce_device_ex(ctx, sendbuf, recvbuf, count, dtype, op, mode) &
                bind(c, name='dtl_nccl_allreduce_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_allreduce_device_ex
        end function

        function dtl_nccl_broadcast_device(ctx, buf, count, dtype, root) &
                bind(c, name='dtl_nccl_broadcast_device')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_nccl_broadcast_device
        end function

        function dtl_nccl_broadcast_device_ex(ctx, buf, count, dtype, root, mode) &
                bind(c, name='dtl_nccl_broadcast_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: root
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_broadcast_device_ex
        end function

        function dtl_nccl_barrier_device(ctx) &
                bind(c, name='dtl_nccl_barrier_device')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_nccl_barrier_device
        end function

        function dtl_nccl_gatherv_device_ex(ctx, sendbuf, sendcount, senddtype, &
                                            recvbuf, recvcounts, displs, recvdtype, &
                                            root, mode) &
                bind(c, name='dtl_nccl_gatherv_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), intent(in) :: recvcounts(*)
            integer(c_int64_t), intent(in) :: displs(*)
            integer(c_int), value :: recvdtype
            integer(c_int), value :: root
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_gatherv_device_ex
        end function

        function dtl_nccl_scatterv_device_ex(ctx, sendbuf, sendcounts, displs, senddtype, &
                                             recvbuf, recvcount, recvdtype, root, mode) &
                bind(c, name='dtl_nccl_scatterv_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), intent(in) :: sendcounts(*)
            integer(c_int64_t), intent(in) :: displs(*)
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: recvcount
            integer(c_int), value :: recvdtype
            integer(c_int), value :: root
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_scatterv_device_ex
        end function

        function dtl_nccl_allgatherv_device_ex(ctx, sendbuf, sendcount, dtype, &
                                               recvbuf, recvcounts, displs, mode) &
                bind(c, name='dtl_nccl_allgatherv_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), value :: sendcount
            integer(c_int), value :: dtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), intent(in) :: recvcounts(*)
            integer(c_int64_t), intent(in) :: displs(*)
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_allgatherv_device_ex
        end function

        function dtl_nccl_alltoallv_device_ex(ctx, sendbuf, sendcounts, sdispls, senddtype, &
                                              recvbuf, recvcounts, rdispls, recvdtype, mode) &
                bind(c, name='dtl_nccl_alltoallv_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            integer(c_int64_t), intent(in) :: sendcounts(*)
            integer(c_int64_t), intent(in) :: sdispls(*)
            integer(c_int), value :: senddtype
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), intent(in) :: recvcounts(*)
            integer(c_int64_t), intent(in) :: rdispls(*)
            integer(c_int), value :: recvdtype
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_alltoallv_device_ex
        end function

        function dtl_nccl_scan_device_ex(ctx, sendbuf, recvbuf, count, dtype, op, mode) &
                bind(c, name='dtl_nccl_scan_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_scan_device_ex
        end function

        function dtl_nccl_exscan_device_ex(ctx, sendbuf, recvbuf, count, dtype, op, mode) &
                bind(c, name='dtl_nccl_exscan_device_ex')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int), value :: mode
            integer(c_int) :: dtl_nccl_exscan_device_ex
        end function

    end interface

end module dtl_communication
