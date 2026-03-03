// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_communicator.h
 * @brief DTL C bindings - Communicator operations
 * @since 0.1.0
 *
 * This header provides C bindings for point-to-point and collective
 * communication operations.
 */

#ifndef DTL_COMMUNICATOR_H
#define DTL_COMMUNICATOR_H

#include "dtl_types.h"
#include "dtl_status.h"
#include "dtl_context.h"

DTL_C_BEGIN

/* ==========================================================================
 * Point-to-Point Communication (Blocking)
 * ========================================================================== */

/**
 * @brief Send data to a specific rank (blocking)
 *
 * Sends count elements of the specified type to the destination rank.
 * This call blocks until the send buffer can be reused.
 *
 * @param ctx The context
 * @param buf Pointer to send buffer (must not be NULL)
 * @param count Number of elements to send
 * @param dtype Data type of elements
 * @param dest Destination rank
 * @param tag Message tag
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre buf must point to at least count * dtype_size(dtype) bytes
 * @pre dest must be a valid rank (0 to size-1)
 */
DTL_API dtl_status dtl_send(dtl_context_t ctx, const void* buf,
                            dtl_size_t count, dtl_dtype dtype,
                            dtl_rank_t dest, dtl_tag_t tag);

/**
 * @brief Receive data from a specific rank (blocking)
 *
 * Receives count elements of the specified type from the source rank.
 * This call blocks until the message is received.
 *
 * @param ctx The context
 * @param buf Pointer to receive buffer (must not be NULL)
 * @param count Number of elements to receive
 * @param dtype Data type of elements
 * @param source Source rank (or DTL_ANY_SOURCE)
 * @param tag Message tag (or DTL_ANY_TAG)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @pre ctx must be a valid context
 * @pre buf must point to at least count * dtype_size(dtype) bytes
 * @pre source must be a valid rank or DTL_ANY_SOURCE
 */
DTL_API dtl_status dtl_recv(dtl_context_t ctx, void* buf,
                            dtl_size_t count, dtl_dtype dtype,
                            dtl_rank_t source, dtl_tag_t tag);

/**
 * @brief Send and receive data simultaneously (blocking)
 *
 * Performs a combined send and receive operation. Useful for
 * exchanging data between pairs of ranks.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer
 * @param sendcount Number of elements to send
 * @param senddtype Data type of send elements
 * @param dest Destination rank
 * @param sendtag Send message tag
 * @param recvbuf Pointer to receive buffer
 * @param recvcount Number of elements to receive
 * @param recvdtype Data type of receive elements
 * @param source Source rank (or DTL_ANY_SOURCE)
 * @param recvtag Receive message tag (or DTL_ANY_TAG)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_sendrecv(dtl_context_t ctx,
                                 const void* sendbuf, dtl_size_t sendcount,
                                 dtl_dtype senddtype, dtl_rank_t dest,
                                 dtl_tag_t sendtag,
                                 void* recvbuf, dtl_size_t recvcount,
                                 dtl_dtype recvdtype, dtl_rank_t source,
                                 dtl_tag_t recvtag);

/* ==========================================================================
 * Point-to-Point Communication (Non-blocking)
 * ========================================================================== */

/**
 * @brief Send data asynchronously (non-blocking)
 *
 * Initiates a non-blocking send. The operation must be completed
 * with dtl_wait() before the send buffer can be modified.
 *
 * @param ctx The context
 * @param buf Pointer to send buffer
 * @param count Number of elements to send
 * @param dtype Data type of elements
 * @param dest Destination rank
 * @param tag Message tag
 * @param[out] request Pointer to receive the request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note The send buffer must not be modified until dtl_wait() completes.
 */
DTL_API dtl_status dtl_isend(dtl_context_t ctx, const void* buf,
                              dtl_size_t count, dtl_dtype dtype,
                              dtl_rank_t dest, dtl_tag_t tag,
                              dtl_request_t* request);

/**
 * @brief Receive data asynchronously (non-blocking)
 *
 * Initiates a non-blocking receive. The operation must be completed
 * with dtl_wait() before reading the receive buffer.
 *
 * @param ctx The context
 * @param buf Pointer to receive buffer
 * @param count Number of elements to receive
 * @param dtype Data type of elements
 * @param source Source rank (or DTL_ANY_SOURCE)
 * @param tag Message tag (or DTL_ANY_TAG)
 * @param[out] request Pointer to receive the request handle
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note The receive buffer must not be read until dtl_wait() completes.
 */
DTL_API dtl_status dtl_irecv(dtl_context_t ctx, void* buf,
                              dtl_size_t count, dtl_dtype dtype,
                              dtl_rank_t source, dtl_tag_t tag,
                              dtl_request_t* request);

/**
 * @brief Wait for a non-blocking operation to complete
 *
 * Blocks until the specified non-blocking operation completes.
 * After completion, the request handle is invalidated.
 *
 * @param request The request to wait on
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @post request is invalidated and must not be reused
 */
DTL_API dtl_status dtl_wait(dtl_request_t request);

/**
 * @brief Wait for all requests to complete
 *
 * Blocks until all specified non-blocking operations complete.
 * After completion, all request handles are invalidated.
 *
 * @param count Number of requests
 * @param requests Array of requests to wait on
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @post All requests are invalidated
 */
DTL_API dtl_status dtl_waitall(dtl_size_t count, dtl_request_t* requests);

/**
 * @brief Test if a non-blocking operation has completed
 *
 * Non-blocking test - returns immediately.
 *
 * @param request The request to test
 * @param[out] completed Set to 1 if complete, 0 if still pending
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note If completed, the request is invalidated.
 */
DTL_API dtl_status dtl_test(dtl_request_t request, int* completed);

/**
 * @brief Free a request handle without waiting
 *
 * Cancels the request if possible and frees associated resources.
 * Use this to clean up requests that are no longer needed.
 *
 * @param request The request to free (may be NULL)
 */
DTL_API void dtl_request_free(dtl_request_t request);

/* ==========================================================================
 * Collective Communication - Synchronization
 * ========================================================================== */

/**
 * @brief Barrier synchronization
 *
 * Blocks until all ranks have called this function.
 *
 * @param ctx The context
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note All ranks must call this function.
 */
DTL_API dtl_status dtl_barrier(dtl_context_t ctx);

/* ==========================================================================
 * Collective Communication - Broadcast
 * ========================================================================== */

/**
 * @brief Broadcast data from root to all ranks
 *
 * The root rank sends its buffer to all other ranks.
 * All ranks receive the data in their buffer.
 *
 * @param ctx The context
 * @param buf Pointer to buffer (send on root, receive on others)
 * @param count Number of elements
 * @param dtype Data type of elements
 * @param root Root rank that broadcasts
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note All ranks must call this with the same count, dtype, and root.
 */
DTL_API dtl_status dtl_broadcast(dtl_context_t ctx, void* buf,
                                  dtl_size_t count, dtl_dtype dtype,
                                  dtl_rank_t root);

/* ==========================================================================
 * Collective Communication - Reduction
 * ========================================================================== */

/**
 * @brief Reduce data from all ranks to root
 *
 * Combines elements from all ranks using the specified operation
 * and places the result on the root rank.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer
 * @param recvbuf Pointer to receive buffer (significant only on root)
 * @param count Number of elements
 * @param dtype Data type of elements
 * @param op Reduction operation
 * @param root Root rank that receives the result
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note sendbuf and recvbuf must not overlap (except on root if in-place)
 * @note All ranks must call this with identical parameters.
 */
DTL_API dtl_status dtl_reduce(dtl_context_t ctx,
                               const void* sendbuf, void* recvbuf,
                               dtl_size_t count, dtl_dtype dtype,
                               dtl_reduce_op op, dtl_rank_t root);

/**
 * @brief Reduce data from all ranks to all ranks
 *
 * Combines elements from all ranks using the specified operation
 * and distributes the result to all ranks.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer
 * @param recvbuf Pointer to receive buffer
 * @param count Number of elements
 * @param dtype Data type of elements
 * @param op Reduction operation
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note sendbuf and recvbuf must not overlap
 * @note All ranks must call this with identical parameters.
 */
DTL_API dtl_status dtl_allreduce(dtl_context_t ctx,
                                  const void* sendbuf, void* recvbuf,
                                  dtl_size_t count, dtl_dtype dtype,
                                  dtl_reduce_op op);

/* ==========================================================================
 * Collective Communication - Gather/Scatter
 * ========================================================================== */

/**
 * @brief Gather data from all ranks to root
 *
 * Each rank sends sendcount elements to root. Root receives
 * all elements concatenated in rank order.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer
 * @param sendcount Number of elements to send
 * @param senddtype Data type of send elements
 * @param recvbuf Pointer to receive buffer (significant only on root)
 * @param recvcount Number of elements to receive from each rank
 * @param recvdtype Data type of receive elements
 * @param root Root rank that gathers
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note recvbuf must have space for size * recvcount elements on root.
 */
DTL_API dtl_status dtl_gather(dtl_context_t ctx,
                               const void* sendbuf, dtl_size_t sendcount,
                               dtl_dtype senddtype,
                               void* recvbuf, dtl_size_t recvcount,
                               dtl_dtype recvdtype, dtl_rank_t root);

/**
 * @brief Gather data from all ranks to all ranks
 *
 * Like dtl_gather, but all ranks receive the gathered data.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer
 * @param sendcount Number of elements to send
 * @param senddtype Data type of send elements
 * @param recvbuf Pointer to receive buffer
 * @param recvcount Number of elements to receive from each rank
 * @param recvdtype Data type of receive elements
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_allgather(dtl_context_t ctx,
                                  const void* sendbuf, dtl_size_t sendcount,
                                  dtl_dtype senddtype,
                                  void* recvbuf, dtl_size_t recvcount,
                                  dtl_dtype recvdtype);

/**
 * @brief Scatter data from root to all ranks
 *
 * Root sends different portions of its buffer to each rank.
 * Each rank receives recvcount elements.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer (significant only on root)
 * @param sendcount Number of elements to send to each rank
 * @param senddtype Data type of send elements
 * @param recvbuf Pointer to receive buffer
 * @param recvcount Number of elements to receive
 * @param recvdtype Data type of receive elements
 * @param root Root rank that scatters
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note sendbuf must have space for size * sendcount elements on root.
 */
DTL_API dtl_status dtl_scatter(dtl_context_t ctx,
                                const void* sendbuf, dtl_size_t sendcount,
                                dtl_dtype senddtype,
                                void* recvbuf, dtl_size_t recvcount,
                                dtl_dtype recvdtype, dtl_rank_t root);

/**
 * @brief All-to-all communication
 *
 * Each rank sends distinct data to every other rank.
 * Rank i sends sendcount elements to rank j and receives
 * recvcount elements from rank j.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer
 * @param sendcount Number of elements to send to each rank
 * @param senddtype Data type of send elements
 * @param recvbuf Pointer to receive buffer
 * @param recvcount Number of elements to receive from each rank
 * @param recvdtype Data type of receive elements
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_alltoall(dtl_context_t ctx,
                                 const void* sendbuf, dtl_size_t sendcount,
                                 dtl_dtype senddtype,
                                 void* recvbuf, dtl_size_t recvcount,
                                 dtl_dtype recvdtype);

/* ==========================================================================
 * Variable-Count Collectives
 * ========================================================================== */

/**
 * @brief Gather variable amounts of data from all ranks to root
 *
 * Like dtl_gather, but each rank can send a different number of elements.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer
 * @param sendcount Number of elements to send from this rank
 * @param senddtype Data type of send elements
 * @param recvbuf Pointer to receive buffer (significant only on root)
 * @param recvcounts Array of receive counts (one per rank, root only)
 * @param displs Array of displacements in recvbuf (one per rank, root only)
 * @param recvdtype Data type of receive elements
 * @param root Root rank that gathers
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_gatherv(dtl_context_t ctx,
                                const void* sendbuf, dtl_size_t sendcount,
                                dtl_dtype senddtype,
                                void* recvbuf, const dtl_size_t* recvcounts,
                                const dtl_size_t* displs, dtl_dtype recvdtype,
                                dtl_rank_t root);

/**
 * @brief Scatter variable amounts of data from root to all ranks
 *
 * Like dtl_scatter, but root can send different amounts to each rank.
 *
 * @param ctx The context
 * @param sendbuf Pointer to send buffer (significant only on root)
 * @param sendcounts Array of send counts (one per rank, root only)
 * @param displs Array of displacements in sendbuf (one per rank, root only)
 * @param senddtype Data type of send elements
 * @param recvbuf Pointer to receive buffer
 * @param recvcount Number of elements to receive
 * @param recvdtype Data type of receive elements
 * @param root Root rank that scatters
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_scatterv(dtl_context_t ctx,
                                 const void* sendbuf,
                                 const dtl_size_t* sendcounts,
                                 const dtl_size_t* displs, dtl_dtype senddtype,
                                 void* recvbuf, dtl_size_t recvcount,
                                 dtl_dtype recvdtype, dtl_rank_t root);

/* ==========================================================================
 * Variable-Size Collectives (Phase 12.5)
 * ========================================================================== */

/**
 * @brief All-gather with variable counts
 *
 * Each rank contributes a potentially different number of elements.
 * All ranks receive all data.
 *
 * @param ctx The DTL context
 * @param sendbuf Send buffer
 * @param sendcount Number of elements to send from this rank
 * @param dtype Data type
 * @param recvbuf Receive buffer (must be large enough for all data)
 * @param recvcounts Array of receive counts (one per rank)
 * @param displs Array of displacements in recvbuf (one per rank)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_allgatherv(dtl_context_t ctx,
                                    const void* sendbuf,
                                    dtl_size_t sendcount, dtl_dtype dtype,
                                    void* recvbuf,
                                    const dtl_size_t* recvcounts,
                                    const dtl_size_t* displs);

/**
 * @brief All-to-all with variable counts
 *
 * Each rank sends a potentially different amount of data to every other rank.
 *
 * @param ctx The DTL context
 * @param sendbuf Send buffer
 * @param sendcounts Array of send counts (one per rank)
 * @param sdispls Array of send displacements (one per rank)
 * @param senddtype Send data type
 * @param recvbuf Receive buffer
 * @param recvcounts Array of receive counts (one per rank)
 * @param rdispls Array of receive displacements (one per rank)
 * @param recvdtype Receive data type
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_alltoallv(dtl_context_t ctx,
                                   const void* sendbuf,
                                   const dtl_size_t* sendcounts,
                                   const dtl_size_t* sdispls,
                                   dtl_dtype senddtype,
                                   void* recvbuf,
                                   const dtl_size_t* recvcounts,
                                   const dtl_size_t* rdispls,
                                   dtl_dtype recvdtype);

/* ==========================================================================
 * Scan / Prefix Operations (Phase 16)
 * ========================================================================== */

/**
 * @brief Inclusive scan (prefix reduction) across ranks
 *
 * Performs an inclusive prefix reduction. On rank i, the result contains
 * the reduction of values from ranks 0..i (inclusive).
 *
 * @param ctx The DTL context
 * @param sendbuf Send buffer
 * @param recvbuf Receive buffer
 * @param count Number of elements
 * @param dtype Data type
 * @param op Reduction operation
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note All ranks must call this with identical count, dtype, and op.
 */
DTL_API dtl_status dtl_scan(dtl_context_t ctx,
                             const void* sendbuf, void* recvbuf,
                             dtl_size_t count, dtl_dtype dtype,
                             dtl_reduce_op op);

/**
 * @brief Exclusive scan (prefix reduction) across ranks
 *
 * Performs an exclusive prefix reduction. On rank i, the result contains
 * the reduction of values from ranks 0..i-1. Rank 0 receives the
 * identity element for the operation.
 *
 * @param ctx The DTL context
 * @param sendbuf Send buffer
 * @param recvbuf Receive buffer (undefined on rank 0 for MPI_Exscan)
 * @param count Number of elements
 * @param dtype Data type
 * @param op Reduction operation
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note All ranks must call this with identical count, dtype, and op.
 * @note On rank 0, recvbuf is filled with the identity for the operation.
 */
DTL_API dtl_status dtl_exscan(dtl_context_t ctx,
                               const void* sendbuf, void* recvbuf,
                               dtl_size_t count, dtl_dtype dtype,
                               dtl_reduce_op op);

/* ==========================================================================
 * Probe Operations (Phase 16)
 * ========================================================================== */

/**
 * @brief Message info structure returned by probe operations
 */
typedef struct dtl_message_info_s {
    dtl_rank_t source;   /**< Source rank of the message */
    dtl_tag_t  tag;      /**< Tag of the message */
    dtl_size_t count;    /**< Number of elements in the message */
} dtl_message_info_t;

/**
 * @brief Probe for an incoming message (blocking)
 *
 * Blocks until a matching message is available. Does not receive the
 * message — use dtl_recv() after probing. The message info is populated
 * with the source, tag, and size of the pending message.
 *
 * @param ctx The context
 * @param source Source rank to match (or DTL_ANY_SOURCE)
 * @param tag Tag to match (or DTL_ANY_TAG)
 * @param dtype Expected data type (used to compute count from byte size)
 * @param[out] info Pointer to receive message info (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 *
 * @note This is a blocking call.
 */
DTL_API dtl_status dtl_probe(dtl_context_t ctx,
                              dtl_rank_t source, dtl_tag_t tag,
                              dtl_dtype dtype,
                              dtl_message_info_t* info);

/**
 * @brief Probe for an incoming message (non-blocking)
 *
 * Checks if a matching message is available without blocking.
 * If a message is available, sets flag to 1 and populates info.
 * If no message is available, sets flag to 0.
 *
 * @param ctx The context
 * @param source Source rank to match (or DTL_ANY_SOURCE)
 * @param tag Tag to match (or DTL_ANY_TAG)
 * @param dtype Expected data type (used to compute count)
 * @param[out] flag Set to 1 if message available, 0 otherwise
 * @param[out] info Pointer to receive message info (may be NULL)
 * @return DTL_SUCCESS on success, error code otherwise
 */
DTL_API dtl_status dtl_iprobe(dtl_context_t ctx,
                               dtl_rank_t source, dtl_tag_t tag,
                               dtl_dtype dtype,
                               int* flag, dtl_message_info_t* info);

DTL_C_END

/* Mark header as available for master include */
#define DTL_COMMUNICATOR_H_AVAILABLE

#endif /* DTL_COMMUNICATOR_H */
