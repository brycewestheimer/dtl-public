// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_communicator.cpp
 * @brief DTL C bindings - Communicator implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_communicator.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/futures/progress.hpp>

#include "dtl_internal.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

// ============================================================================
// MPI Datatype Mapping
// ============================================================================

#ifdef DTL_HAS_MPI

static MPI_Datatype dtype_to_mpi(dtl_dtype dtype) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    return MPI_INT8_T;
        case DTL_DTYPE_INT16:   return MPI_INT16_T;
        case DTL_DTYPE_INT32:   return MPI_INT32_T;
        case DTL_DTYPE_INT64:   return MPI_INT64_T;
        case DTL_DTYPE_UINT8:   return MPI_UINT8_T;
        case DTL_DTYPE_UINT16:  return MPI_UINT16_T;
        case DTL_DTYPE_UINT32:  return MPI_UINT32_T;
        case DTL_DTYPE_UINT64:  return MPI_UINT64_T;
        case DTL_DTYPE_FLOAT32: return MPI_FLOAT;
        case DTL_DTYPE_FLOAT64: return MPI_DOUBLE;
        case DTL_DTYPE_BYTE:    return MPI_BYTE;
        case DTL_DTYPE_BOOL:    return MPI_UINT8_T;
        default:                return MPI_DATATYPE_NULL;
    }
}

static MPI_Op reduce_op_to_mpi(dtl_reduce_op op) {
    switch (op) {
        case DTL_OP_SUM:    return MPI_SUM;
        case DTL_OP_PROD:   return MPI_PROD;
        case DTL_OP_MIN:    return MPI_MIN;
        case DTL_OP_MAX:    return MPI_MAX;
        case DTL_OP_LAND:   return MPI_LAND;
        case DTL_OP_LOR:    return MPI_LOR;
        case DTL_OP_BAND:   return MPI_BAND;
        case DTL_OP_BOR:    return MPI_BOR;
        case DTL_OP_LXOR:   return MPI_LXOR;
        case DTL_OP_BXOR:   return MPI_BXOR;
        case DTL_OP_MINLOC: return MPI_MINLOC;
        case DTL_OP_MAXLOC: return MPI_MAXLOC;
        default:            return MPI_OP_NULL;
    }
}

static dtl_status to_mpi_count(dtl_size_t count, int* out) {
    if (!checked_size_to_int(count, out)) {
        return DTL_ERROR_OUT_OF_RANGE;
    }
    return DTL_SUCCESS;
}

static dtl_status convert_size_array_to_int(const dtl_size_t* in,
                                            int n,
                                            int* out) {
    if (!in || !out) {
        return DTL_ERROR_NULL_POINTER;
    }
    for (int i = 0; i < n; ++i) {
        if (!checked_size_to_int(in[i], &out[i])) {
            return DTL_ERROR_OUT_OF_RANGE;
        }
    }
    return DTL_SUCCESS;
}

#endif  // DTL_HAS_MPI

// Validation helpers defined in dtl_internal.hpp

// ============================================================================
// Point-to-Point (Blocking)
// ============================================================================

extern "C" {

dtl_status dtl_send(dtl_context_t ctx, const void* buf,
                    dtl_size_t count, dtl_dtype dtype,
                    dtl_rank_t dest, dtl_tag_t tag) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!buf && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        return count_status;
    }

    int err = MPI_Send(buf, mpi_count, mpi_type, dest, tag, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_SEND_FAILED;
    }
    return DTL_SUCCESS;
#else
    (void)count;
    (void)dtype;
    (void)dest;
    (void)tag;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#endif
}

dtl_status dtl_recv(dtl_context_t ctx, void* buf,
                    dtl_size_t count, dtl_dtype dtype,
                    dtl_rank_t source, dtl_tag_t tag) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!buf && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_source = (source == DTL_ANY_SOURCE) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == DTL_ANY_TAG) ? MPI_ANY_TAG : tag;

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        return count_status;
    }

    int err = MPI_Recv(buf, mpi_count, mpi_type, mpi_source,
                       mpi_tag, ctx->comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_RECV_FAILED;
    }
    return DTL_SUCCESS;
#else
    (void)count;
    (void)dtype;
    (void)source;
    (void)tag;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#endif
}

dtl_status dtl_sendrecv(dtl_context_t ctx,
                         const void* sendbuf, dtl_size_t sendcount,
                         dtl_dtype senddtype, dtl_rank_t dest,
                         dtl_tag_t sendtag,
                         void* recvbuf, dtl_size_t recvcount,
                         dtl_dtype recvdtype, dtl_rank_t source,
                         dtl_tag_t recvtag) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_source = (source == DTL_ANY_SOURCE) ? MPI_ANY_SOURCE : source;
    int mpi_recvtag = (recvtag == DTL_ANY_TAG) ? MPI_ANY_TAG : recvtag;

    int mpi_sendcount = 0;
    int mpi_recvcount = 0;
    dtl_status send_status = to_mpi_count(sendcount, &mpi_sendcount);
    if (send_status != DTL_SUCCESS) {
        return send_status;
    }
    dtl_status recv_status = to_mpi_count(recvcount, &mpi_recvcount);
    if (recv_status != DTL_SUCCESS) {
        return recv_status;
    }

    int err = MPI_Sendrecv(sendbuf, mpi_sendcount, send_type,
                           dest, sendtag,
                           recvbuf, mpi_recvcount, recv_type,
                           mpi_source, mpi_recvtag,
                           ctx->comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COMMUNICATION;
    }
    return DTL_SUCCESS;
#else
    (void)sendbuf;
    (void)sendcount;
    (void)senddtype;
    (void)dest;
    (void)sendtag;
    (void)recvbuf;
    (void)recvcount;
    (void)recvdtype;
    (void)source;
    (void)recvtag;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#endif
}

// ============================================================================
// Point-to-Point (Non-blocking)
// ============================================================================

dtl_status dtl_isend(dtl_context_t ctx, const void* buf,
                      dtl_size_t count, dtl_dtype dtype,
                      dtl_rank_t dest, dtl_tag_t tag,
                      dtl_request_t* request) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!request) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Allocate request
    dtl_request_s* req = nullptr;
    try {
        req = new dtl_request_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }
    req->magic = dtl_request_s::VALID_MAGIC;
    req->is_mpi_request = true;
    req->state = std::make_shared<dtl_request_s::async_state>();

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        delete req;
        return count_status;
    }

    int err = MPI_Isend(buf, mpi_count, mpi_type,
                        dest, tag, ctx->comm, &req->mpi_request);
    if (err != MPI_SUCCESS) {
        delete req;
        return DTL_ERROR_SEND_FAILED;
    }

    *request = req;
    return DTL_SUCCESS;
#else
    (void)buf;
    (void)count;
    (void)dtype;
    (void)dest;
    (void)tag;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#endif
}

dtl_status dtl_irecv(dtl_context_t ctx, void* buf,
                      dtl_size_t count, dtl_dtype dtype,
                      dtl_rank_t source, dtl_tag_t tag,
                      dtl_request_t* request) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!request) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Allocate request
    dtl_request_s* req = nullptr;
    try {
        req = new dtl_request_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }
    req->magic = dtl_request_s::VALID_MAGIC;
    req->is_mpi_request = true;
    req->state = std::make_shared<dtl_request_s::async_state>();

    int mpi_source = (source == DTL_ANY_SOURCE) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == DTL_ANY_TAG) ? MPI_ANY_TAG : tag;

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        delete req;
        return count_status;
    }

    int err = MPI_Irecv(buf, mpi_count, mpi_type,
                        mpi_source, mpi_tag, ctx->comm, &req->mpi_request);
    if (err != MPI_SUCCESS) {
        delete req;
        return DTL_ERROR_RECV_FAILED;
    }

    *request = req;
    return DTL_SUCCESS;
#else
    (void)buf;
    (void)count;
    (void)dtype;
    (void)source;
    (void)tag;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#endif
}

dtl_status dtl_wait(dtl_request_t request) {
    if (!is_valid_request(request)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    // Only call MPI_Wait if this is a real MPI request
    if (request->is_mpi_request) {
        int err = MPI_Wait(&request->mpi_request, MPI_STATUS_IGNORE);
        request->magic = 0;  // Invalidate
        delete request;
        if (err != MPI_SUCCESS) {
            return DTL_ERROR_COMMUNICATION;
        }
        return DTL_SUCCESS;
    }
    // Fall through for non-MPI requests (e.g., local RMA operations)
#endif
    // Non-MPI request: drive progress engine until completion.
    while (!(request->state && request->state->completed.load(std::memory_order_acquire))) {
        dtl::futures::progress_engine::instance().poll();
        std::this_thread::yield();
    }
    request->magic = 0;
    delete request;
    return DTL_SUCCESS;
}

dtl_status dtl_waitall(dtl_size_t count, dtl_request_t* requests) {
    if (!requests && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

    for (dtl_size_t i = 0; i < count; ++i) {
        if (!is_valid_request(requests[i])) {
            return DTL_ERROR_INVALID_ARGUMENT;
        }
    }

#ifdef DTL_HAS_MPI
    // Collect only real MPI requests.
    std::vector<MPI_Request> mpi_requests;
    mpi_requests.reserve(count);
    for (dtl_size_t i = 0; i < count; ++i) {
        if (requests[i]->is_mpi_request) {
            mpi_requests.push_back(requests[i]->mpi_request);
        }
    }

    if (!mpi_requests.empty()) {
        int mpi_count = 0;
        dtl_status count_status = to_mpi_count(
            static_cast<dtl_size_t>(mpi_requests.size()), &mpi_count);
        if (count_status != DTL_SUCCESS) {
            return count_status;
        }

        int err = MPI_Waitall(mpi_count, mpi_requests.data(), MPI_STATUSES_IGNORE);
        if (err != MPI_SUCCESS) {
            return DTL_ERROR_COMMUNICATION;
        }
    }

    // Drive completion for non-MPI requests.
    for (dtl_size_t i = 0; i < count; ++i) {
        if (requests[i]->is_mpi_request) {
            continue;
        }
        while (!(requests[i]->state &&
                 requests[i]->state->completed.load(std::memory_order_acquire))) {
            dtl::futures::progress_engine::instance().poll();
            std::this_thread::yield();
        }
    }
#else
    // Non-MPI mode: drive completion for all requests.
    for (dtl_size_t i = 0; i < count; ++i) {
        while (!(requests[i]->state &&
                 requests[i]->state->completed.load(std::memory_order_acquire))) {
            dtl::futures::progress_engine::instance().poll();
            std::this_thread::yield();
        }
    }
#endif

    // Invalidate and free all requests.
    for (dtl_size_t i = 0; i < count; ++i) {
        if (requests[i]->progress_callback_id != static_cast<dtl::size_type>(-1)) {
            dtl::futures::progress_engine::instance().unregister_callback(
                requests[i]->progress_callback_id);
            requests[i]->progress_callback_id = static_cast<dtl::size_type>(-1);
        }
        requests[i]->magic = 0;
        delete requests[i];
        requests[i] = nullptr;
    }
    return DTL_SUCCESS;
}

dtl_status dtl_test(dtl_request_t request, int* completed) {
    if (!is_valid_request(request) || !completed) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    // Only call MPI_Test if this is a real MPI request
    if (request->is_mpi_request) {
        int flag = 0;
        int err = MPI_Test(&request->mpi_request, &flag, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) {
            return DTL_ERROR_COMMUNICATION;
        }

        *completed = flag ? 1 : 0;
        if (flag) {
            request->magic = 0;
            delete request;
        }
        return DTL_SUCCESS;
    }
    // Fall through for non-MPI requests
#endif
    dtl::futures::progress_engine::instance().poll();
    *completed = (request->state && request->state->completed.load(std::memory_order_acquire)) ? 1 : 0;
    if (*completed) {
        request->magic = 0;
        delete request;
    }
    return DTL_SUCCESS;
}

void dtl_request_free(dtl_request_t request) {
    if (!is_valid_request(request)) {
        return;
    }

#ifdef DTL_HAS_MPI
    // Only call MPI functions if this is a real MPI request
    if (request->is_mpi_request && request->mpi_request != MPI_REQUEST_NULL) {
        MPI_Cancel(&request->mpi_request);
        MPI_Request_free(&request->mpi_request);
    }
#endif
    if (request->state) {
        request->state->cancelled.store(true, std::memory_order_release);
        request->state->completed.store(true, std::memory_order_release);
    }
    if (request->progress_callback_id != static_cast<dtl::size_type>(-1)) {
        dtl::futures::progress_engine::instance().unregister_callback(request->progress_callback_id);
        request->progress_callback_id = static_cast<dtl::size_type>(-1);
    }
    request->magic = 0;
    delete request;
}

// ============================================================================
// Collective - Synchronization
// ============================================================================

dtl_status dtl_barrier(dtl_context_t ctx) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    int err = MPI_Barrier(ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_BARRIER_FAILED;
    }
    return DTL_SUCCESS;
#else
    return DTL_SUCCESS;  // No-op for single process
#endif
}

// ============================================================================
// Collective - Broadcast
// ============================================================================

dtl_status dtl_broadcast(dtl_context_t ctx, void* buf,
                          dtl_size_t count, dtl_dtype dtype,
                          dtl_rank_t root) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!buf && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        return count_status;
    }

    int err = MPI_Bcast(buf, mpi_count, mpi_type, root, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_BROADCAST_FAILED;
    }
    return DTL_SUCCESS;
#else
    (void)count;
    (void)dtype;
    (void)root;
    return DTL_SUCCESS;  // No-op for single process
#endif
}

// ============================================================================
// Collective - Reduction
// ============================================================================

dtl_status dtl_reduce(dtl_context_t ctx,
                       const void* sendbuf, void* recvbuf,
                       dtl_size_t count, dtl_dtype dtype,
                       dtl_reduce_op op, dtl_rank_t root) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if ((!sendbuf || !recvbuf) && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    MPI_Op mpi_op = reduce_op_to_mpi(op);
    if (mpi_type == MPI_DATATYPE_NULL || mpi_op == MPI_OP_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        return count_status;
    }

    int err = MPI_Reduce(sendbuf, recvbuf, mpi_count,
                         mpi_type, mpi_op, root, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_REDUCE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, count * dtl_dtype_size(dtype));
    (void)op;
    (void)root;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_allreduce(dtl_context_t ctx,
                          const void* sendbuf, void* recvbuf,
                          dtl_size_t count, dtl_dtype dtype,
                          dtl_reduce_op op) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if ((!sendbuf || !recvbuf) && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    MPI_Op mpi_op = reduce_op_to_mpi(op);
    if (mpi_type == MPI_DATATYPE_NULL || mpi_op == MPI_OP_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        return count_status;
    }

    int err = MPI_Allreduce(sendbuf, recvbuf, mpi_count,
                            mpi_type, mpi_op, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_REDUCE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, count * dtl_dtype_size(dtype));
    (void)op;
    return DTL_SUCCESS;
#endif
}

// ============================================================================
// Collective - Gather/Scatter
// ============================================================================

dtl_status dtl_gather(dtl_context_t ctx,
                       const void* sendbuf, dtl_size_t sendcount,
                       dtl_dtype senddtype,
                       void* recvbuf, dtl_size_t recvcount,
                       dtl_dtype recvdtype, dtl_rank_t root) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_sendcount = 0;
    int mpi_recvcount = 0;
    dtl_status send_status = to_mpi_count(sendcount, &mpi_sendcount);
    if (send_status != DTL_SUCCESS) {
        return send_status;
    }
    dtl_status recv_status = to_mpi_count(recvcount, &mpi_recvcount);
    if (recv_status != DTL_SUCCESS) {
        return recv_status;
    }

    int err = MPI_Gather(sendbuf, mpi_sendcount, send_type,
                         recvbuf, mpi_recvcount, recv_type,
                         root, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, sendcount * dtl_dtype_size(senddtype));
    (void)recvcount;
    (void)recvdtype;
    (void)root;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_allgather(dtl_context_t ctx,
                          const void* sendbuf, dtl_size_t sendcount,
                          dtl_dtype senddtype,
                          void* recvbuf, dtl_size_t recvcount,
                          dtl_dtype recvdtype) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_sendcount = 0;
    int mpi_recvcount = 0;
    dtl_status send_status = to_mpi_count(sendcount, &mpi_sendcount);
    if (send_status != DTL_SUCCESS) {
        return send_status;
    }
    dtl_status recv_status = to_mpi_count(recvcount, &mpi_recvcount);
    if (recv_status != DTL_SUCCESS) {
        return recv_status;
    }

    int err = MPI_Allgather(sendbuf, mpi_sendcount, send_type,
                            recvbuf, mpi_recvcount, recv_type,
                            ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, sendcount * dtl_dtype_size(senddtype));
    (void)recvcount;
    (void)recvdtype;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_scatter(dtl_context_t ctx,
                        const void* sendbuf, dtl_size_t sendcount,
                        dtl_dtype senddtype,
                        void* recvbuf, dtl_size_t recvcount,
                        dtl_dtype recvdtype, dtl_rank_t root) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_sendcount = 0;
    int mpi_recvcount = 0;
    dtl_status send_status = to_mpi_count(sendcount, &mpi_sendcount);
    if (send_status != DTL_SUCCESS) {
        return send_status;
    }
    dtl_status recv_status = to_mpi_count(recvcount, &mpi_recvcount);
    if (recv_status != DTL_SUCCESS) {
        return recv_status;
    }

    int err = MPI_Scatter(sendbuf, mpi_sendcount, send_type,
                          recvbuf, mpi_recvcount, recv_type,
                          root, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, sendcount * dtl_dtype_size(senddtype));
    (void)recvcount;
    (void)recvdtype;
    (void)root;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_alltoall(dtl_context_t ctx,
                         const void* sendbuf, dtl_size_t sendcount,
                         dtl_dtype senddtype,
                         void* recvbuf, dtl_size_t recvcount,
                         dtl_dtype recvdtype) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_sendcount = 0;
    int mpi_recvcount = 0;
    dtl_status send_status = to_mpi_count(sendcount, &mpi_sendcount);
    if (send_status != DTL_SUCCESS) {
        return send_status;
    }
    dtl_status recv_status = to_mpi_count(recvcount, &mpi_recvcount);
    if (recv_status != DTL_SUCCESS) {
        return recv_status;
    }

    int err = MPI_Alltoall(sendbuf, mpi_sendcount, send_type,
                           recvbuf, mpi_recvcount, recv_type,
                           ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, sendcount * dtl_dtype_size(senddtype));
    (void)recvcount;
    (void)recvdtype;
    return DTL_SUCCESS;
#endif
}

// ============================================================================
// Variable-Count Collectives
// ============================================================================

dtl_status dtl_gatherv(dtl_context_t ctx,
                        const void* sendbuf, dtl_size_t sendcount,
                        dtl_dtype senddtype,
                        void* recvbuf, const dtl_size_t* recvcounts,
                        const dtl_size_t* displs, dtl_dtype recvdtype,
                        dtl_rank_t root) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Convert counts/displs to int arrays for MPI
    int size = ctx->size;
    std::vector<int> int_recvcounts(size);
    std::vector<int> int_displs(size);

    if (ctx->rank == root && recvcounts && displs) {
        dtl_status conv_counts = convert_size_array_to_int(recvcounts, size, int_recvcounts.data());
        if (conv_counts != DTL_SUCCESS) {
            return conv_counts;
        }
        dtl_status conv_displs = convert_size_array_to_int(displs, size, int_displs.data());
        if (conv_displs != DTL_SUCCESS) {
            return conv_displs;
        }
    }

    int mpi_sendcount = 0;
    dtl_status send_status = to_mpi_count(sendcount, &mpi_sendcount);
    if (send_status != DTL_SUCCESS) {
        return send_status;
    }

    int err = MPI_Gatherv(sendbuf, mpi_sendcount, send_type,
                          recvbuf, int_recvcounts.data(), int_displs.data(),
                          recv_type, root, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, sendcount * dtl_dtype_size(senddtype));
    (void)recvcounts;
    (void)displs;
    (void)recvdtype;
    (void)root;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_scatterv(dtl_context_t ctx,
                         const void* sendbuf,
                         const dtl_size_t* sendcounts,
                         const dtl_size_t* displs, dtl_dtype senddtype,
                         void* recvbuf, dtl_size_t recvcount,
                         dtl_dtype recvdtype, dtl_rank_t root) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Convert counts/displs to int arrays for MPI
    int size = ctx->size;
    std::vector<int> int_sendcounts(size);
    std::vector<int> int_displs(size);

    if (ctx->rank == root && sendcounts && displs) {
        dtl_status conv_counts = convert_size_array_to_int(sendcounts, size, int_sendcounts.data());
        if (conv_counts != DTL_SUCCESS) {
            return conv_counts;
        }
        dtl_status conv_displs = convert_size_array_to_int(displs, size, int_displs.data());
        if (conv_displs != DTL_SUCCESS) {
            return conv_displs;
        }
    }

    int mpi_recvcount = 0;
    dtl_status recv_status = to_mpi_count(recvcount, &mpi_recvcount);
    if (recv_status != DTL_SUCCESS) {
        return recv_status;
    }

    int err = MPI_Scatterv(sendbuf, int_sendcounts.data(), int_displs.data(),
                           send_type, recvbuf, mpi_recvcount,
                           recv_type, root, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    if (sendcounts && displs) {
        const char* src = static_cast<const char*>(sendbuf);
        src += displs[0] * dtl_dtype_size(senddtype);
        std::memcpy(recvbuf, src, recvcount * dtl_dtype_size(recvdtype));
    }
    (void)root;
    return DTL_SUCCESS;
#endif
}

// ============================================================================
// Variable-Size Collectives (Phase 12.5)
// ============================================================================

dtl_status dtl_allgatherv(dtl_context_t ctx,
                            const void* sendbuf,
                            dtl_size_t sendcount, dtl_dtype dtype,
                            void* recvbuf,
                            const dtl_size_t* recvcounts,
                            const dtl_size_t* displs) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    if (mpi_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Convert counts/displs to int arrays for MPI
    int size = ctx->size;
    std::vector<int> int_recvcounts(size);
    std::vector<int> int_displs(size);

    if (recvcounts && displs) {
        dtl_status conv_counts = convert_size_array_to_int(recvcounts, size, int_recvcounts.data());
        if (conv_counts != DTL_SUCCESS) {
            return conv_counts;
        }
        dtl_status conv_displs = convert_size_array_to_int(displs, size, int_displs.data());
        if (conv_displs != DTL_SUCCESS) {
            return conv_displs;
        }
    }

    int mpi_sendcount = 0;
    dtl_status send_status = to_mpi_count(sendcount, &mpi_sendcount);
    if (send_status != DTL_SUCCESS) {
        return send_status;
    }

    int err = MPI_Allgatherv(sendbuf, mpi_sendcount, mpi_type,
                              recvbuf, int_recvcounts.data(), int_displs.data(),
                              mpi_type, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf
    std::memcpy(recvbuf, sendbuf, sendcount * dtl_dtype_size(dtype));
    (void)recvcounts;
    (void)displs;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_alltoallv(dtl_context_t ctx,
                           const void* sendbuf,
                           const dtl_size_t* sendcounts,
                           const dtl_size_t* sdispls,
                           dtl_dtype senddtype,
                           void* recvbuf,
                           const dtl_size_t* recvcounts,
                           const dtl_size_t* rdispls,
                           dtl_dtype recvdtype) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype send_type = dtype_to_mpi(senddtype);
    MPI_Datatype recv_type = dtype_to_mpi(recvdtype);
    if (send_type == MPI_DATATYPE_NULL || recv_type == MPI_DATATYPE_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Convert counts/displs to int arrays for MPI
    int size = ctx->size;
    std::vector<int> int_sendcounts(size);
    std::vector<int> int_sdispls(size);
    std::vector<int> int_recvcounts(size);
    std::vector<int> int_rdispls(size);

    if (sendcounts && sdispls && recvcounts && rdispls) {
        dtl_status conv_sendcounts = convert_size_array_to_int(sendcounts, size, int_sendcounts.data());
        if (conv_sendcounts != DTL_SUCCESS) {
            return conv_sendcounts;
        }
        dtl_status conv_sdispls = convert_size_array_to_int(sdispls, size, int_sdispls.data());
        if (conv_sdispls != DTL_SUCCESS) {
            return conv_sdispls;
        }
        dtl_status conv_recvcounts = convert_size_array_to_int(recvcounts, size, int_recvcounts.data());
        if (conv_recvcounts != DTL_SUCCESS) {
            return conv_recvcounts;
        }
        dtl_status conv_rdispls = convert_size_array_to_int(rdispls, size, int_rdispls.data());
        if (conv_rdispls != DTL_SUCCESS) {
            return conv_rdispls;
        }
    }

    int err = MPI_Alltoallv(sendbuf, int_sendcounts.data(), int_sdispls.data(),
                             send_type,
                             recvbuf, int_recvcounts.data(), int_rdispls.data(),
                             recv_type, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: copy sendbuf to recvbuf based on counts
    if (sendcounts && sdispls) {
        const char* src = static_cast<const char*>(sendbuf);
        src += sdispls[0] * dtl_dtype_size(senddtype);
        dtl_size_t count = sendcounts[0];
        std::memcpy(recvbuf, src, count * dtl_dtype_size(senddtype));
    }
    (void)recvcounts;
    (void)rdispls;
    (void)recvdtype;
    return DTL_SUCCESS;
#endif
}

}  // extern "C"

// ============================================================================
// Scan / Prefix Operations (Phase 16)
// ============================================================================

// Identity helpers for exscan rank-0 initialization
namespace {

template <typename T>
static void fill_identity(void* buf, dtl_size_t count, dtl_reduce_op op) {
    T* data = static_cast<T*>(buf);
    T identity{};
    switch (op) {
        case DTL_OP_SUM:  identity = T{0}; break;
        case DTL_OP_PROD: identity = T{1}; break;
        case DTL_OP_MIN:  identity = std::numeric_limits<T>::max(); break;
        case DTL_OP_MAX:  identity = std::numeric_limits<T>::lowest(); break;
        default:          identity = T{0}; break;
    }
    for (dtl_size_t i = 0; i < count; ++i) {
        data[i] = identity;
    }
}

static void fill_identity_for_dtype(void* buf, dtl_size_t count, dtl_dtype dtype, dtl_reduce_op op) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    fill_identity<int8_t>(buf, count, op);   break;
        case DTL_DTYPE_INT16:   fill_identity<int16_t>(buf, count, op);  break;
        case DTL_DTYPE_INT32:   fill_identity<int32_t>(buf, count, op);  break;
        case DTL_DTYPE_INT64:   fill_identity<int64_t>(buf, count, op);  break;
        case DTL_DTYPE_UINT8:
        case DTL_DTYPE_BYTE:
        case DTL_DTYPE_BOOL:    fill_identity<uint8_t>(buf, count, op);  break;
        case DTL_DTYPE_UINT16:  fill_identity<uint16_t>(buf, count, op); break;
        case DTL_DTYPE_UINT32:  fill_identity<uint32_t>(buf, count, op); break;
        case DTL_DTYPE_UINT64:  fill_identity<uint64_t>(buf, count, op); break;
        case DTL_DTYPE_FLOAT32: fill_identity<float>(buf, count, op);    break;
        case DTL_DTYPE_FLOAT64: fill_identity<double>(buf, count, op);   break;
        default: break;
    }
}

} // anonymous namespace

extern "C" {

dtl_status dtl_scan(dtl_context_t ctx,
                     const void* sendbuf, void* recvbuf,
                     dtl_size_t count, dtl_dtype dtype,
                     dtl_reduce_op op) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if ((!sendbuf || !recvbuf) && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    MPI_Op mpi_op = reduce_op_to_mpi(op);
    if (mpi_type == MPI_DATATYPE_NULL || mpi_op == MPI_OP_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        return count_status;
    }

    int err = MPI_Scan(sendbuf, recvbuf, mpi_count,
                       mpi_type, mpi_op, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }
    return DTL_SUCCESS;
#else
    // Single process: inclusive scan is just a copy
    std::memcpy(recvbuf, sendbuf, count * dtl_dtype_size(dtype));
    (void)op;
    return DTL_SUCCESS;
#endif
}

dtl_status dtl_exscan(dtl_context_t ctx,
                       const void* sendbuf, void* recvbuf,
                       dtl_size_t count, dtl_dtype dtype,
                       dtl_reduce_op op) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if ((!sendbuf || !recvbuf) && count > 0) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    MPI_Datatype mpi_type = dtype_to_mpi(dtype);
    MPI_Op mpi_op = reduce_op_to_mpi(op);
    if (mpi_type == MPI_DATATYPE_NULL || mpi_op == MPI_OP_NULL) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    int mpi_count = 0;
    dtl_status count_status = to_mpi_count(count, &mpi_count);
    if (count_status != DTL_SUCCESS) {
        return count_status;
    }

    int err = MPI_Exscan(sendbuf, recvbuf, mpi_count,
                         mpi_type, mpi_op, ctx->comm);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COLLECTIVE_FAILED;
    }

    // MPI_Exscan leaves recvbuf undefined on rank 0 — fill with identity
    if (ctx->rank == 0) {
        fill_identity_for_dtype(recvbuf, count, dtype, op);
    }
    return DTL_SUCCESS;
#else
    // Single process: exclusive scan produces identity on rank 0
    fill_identity_for_dtype(recvbuf, count, dtype, op);
    (void)sendbuf;
    return DTL_SUCCESS;
#endif
}

// ============================================================================
// Probe Operations (Phase 16)
// ============================================================================

dtl_status dtl_probe(dtl_context_t ctx,
                      dtl_rank_t source, dtl_tag_t tag,
                      dtl_dtype dtype,
                      dtl_message_info_t* info) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

#ifdef DTL_HAS_MPI
    int mpi_source = (source == DTL_ANY_SOURCE) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == DTL_ANY_TAG) ? MPI_ANY_TAG : tag;

    MPI_Status status;
    int err = MPI_Probe(mpi_source, mpi_tag, ctx->comm, &status);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COMMUNICATION;
    }

    if (info) {
        info->source = status.MPI_SOURCE;
        info->tag = status.MPI_TAG;

        MPI_Datatype mpi_type = dtype_to_mpi(dtype);
        if (mpi_type != MPI_DATATYPE_NULL) {
            int mpi_count = 0;
            MPI_Get_count(&status, mpi_type, &mpi_count);
            info->count = (mpi_count == MPI_UNDEFINED) ? 0 : static_cast<dtl_size_t>(mpi_count);
        } else {
            info->count = 0;
        }
    }

    return DTL_SUCCESS;
#else
    (void)source;
    (void)tag;
    (void)dtype;
    (void)info;
    return DTL_ERROR_BACKEND_UNAVAILABLE;
#endif
}

dtl_status dtl_iprobe(dtl_context_t ctx,
                       dtl_rank_t source, dtl_tag_t tag,
                       dtl_dtype dtype,
                       int* flag, dtl_message_info_t* info) {
    if (!is_valid_context(ctx)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!flag) {
        return DTL_ERROR_NULL_POINTER;
    }

#ifdef DTL_HAS_MPI
    int mpi_source = (source == DTL_ANY_SOURCE) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == DTL_ANY_TAG) ? MPI_ANY_TAG : tag;

    MPI_Status status;
    int mpi_flag = 0;
    int err = MPI_Iprobe(mpi_source, mpi_tag, ctx->comm, &mpi_flag, &status);
    if (err != MPI_SUCCESS) {
        return DTL_ERROR_COMMUNICATION;
    }

    *flag = mpi_flag ? 1 : 0;

    if (mpi_flag && info) {
        info->source = status.MPI_SOURCE;
        info->tag = status.MPI_TAG;

        MPI_Datatype mpi_type = dtype_to_mpi(dtype);
        if (mpi_type != MPI_DATATYPE_NULL) {
            int mpi_count = 0;
            MPI_Get_count(&status, mpi_type, &mpi_count);
            info->count = (mpi_count == MPI_UNDEFINED) ? 0 : static_cast<dtl_size_t>(mpi_count);
        } else {
            info->count = 0;
        }
    }

    return DTL_SUCCESS;
#else
    (void)source;
    (void)tag;
    (void)dtype;
    (void)info;
    *flag = 0;
    return DTL_SUCCESS;  // No messages available without MPI
#endif
}

}  // extern "C"
