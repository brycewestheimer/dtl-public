// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file nccl_communicator.hpp
/// @brief NCCL communicator for GPU collective operations
/// @details Provides high-performance GPU-to-GPU collective communication
///          using NVIDIA NCCL library.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/reduction_ops.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#if DTL_ENABLE_NCCL
#include <nccl.h>
#endif

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dtl {
namespace nccl {

// ============================================================================
// NCCL Data Types
// ============================================================================

/// @brief NCCL data type enumeration
enum class nccl_dtype {
    int8,
    uint8,
    int32,
    uint32,
    int64,
    uint64,
    float16,
    float32,
    float64
};

/// @brief NCCL reduction operations
enum class nccl_op {
    sum,
    prod,
    max,
    min,
    avg
};

// ============================================================================
// NCCL Communicator
// ============================================================================

/// @brief NCCL-based communicator for GPU collective operations
/// @details Wraps NCCL communicator handle and provides DTL interface
///          for high-performance multi-GPU collective communication.
class nccl_communicator {
public:
    using size_type = dtl::size_type;
    /// @brief Default constructor
    nccl_communicator() = default;

#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
    /// @brief Construct from NCCL communicator
    /// @param comm NCCL communicator handle
    /// @param rank This process's rank
    /// @param size Total number of ranks
    /// @param stream CUDA stream for operations
    /// @param owns_stream Whether this instance owns and destroys the stream
    explicit nccl_communicator(ncclComm_t comm, rank_t rank, rank_t size,
                               cudaStream_t stream = nullptr,
                               bool owns_stream = true)
        : comm_(comm)
        , rank_(rank)
        , size_(size)
        , stream_(stream)
        , owns_stream_(owns_stream)
        , barrier_scratch_(nullptr) {
        // Pre-allocate persistent scratch buffer for barrier
        cudaError_t alloc_err = cudaMalloc(&barrier_scratch_, sizeof(int));
        if (alloc_err != cudaSuccess) {
            barrier_scratch_ = nullptr;
            // barrier() will return error when called
        }
    }
#endif

    /// @brief Destructor - frees NCCL/CUDA resources if owned
    ~nccl_communicator() {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        (void)release_resources(/*destroy_comm=*/true, /*destroy_stream=*/true);
#endif
    }

    // Non-copyable
    nccl_communicator(const nccl_communicator&) = delete;
    nccl_communicator& operator=(const nccl_communicator&) = delete;

    // Movable
    nccl_communicator(nccl_communicator&& other) noexcept
        : rank_(other.rank_)
        , size_(other.size_)
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        , comm_(other.comm_)
        , stream_(other.stream_)
        , owns_stream_(other.owns_stream_)
        , barrier_scratch_(other.barrier_scratch_)
#endif
    {
#if defined(DTL_ENABLE_NCCL)
        other.comm_ = nullptr;
#endif
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        other.stream_ = nullptr;
        other.owns_stream_ = false;
        other.barrier_scratch_ = nullptr;
#endif
    }

    nccl_communicator& operator=(nccl_communicator&& other) noexcept {
        if (this != &other) {
            rank_ = other.rank_;
            size_ = other.size_;
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
            (void)release_resources(/*destroy_comm=*/true, /*destroy_stream=*/true);
            comm_ = other.comm_;
            stream_ = other.stream_;
            owns_stream_ = other.owns_stream_;
            barrier_scratch_ = other.barrier_scratch_;
            other.comm_ = nullptr;
            other.stream_ = nullptr;
            other.owns_stream_ = false;
            other.barrier_scratch_ = nullptr;
#endif
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Communicator Interface
    // ------------------------------------------------------------------------

    [[nodiscard]] rank_t rank() const noexcept { return rank_; }
    [[nodiscard]] rank_t size() const noexcept { return size_; }

    [[nodiscard]] communicator_properties properties() const noexcept {
        return communicator_properties{
            .size = size_,
            .rank = rank_,
            .is_inter = false,
            .is_derived = false,
            .name = "nccl"
        };
    }

    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_NCCL
        return comm_ != nullptr;
#else
        return false;
#endif
    }

    // ------------------------------------------------------------------------
    // Point-to-Point (NCCL 2.7+ via ncclSend/ncclRecv)
    // ------------------------------------------------------------------------

    /// @brief Blocking send using ncclSend (NCCL 2.7+)
    /// @details NCCL does not support message tags; the tag parameter is ignored.
    ///          Data must reside in GPU memory.
    result<void> send_impl(const void* data, size_type count,
                          size_type elem_size, rank_t dest, int tag) {
        (void)tag;  // NCCL does not support message tags
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(data, count * elem_size, "send");
            !buffer_check) {
            return buffer_check;
        }
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 7)
        size_type bytes = count * elem_size;
        ncclResult_t nccl_res = ncclGroupStart();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "ncclGroupStart failed in send");
        }
        nccl_res = ncclSend(data, bytes, ncclChar, dest, comm_, stream_);
        if (nccl_res != ncclSuccess) {
            ncclGroupEnd();
            return make_error<void>(status_code::backend_error,
                                   "ncclSend failed");
        }
        nccl_res = ncclGroupEnd();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "ncclGroupEnd failed in send");
        }
        return synchronize_blocking("send");
#else
        (void)data; (void)count; (void)elem_size; (void)dest;
        return make_error<void>(status_code::not_supported,
                               "NCCL send requires NCCL 2.7+");
#endif
#else
        (void)data; (void)count; (void)elem_size; (void)dest;
        return make_error<void>(status_code::not_supported,
                               "NCCL/CUDA support not enabled");
#endif
    }

    /// @brief Blocking recv using ncclRecv (NCCL 2.7+)
    /// @details NCCL does not support message tags; the tag parameter is ignored.
    ///          Data must reside in GPU memory.
    result<void> recv_impl(void* data, size_type count,
                          size_type elem_size, rank_t source, int tag) {
        (void)tag;  // NCCL does not support message tags
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(data, count * elem_size, "recv");
            !buffer_check) {
            return buffer_check;
        }
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 7)
        size_type bytes = count * elem_size;
        ncclResult_t nccl_res = ncclGroupStart();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "ncclGroupStart failed in recv");
        }
        nccl_res = ncclRecv(data, bytes, ncclChar, source, comm_, stream_);
        if (nccl_res != ncclSuccess) {
            ncclGroupEnd();
            return make_error<void>(status_code::backend_error,
                                   "ncclRecv failed");
        }
        nccl_res = ncclGroupEnd();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "ncclGroupEnd failed in recv");
        }
        return synchronize_blocking("recv");
#else
        (void)data; (void)count; (void)elem_size; (void)source;
        return make_error<void>(status_code::not_supported,
                               "NCCL recv requires NCCL 2.7+");
#endif
#else
        (void)data; (void)count; (void)elem_size; (void)source;
        return make_error<void>(status_code::not_supported,
                               "NCCL/CUDA support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Non-Blocking Point-to-Point (NCCL 2.7+ with CUDA events)
    // ------------------------------------------------------------------------

    /// @brief Non-blocking send using ncclSend + CUDA event
    /// @details Enqueues ncclSend on the stream without synchronizing.
    ///          Returns a request_handle wrapping a CUDA event pointer.
    ///          Use wait_event() or test_event() to check completion.
    /// @note NCCL does not support message tags; the tag parameter is ignored.
    result<request_handle> isend(const void* data, size_type count,
                                 size_type elem_size, rank_t dest, int tag) {
        (void)tag;
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<request_handle>(status_code::invalid_state,
                                             "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(data, count * elem_size, "isend");
            !buffer_check) {
            return result<request_handle>{buffer_check.error()};
        }
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 7)
        size_type bytes = count * elem_size;
        ncclResult_t nccl_res = ncclGroupStart();
        if (nccl_res != ncclSuccess) {
            return make_error<request_handle>(status_code::backend_error,
                                             "ncclGroupStart failed in isend");
        }
        nccl_res = ncclSend(data, bytes, ncclChar, dest, comm_, stream_);
        if (nccl_res != ncclSuccess) {
            ncclGroupEnd();
            return make_error<request_handle>(status_code::backend_error,
                                             "ncclSend failed in isend");
        }
        nccl_res = ncclGroupEnd();
        if (nccl_res != ncclSuccess) {
            return make_error<request_handle>(status_code::backend_error,
                                             "ncclGroupEnd failed in isend");
        }
        // Record a CUDA event to track async completion
        cudaEvent_t* event = new (std::nothrow) cudaEvent_t;
        if (!event) {
            return make_error<request_handle>(status_code::out_of_memory,
                                             "failed to allocate CUDA event");
        }
        cudaError_t cuda_err = cudaEventCreate(event);
        if (cuda_err != cudaSuccess) {
            delete event;
            return make_error<request_handle>(status_code::backend_error,
                                             "cudaEventCreate failed in isend");
        }
        cuda_err = cudaEventRecord(*event, stream_);
        if (cuda_err != cudaSuccess) {
            cudaEventDestroy(*event);
            delete event;
            return make_error<request_handle>(status_code::backend_error,
                                             "cudaEventRecord failed in isend");
        }
        return request_handle{static_cast<void*>(event)};
#else
        (void)data; (void)count; (void)elem_size; (void)dest;
        return make_error<request_handle>(status_code::not_supported,
                                         "NCCL isend requires NCCL 2.7+");
#endif
#else
        (void)data; (void)count; (void)elem_size; (void)dest;
        return make_error<request_handle>(status_code::not_supported,
                                         "NCCL/CUDA support not enabled");
#endif
    }

    /// @brief Non-blocking recv using ncclRecv + CUDA event
    /// @details Enqueues ncclRecv on the stream without synchronizing.
    ///          Returns a request_handle wrapping a CUDA event pointer.
    ///          Use wait_event() or test_event() to check completion.
    /// @note NCCL does not support message tags; the tag parameter is ignored.
    result<request_handle> irecv(void* data, size_type count,
                                 size_type elem_size, rank_t source, int tag) {
        (void)tag;
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<request_handle>(status_code::invalid_state,
                                             "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(data, count * elem_size, "irecv");
            !buffer_check) {
            return result<request_handle>{buffer_check.error()};
        }
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 7)
        size_type bytes = count * elem_size;
        ncclResult_t nccl_res = ncclGroupStart();
        if (nccl_res != ncclSuccess) {
            return make_error<request_handle>(status_code::backend_error,
                                             "ncclGroupStart failed in irecv");
        }
        nccl_res = ncclRecv(data, bytes, ncclChar, source, comm_, stream_);
        if (nccl_res != ncclSuccess) {
            ncclGroupEnd();
            return make_error<request_handle>(status_code::backend_error,
                                             "ncclRecv failed in irecv");
        }
        nccl_res = ncclGroupEnd();
        if (nccl_res != ncclSuccess) {
            return make_error<request_handle>(status_code::backend_error,
                                             "ncclGroupEnd failed in irecv");
        }
        // Record a CUDA event to track async completion
        cudaEvent_t* event = new (std::nothrow) cudaEvent_t;
        if (!event) {
            return make_error<request_handle>(status_code::out_of_memory,
                                             "failed to allocate CUDA event");
        }
        cudaError_t cuda_err = cudaEventCreate(event);
        if (cuda_err != cudaSuccess) {
            delete event;
            return make_error<request_handle>(status_code::backend_error,
                                             "cudaEventCreate failed in irecv");
        }
        cuda_err = cudaEventRecord(*event, stream_);
        if (cuda_err != cudaSuccess) {
            cudaEventDestroy(*event);
            delete event;
            return make_error<request_handle>(status_code::backend_error,
                                             "cudaEventRecord failed in irecv");
        }
        return request_handle{static_cast<void*>(event)};
#else
        (void)data; (void)count; (void)elem_size; (void)source;
        return make_error<request_handle>(status_code::not_supported,
                                         "NCCL irecv requires NCCL 2.7+");
#endif
#else
        (void)data; (void)count; (void)elem_size; (void)source;
        return make_error<request_handle>(status_code::not_supported,
                                         "NCCL/CUDA support not enabled");
#endif
    }

    /// @brief Wait for a NCCL async operation to complete
    /// @details Calls cudaEventSynchronize on the CUDA event stored in the
    ///          request handle. Destroys the event and frees the handle.
    result<void> wait_event(request_handle& req) {
#if defined(DTL_ENABLE_CUDA)
        if (!req.handle) {
            return {};
        }
        auto* event = static_cast<cudaEvent_t*>(req.handle);
        cudaError_t sync_err = cudaEventSynchronize(*event);
        cudaError_t destroy_err = cudaEventDestroy(*event);
        delete event;
        req.handle = nullptr;

        if (sync_err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaEventSynchronize failed in wait_event");
        }
        if (destroy_err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaEventDestroy failed in wait_event");
        }
        return {};
#else
        (void)req;
        return {};
#endif
    }

    /// @brief Test if a NCCL async operation has completed
    /// @details Calls cudaEventQuery on the CUDA event. If complete,
    ///          destroys the event and frees the handle.
    /// @return true if the operation has completed
    [[nodiscard]] result<bool> test_event(request_handle& req) {
#if defined(DTL_ENABLE_CUDA)
        if (!req.handle) return result<bool>::success(true);
        auto* event = static_cast<cudaEvent_t*>(req.handle);
        cudaError_t err = cudaEventQuery(*event);
        if (err == cudaSuccess) {
            cudaError_t destroy_err = cudaEventDestroy(*event);
            delete event;
            req.handle = nullptr;
            if (destroy_err != cudaSuccess) {
                return make_error<bool>(status_code::backend_error,
                                        "cudaEventDestroy failed in test_event");
            }
            return result<bool>::success(true);
        }
        if (err == cudaErrorNotReady) {
            return result<bool>::success(false);
        }
        cudaEventDestroy(*event);
        delete event;
        req.handle = nullptr;
        return make_error<bool>(status_code::backend_error,
                                "cudaEventQuery failed in test_event");
#else
        (void)req;
        return result<bool>::success(true);
#endif
    }

    // ------------------------------------------------------------------------
    // Collective Communication
    // ------------------------------------------------------------------------

    result<void> barrier() {
        // NCCL doesn't have barrier - use allreduce on single int
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid() || !barrier_scratch_) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL barrier scratch buffer not allocated");
        }

        int dummy = 0;
        cudaError_t copy_err =
            cudaMemcpy(barrier_scratch_, &dummy, sizeof(int), cudaMemcpyHostToDevice);
        if (copy_err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMemcpy failed in barrier");
        }

        ncclResult_t result = ncclAllReduce(barrier_scratch_, barrier_scratch_, 1,
                                            ncclInt, ncclSum, comm_, stream_);
        if (result != ncclSuccess) {
            return make_error<void>(status_code::barrier_failed,
                                   "NCCL barrier (allreduce) failed");
        }
        return synchronize_blocking("barrier");
#else
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    result<void> broadcast_impl(void* data, size_type count,
                               size_type elem_size, rank_t root) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(data, count * elem_size, "broadcast");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t result = ncclBroadcast(data, data, count * elem_size,
                                            ncclChar, root, comm_, stream_);
        if (result != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclBroadcast failed");
        }
        return synchronize_blocking("broadcast");
#else
        (void)data; (void)count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    result<void> gather_impl(const void* send_data, size_type send_count,
                            void* recv_data, size_type recv_count,
                            size_type elem_size, rank_t root) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(send_data, send_count * elem_size, "gather send");
            !buffer_check) {
            return buffer_check;
        }
        if (rank_ == root) {
            if (auto buffer_check = require_device_buffer(
                    recv_data, recv_count * elem_size * static_cast<size_type>(size_), "gather recv");
                !buffer_check) {
                return buffer_check;
            }
        }
        // Use ncclSend/ncclRecv within group ops (NCCL 2.7+)
        size_type send_bytes = send_count * elem_size;

        ncclResult_t nccl_res = ncclGroupStart();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclGroupStart failed in gather");
        }

        // Every rank sends its data to root
        nccl_res = ncclSend(send_data, send_bytes, ncclChar, root, comm_, stream_);
        if (nccl_res != ncclSuccess) {
            ncclGroupEnd();
            return make_error<void>(status_code::collective_failure,
                                   "ncclSend in gather failed");
        }

        // Root receives from all ranks
        if (rank_ == root) {
            size_type recv_bytes = recv_count * elem_size;
            for (rank_t r = 0; r < size_; ++r) {
                void* recv_ptr = static_cast<char*>(recv_data) +
                                 static_cast<size_type>(r) * recv_bytes;
                nccl_res = ncclRecv(recv_ptr, recv_bytes, ncclChar, r, comm_, stream_);
                if (nccl_res != ncclSuccess) {
                    ncclGroupEnd();
                    return make_error<void>(status_code::collective_failure,
                                           "ncclRecv in gather failed");
                }
            }
        }

        nccl_res = ncclGroupEnd();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclGroupEnd failed in gather");
        }
        return synchronize_blocking("gather");
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL/CUDA support not enabled");
#endif
    }

    result<void> scatter_impl(const void* send_data, size_type send_count,
                             void* recv_data, size_type recv_count,
                             size_type elem_size, rank_t root) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (rank_ == root) {
            if (auto buffer_check = require_device_buffer(
                    send_data, send_count * elem_size * static_cast<size_type>(size_), "scatter send");
                !buffer_check) {
                return buffer_check;
            }
        }
        if (auto buffer_check = require_device_buffer(recv_data, recv_count * elem_size, "scatter recv");
            !buffer_check) {
            return buffer_check;
        }
        // Use ncclSend/ncclRecv within group ops (NCCL 2.7+)
        size_type recv_bytes = recv_count * elem_size;

        ncclResult_t nccl_res = ncclGroupStart();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclGroupStart failed in scatter");
        }

        // Root sends to all ranks
        if (rank_ == root) {
            size_type send_bytes = send_count * elem_size;
            for (rank_t r = 0; r < size_; ++r) {
                const void* send_ptr = static_cast<const char*>(send_data) +
                                       static_cast<size_type>(r) * send_bytes;
                nccl_res = ncclSend(send_ptr, send_bytes, ncclChar, r, comm_, stream_);
                if (nccl_res != ncclSuccess) {
                    ncclGroupEnd();
                    return make_error<void>(status_code::collective_failure,
                                           "ncclSend in scatter failed");
                }
            }
        }

        // Every rank receives from root
        nccl_res = ncclRecv(recv_data, recv_bytes, ncclChar, root, comm_, stream_);
        if (nccl_res != ncclSuccess) {
            ncclGroupEnd();
            return make_error<void>(status_code::collective_failure,
                                   "ncclRecv in scatter failed");
        }

        nccl_res = ncclGroupEnd();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclGroupEnd failed in scatter");
        }
        return synchronize_blocking("scatter");
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL/CUDA support not enabled");
#endif
    }

    result<void> allgather_impl(const void* send_data, size_type send_count,
                               void* recv_data, size_type recv_count,
                               size_type elem_size) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(
                send_data, send_count * elem_size, "allgather send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(
                recv_data, recv_count * elem_size * static_cast<size_type>(size_), "allgather recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t result = ncclAllGather(send_data, recv_data,
                                            send_count * elem_size,
                                            ncclChar, comm_, stream_);
        if (result != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclAllGather failed");
        }
        (void)recv_count;
        return synchronize_blocking("allgather");
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief All-to-all collective using grouped NCCL Send/Recv
    /// @details NCCL has no native alltoall, so this emulates it by issuing
    ///          one ncclSend and one ncclRecv per peer within a group operation.
    /// @param send_data Send buffer (size * count * elem_size bytes)
    /// @param recv_data Receive buffer (size * count * elem_size bytes)
    /// @param count Number of elements per rank
    /// @param elem_size Size of each element in bytes
    /// @return Success or error
    result<void> alltoall_impl(const void* send_data, void* recv_data,
                               size_type count, size_type elem_size) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(
                send_data, count * elem_size * static_cast<size_type>(size_), "alltoall send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(
                recv_data, count * elem_size * static_cast<size_type>(size_), "alltoall recv");
            !buffer_check) {
            return buffer_check;
        }
        // NCCL has no native alltoall — emulate with grouped Send/Recv
        size_type chunk_bytes = count * elem_size;

        ncclResult_t nccl_res = ncclGroupStart();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclGroupStart failed in alltoall");
        }

        for (rank_t r = 0; r < size_; ++r) {
            const void* send_ptr = static_cast<const char*>(send_data) +
                                   static_cast<size_type>(r) * chunk_bytes;
            nccl_res = ncclSend(send_ptr, chunk_bytes, ncclChar, r, comm_, stream_);
            if (nccl_res != ncclSuccess) {
                ncclGroupEnd();
                return make_error<void>(status_code::collective_failure,
                                       "ncclSend in alltoall failed");
            }

            void* recv_ptr = static_cast<char*>(recv_data) +
                             static_cast<size_type>(r) * chunk_bytes;
            nccl_res = ncclRecv(recv_ptr, chunk_bytes, ncclChar, r, comm_, stream_);
            if (nccl_res != ncclSuccess) {
                ncclGroupEnd();
                return make_error<void>(status_code::collective_failure,
                                       "ncclRecv in alltoall failed");
            }
        }

        nccl_res = ncclGroupEnd();
        if (nccl_res != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclGroupEnd failed in alltoall");
        }
        return synchronize_blocking("alltoall");
#else
        (void)send_data; (void)recv_data; (void)count; (void)elem_size;
        return make_error<void>(status_code::not_supported,
                               "NCCL/CUDA support not enabled");
#endif
    }

    /// @brief Reduce-scatter collective using native ncclReduceScatter
    /// @details Byte-level reduce-scatter with ncclSum on ncclChar.
    ///          For typed reductions with proper semantics, prefer the
    ///          typed reduce_scatter<T>() template method.
    /// @param send_data Send buffer (size * recv_count * elem_size bytes)
    /// @param recv_data Receive buffer (recv_count * elem_size bytes)
    /// @param recv_count Number of elements this rank receives
    /// @param elem_size Size of each element in bytes
    /// @return Success or error
    result<void> reduce_scatter_impl(const void* send_data, void* recv_data,
                                     size_type recv_count, size_type elem_size) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(
                send_data, recv_count * elem_size * static_cast<size_type>(size_), "reduce_scatter send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(
                recv_data, recv_count * elem_size, "reduce_scatter recv");
            !buffer_check) {
            return buffer_check;
        }
        // Use ncclReduceScatter with ncclChar (byte-level) and ncclSum.
        // This is correct for byte-level reduction only when used with sum
        // on raw bytes (identity for copy operations). For typed reductions,
        // use the typed reduce_scatter<T>() template instead.
        ncclResult_t result = ncclReduceScatter(send_data, recv_data,
                                                recv_count * elem_size,
                                                ncclChar, ncclSum, comm_, stream_);
        if (result != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclReduceScatter failed");
        }
        return synchronize_blocking("reduce_scatter");
#else
        (void)send_data; (void)recv_data; (void)recv_count; (void)elem_size;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Void-Pointer Reductions (Layer 1 — used by nccl_comm_adapter)
    // ------------------------------------------------------------------------

    /// @brief Reduce with sum to root (double)
    result<void> reduce_sum_impl(const void* sendbuf, void* recvbuf,
                                  size_type count, rank_t root) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(double), "reduce_sum send");
            !buffer_check) {
            return buffer_check;
        }
        if (rank_ == root) {
            if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(double), "reduce_sum recv");
                !buffer_check) {
                return buffer_check;
            }
        }
        ncclResult_t res = ncclReduce(sendbuf, recvbuf, count,
                                       ncclFloat64, ncclSum, root, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclReduce (sum, double) failed");
        }
        return synchronize_blocking("reduce_sum");
#else
        (void)sendbuf; (void)recvbuf; (void)count; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Reduce with sum to root (int)
    result<void> reduce_sum_int_impl(const void* sendbuf, void* recvbuf,
                                      size_type count, rank_t root) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int32_t), "reduce_sum_int send");
            !buffer_check) {
            return buffer_check;
        }
        if (rank_ == root) {
            if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int32_t), "reduce_sum_int recv");
                !buffer_check) {
                return buffer_check;
            }
        }
        ncclResult_t res = ncclReduce(sendbuf, recvbuf, count,
                                       ncclInt32, ncclSum, root, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclReduce (sum, int) failed");
        }
        return synchronize_blocking("reduce_sum_int");
#else
        (void)sendbuf; (void)recvbuf; (void)count; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with sum (double)
    result<void> allreduce_sum_impl(const void* sendbuf, void* recvbuf,
                                     size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(double), "allreduce_sum send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(double), "allreduce_sum recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclFloat64, ncclSum, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (sum, double) failed");
        }
        return synchronize_blocking("allreduce_sum");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with sum (int)
    result<void> allreduce_sum_int_impl(const void* sendbuf, void* recvbuf,
                                         size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int32_t), "allreduce_sum_int send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int32_t), "allreduce_sum_int recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt32, ncclSum, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (sum, int) failed");
        }
        return synchronize_blocking("allreduce_sum_int");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with sum (long / int64)
    result<void> allreduce_sum_long_impl(const void* sendbuf, void* recvbuf,
                                          size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int64_t), "allreduce_sum_long send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int64_t), "allreduce_sum_long recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt64, ncclSum, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (sum, long) failed");
        }
        return synchronize_blocking("allreduce_sum_long");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Extended Typed Reductions (min, max, prod)
    // ------------------------------------------------------------------------

    /// @brief Allreduce with min (double)
    result<void> allreduce_min_impl(const void* sendbuf, void* recvbuf,
                                     size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(double), "allreduce_min send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(double), "allreduce_min recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclFloat64, ncclMin, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (min, double) failed");
        }
        return synchronize_blocking("allreduce_min");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with min (int)
    result<void> allreduce_min_int_impl(const void* sendbuf, void* recvbuf,
                                         size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int32_t), "allreduce_min_int send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int32_t), "allreduce_min_int recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt32, ncclMin, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (min, int) failed");
        }
        return synchronize_blocking("allreduce_min_int");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with min (long / int64)
    result<void> allreduce_min_long_impl(const void* sendbuf, void* recvbuf,
                                          size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int64_t), "allreduce_min_long send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int64_t), "allreduce_min_long recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt64, ncclMin, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (min, long) failed");
        }
        return synchronize_blocking("allreduce_min_long");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with max (double)
    result<void> allreduce_max_impl(const void* sendbuf, void* recvbuf,
                                     size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(double), "allreduce_max send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(double), "allreduce_max recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclFloat64, ncclMax, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (max, double) failed");
        }
        return synchronize_blocking("allreduce_max");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with max (int)
    result<void> allreduce_max_int_impl(const void* sendbuf, void* recvbuf,
                                         size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int32_t), "allreduce_max_int send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int32_t), "allreduce_max_int recv");
            !buffer_check) {
                return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt32, ncclMax, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (max, int) failed");
        }
        return synchronize_blocking("allreduce_max_int");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with max (long / int64)
    result<void> allreduce_max_long_impl(const void* sendbuf, void* recvbuf,
                                          size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int64_t), "allreduce_max_long send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int64_t), "allreduce_max_long recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt64, ncclMax, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (max, long) failed");
        }
        return synchronize_blocking("allreduce_max_long");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with product (double)
    result<void> allreduce_prod_impl(const void* sendbuf, void* recvbuf,
                                      size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(double), "allreduce_prod send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(double), "allreduce_prod recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclFloat64, ncclProd, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (prod, double) failed");
        }
        return synchronize_blocking("allreduce_prod");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with product (int)
    result<void> allreduce_prod_int_impl(const void* sendbuf, void* recvbuf,
                                          size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int32_t), "allreduce_prod_int send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int32_t), "allreduce_prod_int recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt32, ncclProd, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (prod, int) failed");
        }
        return synchronize_blocking("allreduce_prod_int");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Allreduce with product (long / int64)
    result<void> allreduce_prod_long_impl(const void* sendbuf, void* recvbuf,
                                           size_type count) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(sendbuf, count * sizeof(std::int64_t), "allreduce_prod_long send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recvbuf, count * sizeof(std::int64_t), "allreduce_prod_long recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclResult_t res = ncclAllReduce(sendbuf, recvbuf, count,
                                          ncclInt64, ncclProd, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce (prod, long) failed");
        }
        return synchronize_blocking("allreduce_prod_long");
#else
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Logical Reductions
    // ------------------------------------------------------------------------

    /// @brief Allreduce with logical AND
    /// @details NCCL does not provide logical reductions. This operation is
    ///          intentionally unsupported at this layer.
    result<void> allreduce_land_impl(const void* sendbuf, void* recvbuf,
                                      size_type count) {
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL logical allreduce (land) is not supported");
    }

    /// @brief Allreduce with logical OR
    /// @details NCCL does not provide logical reductions. This operation is
    ///          intentionally unsupported at this layer.
    result<void> allreduce_lor_impl(const void* sendbuf, void* recvbuf,
                                     size_type count) {
        (void)sendbuf; (void)recvbuf; (void)count;
        return make_error<void>(status_code::not_supported,
                               "NCCL logical allreduce (lor) is not supported");
    }

    // ------------------------------------------------------------------------
    // Variable-Size Collectives (gatherv, scatterv, allgatherv, alltoallv)
    // ------------------------------------------------------------------------

    /// @brief Variable-size gather using grouped ncclSend/ncclRecv
    result<void> gatherv_impl(const void* sendbuf, size_type sendcount,
                               void* recvbuf, const int* recvcounts,
                               const int* displs, size_type elem_size, rank_t root) {
        (void)sendbuf; (void)sendcount; (void)recvbuf;
        (void)recvcounts; (void)displs; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL variable-size gather is not supported");
    }

    /// @brief Variable-size scatter using grouped ncclSend/ncclRecv
    result<void> scatterv_impl(const void* sendbuf, const int* sendcounts,
                                const int* displs, void* recvbuf,
                                size_type recvcount, size_type elem_size, rank_t root) {
        (void)sendbuf; (void)sendcounts; (void)displs;
        (void)recvbuf; (void)recvcount; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL variable-size scatter is not supported");
    }

    /// @brief Variable-size allgather using grouped ncclSend/ncclRecv
    result<void> allgatherv_impl(const void* sendbuf, size_type sendcount,
                                  void* recvbuf, const int* recvcounts,
                                  const int* displs, size_type elem_size) {
        (void)sendbuf; (void)sendcount; (void)recvbuf;
        (void)recvcounts; (void)displs; (void)elem_size;
        return make_error<void>(status_code::not_supported,
                               "NCCL variable-size allgather is not supported");
    }

    /// @brief Variable-size all-to-all using grouped ncclSend/ncclRecv
    result<void> alltoallv_impl(const void* sendbuf, const int* sendcounts,
                                 const int* sdispls, void* recvbuf,
                                 const int* recvcounts, const int* rdispls,
                                 size_type elem_size) {
        (void)sendbuf; (void)sendcounts; (void)sdispls;
        (void)recvbuf; (void)recvcounts; (void)rdispls; (void)elem_size;
        return make_error<void>(status_code::not_supported,
                               "NCCL variable-size alltoall is not supported");
    }

    // ------------------------------------------------------------------------
    // Typed Broadcast
    // ------------------------------------------------------------------------

    /// @brief Typed broadcast using proper NCCL data type
    template <typename T>
    result<void> broadcast(T* data, size_type count, rank_t root) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(data, count * sizeof(T), "typed broadcast");
            !buffer_check) {
            return buffer_check;
        }
        ncclDataType_t dtype = get_nccl_dtype<T>();
        ncclResult_t res = ncclBroadcast(data, data, count, dtype, root, comm_, stream_);
        if (res != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "ncclBroadcast (typed) failed");
        }
        return synchronize_blocking("typed broadcast");
#else
        (void)data; (void)count; (void)root;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // NCCL-Specific Reductions
    // ------------------------------------------------------------------------

    /// @brief Typed allreduce operation
    /// @tparam T Data type (must be NCCL-compatible)
    /// @param send_buf Send buffer (device memory)
    /// @param recv_buf Receive buffer (device memory)
    /// @param count Number of elements
    /// @param op Reduction operation
    /// @return Success or error
    template <typename T>
    result<void> allreduce(const T* send_buf, T* recv_buf, size_type count,
                           nccl_op op = nccl_op::sum) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(send_buf, count * sizeof(T), "typed allreduce send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(recv_buf, count * sizeof(T), "typed allreduce recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclDataType_t dtype = get_nccl_dtype<T>();
        ncclRedOp_t nccl_op_val = get_nccl_op(op);

        ncclResult_t result = ncclAllReduce(send_buf, recv_buf, count,
                                            dtype, nccl_op_val, comm_, stream_);
        if (result != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclAllReduce failed");
        }
        return synchronize_blocking("typed allreduce");
#else
        (void)send_buf; (void)recv_buf; (void)count; (void)op;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief In-place allreduce
    template <typename T>
    result<void> allreduce_inplace(T* buf, size_type count,
                                   nccl_op op = nccl_op::sum) {
        return allreduce(buf, buf, count, op);
    }

    /// @brief Typed reduce operation
    template <typename T>
    result<void> reduce(const T* send_buf, T* recv_buf, size_type count,
                        rank_t root, nccl_op op = nccl_op::sum) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(send_buf, count * sizeof(T), "typed reduce send");
            !buffer_check) {
            return buffer_check;
        }
        if (rank_ == root) {
            if (auto buffer_check = require_device_buffer(recv_buf, count * sizeof(T), "typed reduce recv");
                !buffer_check) {
                return buffer_check;
            }
        }
        ncclDataType_t dtype = get_nccl_dtype<T>();
        ncclRedOp_t nccl_op_val = get_nccl_op(op);

        ncclResult_t result = ncclReduce(send_buf, recv_buf, count,
                                         dtype, nccl_op_val, root, comm_, stream_);
        if (result != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclReduce failed");
        }
        return synchronize_blocking("typed reduce");
#else
        (void)send_buf; (void)recv_buf; (void)count; (void)root; (void)op;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    /// @brief Reduce-scatter operation
    template <typename T>
    result<void> reduce_scatter(const T* send_buf, T* recv_buf, size_type recv_count,
                                nccl_op op = nccl_op::sum) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        if (!valid()) {
            return make_error<void>(status_code::invalid_state,
                                   "NCCL communicator is not initialized");
        }
        if (auto buffer_check = require_device_buffer(
                send_buf, recv_count * sizeof(T) * static_cast<size_type>(size_),
                "typed reduce_scatter send");
            !buffer_check) {
            return buffer_check;
        }
        if (auto buffer_check = require_device_buffer(
                recv_buf, recv_count * sizeof(T), "typed reduce_scatter recv");
            !buffer_check) {
            return buffer_check;
        }
        ncclDataType_t dtype = get_nccl_dtype<T>();
        ncclRedOp_t nccl_op_val = get_nccl_op(op);

        ncclResult_t result = ncclReduceScatter(send_buf, recv_buf, recv_count,
                                                dtype, nccl_op_val, comm_, stream_);
        if (result != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "ncclReduceScatter failed");
        }
        return synchronize_blocking("typed reduce_scatter");
#else
        (void)send_buf; (void)recv_buf; (void)recv_count; (void)op;
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Stream Management
    // ------------------------------------------------------------------------

#if defined(DTL_ENABLE_CUDA)
    /// @brief Set the CUDA stream for operations
    void set_stream(cudaStream_t stream) noexcept { stream_ = stream; }

    /// @brief Get the current CUDA stream
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
#endif

    /// @brief Synchronize the CUDA stream
    result<void> synchronize() {
#if defined(DTL_ENABLE_CUDA)
        cudaError_t err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaStreamSynchronize failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Lifecycle Management
    // ------------------------------------------------------------------------

    /// @brief Destroy the NCCL communicator
    result<void> destroy() {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
        auto cleanup_result = release_resources(/*destroy_comm=*/true, /*destroy_stream=*/true);
        if (!cleanup_result) {
            return cleanup_result;
        }
        return {};
#else
        return make_error<void>(status_code::not_supported,
                               "NCCL support not enabled");
#endif
    }

#if defined(DTL_ENABLE_NCCL)
    /// @brief Get the native NCCL communicator
    [[nodiscard]] ncclComm_t native_handle() const noexcept { return comm_; }
#endif

private:
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
    [[nodiscard]] result<void> release_resources(bool destroy_comm, bool destroy_stream) noexcept {
        status first_error;
        bool has_error = false;

        if (barrier_scratch_ != nullptr) {
            cudaError_t free_err = cudaFree(barrier_scratch_);
            barrier_scratch_ = nullptr;
            if (free_err != cudaSuccess && !has_error) {
                has_error = true;
                first_error = status{status_code::backend_error, no_rank,
                                     "cudaFree failed for NCCL barrier scratch"};
            }
        }

        if (destroy_stream && owns_stream_ && stream_ != nullptr) {
            cudaError_t destroy_err = cudaStreamDestroy(stream_);
            stream_ = nullptr;
            owns_stream_ = false;
            if (destroy_err != cudaSuccess && !has_error) {
                has_error = true;
                first_error = status{status_code::backend_error, no_rank,
                                     "cudaStreamDestroy failed for NCCL stream"};
            }
        }

        if (destroy_comm && comm_ != nullptr) {
            ncclResult_t comm_err = ncclCommDestroy(comm_);
            comm_ = nullptr;
            if (comm_err != ncclSuccess && !has_error) {
                has_error = true;
                first_error = status{status_code::backend_error, no_rank,
                                     "ncclCommDestroy failed"};
            }
        }

        if (has_error) {
            return result<void>::failure(first_error);
        }
        return {};
    }
#endif

#if defined(DTL_ENABLE_CUDA)
    [[nodiscard]] result<void> require_device_buffer(const void* ptr,
                                                     size_type bytes,
                                                     std::string_view operation) const {
        if (bytes == 0) {
            return {};
        }
        if (ptr == nullptr) {
            return make_error<void>(
                status_code::invalid_argument,
                std::string("NCCL ") + std::string(operation) +
                    " requires a non-null CUDA device buffer");
        }

        cudaPointerAttributes attrs{};
        cudaError_t attr_err = cudaPointerGetAttributes(&attrs, ptr);
        if (attr_err != cudaSuccess) {
            cudaGetLastError();
            return make_error<void>(
                status_code::invalid_argument,
                std::string("NCCL ") + std::string(operation) +
                    " requires CUDA device memory");
        }
#if CUDART_VERSION >= 10000
        if (attrs.type != cudaMemoryTypeDevice) {
#else
        if (attrs.memoryType != cudaMemoryTypeDevice) {
#endif
            return make_error<void>(
                status_code::invalid_argument,
                std::string("NCCL ") + std::string(operation) +
                    " requires CUDA device memory");
        }
        return {};
    }

    [[nodiscard]] result<void> synchronize_blocking(std::string_view operation) {
        cudaError_t cuda_err = cudaStreamSynchronize(stream_);
        if (cuda_err != cudaSuccess) {
            return make_error<void>(
                status_code::backend_error,
                std::string("cudaStreamSynchronize failed after NCCL ") +
                    std::string(operation));
        }
        return {};
    }
#endif

#if defined(DTL_ENABLE_NCCL)
    template <typename T>
    static ncclDataType_t get_nccl_dtype() {
        if constexpr (std::is_same_v<T, int8_t>) return ncclInt8;
        else if constexpr (std::is_same_v<T, uint8_t>) return ncclUint8;
        else if constexpr (std::is_same_v<T, int32_t>) return ncclInt32;
        else if constexpr (std::is_same_v<T, uint32_t>) return ncclUint32;
        else if constexpr (std::is_same_v<T, int64_t>) return ncclInt64;
        else if constexpr (std::is_same_v<T, uint64_t>) return ncclUint64;
        else if constexpr (std::is_same_v<T, float>) return ncclFloat32;
        else if constexpr (std::is_same_v<T, double>) return ncclFloat64;
        else return ncclChar;  // Fallback for unknown types
    }

    static ncclRedOp_t get_nccl_op(nccl_op op) {
        switch (op) {
            case nccl_op::sum: return ncclSum;
            case nccl_op::prod: return ncclProd;
            case nccl_op::max: return ncclMax;
            case nccl_op::min: return ncclMin;
            case nccl_op::avg: return ncclAvg;
            default: return ncclSum;
        }
    }

    ncclComm_t comm_ = nullptr;
#endif

#if defined(DTL_ENABLE_CUDA)
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;
    int* barrier_scratch_ = nullptr;  ///< Persistent scratch buffer for barrier
#endif

    rank_t rank_ = no_rank;
    rank_t size_ = 0;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Initialize NCCL communicators for all GPUs
/// @param num_gpus Number of GPUs
/// @param gpu_ranks Mapping from GPU index to global rank
/// @return Vector of communicators or error
[[nodiscard]] inline result<std::vector<std::unique_ptr<nccl_communicator>>>
create_nccl_communicators([[maybe_unused]] int num_gpus,
                          [[maybe_unused]] const std::vector<rank_t>& gpu_ranks) {
#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)
    ncclUniqueId id;
    ncclGetUniqueId(&id);

    std::vector<ncclComm_t> comms(num_gpus);
    ncclResult_t result = ncclCommInitAll(comms.data(), num_gpus, nullptr);
    if (result != ncclSuccess) {
        return make_error<std::vector<std::unique_ptr<nccl_communicator>>>(
            status_code::nccl_error, "ncclCommInitAll failed");
    }

    std::vector<std::unique_ptr<nccl_communicator>> communicators;
    communicators.reserve(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        cudaStream_t stream;
        cudaSetDevice(i);
        cudaStreamCreate(&stream);

        communicators.push_back(std::make_unique<nccl_communicator>(
            comms[i], gpu_ranks[i], static_cast<rank_t>(num_gpus), stream));
    }

    return communicators;
#else
    return make_error<std::vector<std::unique_ptr<nccl_communicator>>>(
        status_code::not_supported, "NCCL support not enabled");
#endif
}

#if defined(DTL_ENABLE_NCCL) && defined(DTL_ENABLE_CUDA)

/// @brief Create an NCCL communicator from a unique ID
/// @details This is the low-level factory used by nccl_domain::from_mpi.
///          Each rank must call this with the same unique_id (broadcast from rank 0)
///          and its own rank/size within the NCCL communicator group.
/// @param unique_id NCCL unique ID (broadcast from rank 0)
/// @param rank This rank's position in the NCCL communicator
/// @param size Total number of ranks in the NCCL communicator
/// @param device_id CUDA device to use for this rank
/// @return Shared pointer to the new communicator or error
[[nodiscard]] inline result<std::shared_ptr<nccl_communicator>>
create_communicator_from_unique_id(const ncclUniqueId& unique_id,
                                   rank_t rank, rank_t size, int device_id) {
    // Set the CUDA device for this rank
    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        return make_error<std::shared_ptr<nccl_communicator>>(
            status_code::backend_error,
            "cudaSetDevice failed in NCCL communicator creation");
    }

    // Create a CUDA stream for NCCL operations
    cudaStream_t stream = nullptr;
    cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
        return make_error<std::shared_ptr<nccl_communicator>>(
            status_code::backend_error,
            "cudaStreamCreate failed in NCCL communicator creation");
    }

    // Initialize the NCCL communicator
    ncclComm_t comm = nullptr;
    ncclResult_t nccl_err = ncclCommInitRank(&comm, static_cast<int>(size),
                                              unique_id, static_cast<int>(rank));
    if (nccl_err != ncclSuccess) {
        cudaStreamDestroy(stream);
        return make_error<std::shared_ptr<nccl_communicator>>(
            status_code::nccl_error,
            "ncclCommInitRank failed");
    }

    // Create the wrapper with a custom deleter for the stream
    auto communicator = std::make_shared<nccl_communicator>(comm, rank, size, stream);
    return communicator;
}

/// @brief Get a new NCCL unique ID (typically called only on rank 0)
/// @return The unique ID or error
[[nodiscard]] inline result<ncclUniqueId> get_unique_id() {
    ncclUniqueId id;
    ncclResult_t err = ncclGetUniqueId(&id);
    if (err != ncclSuccess) {
        return make_error<ncclUniqueId>(status_code::backend_error,
                                        "ncclGetUniqueId failed");
    }
    return id;
}

#endif  // DTL_ENABLE_NCCL && DTL_ENABLE_CUDA

}  // namespace nccl
}  // namespace dtl
