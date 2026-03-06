// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_communicator.hpp
/// @brief MPI communicator satisfying Communicator concept
/// @details Provides MPI-based point-to-point and collective communication.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/reduction_ops.hpp>
#include <dtl/communication/message_status.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#include <atomic>
#include <memory>
#include <vector>
#include <utility>

namespace dtl {
namespace mpi {

// ============================================================================
// MPI Communicator
// ============================================================================

/// @brief MPI-based communicator implementation
/// @details Wraps an MPI_Comm and provides DTL communicator interface.
///          Provides internal implementation with result<> return types.
class mpi_communicator {
public:
    using size_type = dtl::size_type;

    /// @brief Default constructor (wraps MPI_COMM_WORLD)
    mpi_communicator() = default;

#if DTL_ENABLE_MPI
    /// @brief Construct from MPI communicator
    /// @param comm MPI communicator handle
    /// @param owns_comm Whether this object owns the communicator
    explicit mpi_communicator(MPI_Comm comm, bool owns_comm = false)
        : comm_(comm)
        , owns_comm_(owns_comm) {
        if (comm_ != MPI_COMM_NULL) {
            // MPI_Comm_rank/size are invalid before MPI_Init*.
            // This can happen if a static world communicator is constructed
            // before the program explicitly initializes MPI.
            int initialized = 0;
            int finalized = 0;
            MPI_Initialized(&initialized);
            if (initialized) {
                MPI_Finalized(&finalized);
            }
            if (initialized && !finalized) {
                MPI_Comm_rank(comm_, &rank_);
                MPI_Comm_size(comm_, &size_);
            }
        }
    }
#endif

    /// @brief Destructor (frees owned communicator)
    ~mpi_communicator() {
#if DTL_ENABLE_MPI
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owns_comm_ && comm_ != MPI_COMM_NULL &&
            comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_SELF && !finalized) {
            MPI_Comm_free(&comm_);
        }
#endif
    }

    // Non-copyable
    mpi_communicator(const mpi_communicator&) = delete;
    mpi_communicator& operator=(const mpi_communicator&) = delete;

    // Movable
    mpi_communicator(mpi_communicator&& other) noexcept
        : rank_(other.rank_)
        , size_(other.size_)
#if DTL_ENABLE_MPI
        , comm_(other.comm_)
        , owns_comm_(other.owns_comm_)
#endif
    {
#if DTL_ENABLE_MPI
        other.comm_ = MPI_COMM_NULL;
        other.owns_comm_ = false;
#endif
    }

    mpi_communicator& operator=(mpi_communicator&& other) noexcept {
        if (this != &other) {
#if DTL_ENABLE_MPI
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (owns_comm_ && comm_ != MPI_COMM_NULL && !finalized) {
                MPI_Comm_free(&comm_);
            }
            comm_ = other.comm_;
            owns_comm_ = other.owns_comm_;
            other.comm_ = MPI_COMM_NULL;
            other.owns_comm_ = false;
#endif
            rank_ = other.rank_;
            size_ = other.size_;
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Communicator Interface
    // ------------------------------------------------------------------------

    /// @brief Get this process's rank
    [[nodiscard]] rank_t rank() const noexcept { return rank_; }

    /// @brief Get total number of ranks
    [[nodiscard]] rank_t size() const noexcept { return size_; }

    /// @brief Check if communicator is valid
    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_MPI
        return comm_ != MPI_COMM_NULL;
#else
        return false;
#endif
    }

    /// @brief Get communicator properties
    [[nodiscard]] communicator_properties properties() const noexcept {
        return communicator_properties{
            .size = size_,
            .rank = rank_,
            .is_inter = false,
            .is_derived = false,
            .name = "mpi"
        };
    }

    // ------------------------------------------------------------------------
    // Point-to-Point Communication
    // ------------------------------------------------------------------------

    /// @brief Blocking send
    result<void> send_impl(const void* data, size_type count,
                          size_type elem_size, rank_t dest, int tag) {
#if DTL_ENABLE_MPI
        int result = MPI_Send(data, static_cast<int>(count * elem_size),
                              MPI_BYTE, dest, tag, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Send failed");
        }
        return {};
#else
        (void)data; (void)count; (void)elem_size; (void)dest; (void)tag;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Blocking receive
    result<void> recv_impl(void* data, size_type count,
                          size_type elem_size, rank_t source, int tag) {
#if DTL_ENABLE_MPI
        MPI_Status status;
        int result = MPI_Recv(data, static_cast<int>(count * elem_size),
                              MPI_BYTE, source, tag, comm_, &status);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Recv failed");
        }
        return {};
#else
        (void)data; (void)count; (void)elem_size; (void)source; (void)tag;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Collective Communication
    // ------------------------------------------------------------------------

    /// @brief Barrier synchronization
    result<void> barrier() {
#if DTL_ENABLE_MPI
        int result = MPI_Barrier(comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Barrier failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Broadcast
    result<void> broadcast_impl(void* data, size_type count,
                               size_type elem_size, rank_t root) {
#if DTL_ENABLE_MPI
        int result = MPI_Bcast(data, static_cast<int>(count * elem_size),
                               MPI_BYTE, root, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Bcast failed");
        }
        return {};
#else
        (void)data; (void)count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Gather
    result<void> gather_impl(const void* send_data, size_type send_count,
                            void* recv_data, size_type recv_count,
                            size_type elem_size, rank_t root) {
#if DTL_ENABLE_MPI
        int result = MPI_Gather(send_data, static_cast<int>(send_count * elem_size), MPI_BYTE,
                                recv_data, static_cast<int>(recv_count * elem_size), MPI_BYTE,
                                root, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Gather failed");
        }
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Scatter
    result<void> scatter_impl(const void* send_data, size_type send_count,
                             void* recv_data, size_type recv_count,
                             size_type elem_size, rank_t root) {
#if DTL_ENABLE_MPI
        int result = MPI_Scatter(send_data, static_cast<int>(send_count * elem_size), MPI_BYTE,
                                 recv_data, static_cast<int>(recv_count * elem_size), MPI_BYTE,
                                 root, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Scatter failed");
        }
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allgather
    result<void> allgather_impl(const void* send_data, size_type send_count,
                               void* recv_data, size_type recv_count,
                               size_type elem_size) {
#if DTL_ENABLE_MPI
        int result = MPI_Allgather(send_data, static_cast<int>(send_count * elem_size), MPI_BYTE,
                                   recv_data, static_cast<int>(recv_count * elem_size), MPI_BYTE,
                                   comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allgather failed");
        }
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Alltoall
    result<void> alltoall_impl(const void* send_data, void* recv_data,
                               size_type count, size_type elem_size) {
#if DTL_ENABLE_MPI
        int result = MPI_Alltoall(send_data, static_cast<int>(count * elem_size), MPI_BYTE,
                                  recv_data, static_cast<int>(count * elem_size), MPI_BYTE,
                                  comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Alltoall failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count; (void)elem_size;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Non-blocking Communication
    // ------------------------------------------------------------------------

    /// @brief Non-blocking send
    result<request_handle> isend_impl(const void* data, size_type count,
                                      size_type elem_size, rank_t dest, int tag) {
#if DTL_ENABLE_MPI
        auto* req = new MPI_Request;
        int result = MPI_Isend(data, static_cast<int>(count * elem_size),
                               MPI_BYTE, dest, tag, comm_, req);
        if (result != MPI_SUCCESS) {
            delete req;
            return make_error<request_handle>(status_code::communication_error,
                                              "MPI_Isend failed");
        }
        return request_handle{static_cast<void*>(req)};
#else
        (void)data; (void)count; (void)elem_size; (void)dest; (void)tag;
        return make_error<request_handle>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Non-blocking receive
    result<request_handle> irecv_impl(void* data, size_type count,
                                      size_type elem_size, rank_t source, int tag) {
#if DTL_ENABLE_MPI
        auto* req = new MPI_Request;
        int result = MPI_Irecv(data, static_cast<int>(count * elem_size),
                               MPI_BYTE, source, tag, comm_, req);
        if (result != MPI_SUCCESS) {
            delete req;
            return make_error<request_handle>(status_code::communication_error,
                                              "MPI_Irecv failed");
        }
        return request_handle{static_cast<void*>(req)};
#else
        (void)data; (void)count; (void)elem_size; (void)source; (void)tag;
        return make_error<request_handle>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Wait for non-blocking operation to complete
    result<void> wait_impl(request_handle& req) {
#if DTL_ENABLE_MPI
        if (!req.valid()) {
            return make_error<void>(status_code::invalid_argument, "Invalid request handle");
        }
        auto* mpi_req = static_cast<MPI_Request*>(req.handle);
        MPI_Status status;
        int result = MPI_Wait(mpi_req, &status);
        delete mpi_req;
        req.handle = nullptr;
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Wait failed");
        }
        return {};
#else
        (void)req;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Test if non-blocking operation completed
    result<bool> test_impl(request_handle& req) {
#if DTL_ENABLE_MPI
        if (!req.valid()) {
            return true;  // Invalid request considered complete
        }
        auto* mpi_req = static_cast<MPI_Request*>(req.handle);
        int flag = 0;
        MPI_Status status;
        int result = MPI_Test(mpi_req, &flag, &status);
        if (result != MPI_SUCCESS) {
            return make_error<bool>(status_code::communication_error,
                                    "MPI_Test failed");
        }
        if (flag) {
            delete mpi_req;
            req.handle = nullptr;
        }
        return flag != 0;
#else
        (void)req;
        return make_error<bool>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Reduction Operations
    // ------------------------------------------------------------------------

    /// @brief Reduce with sum operation
    result<void> reduce_sum_impl(const void* send_data, void* recv_data,
                                 size_type count, rank_t root) {
#if DTL_ENABLE_MPI
        int result = MPI_Reduce(send_data, recv_data, static_cast<int>(count),
                                MPI_DOUBLE, MPI_SUM, root, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Reduce failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count; (void)root;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with sum operation (double)
    result<void> allreduce_sum_impl(const void* send_data, void* recv_data,
                                    size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_DOUBLE, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with sum operation (int)
    result<void> allreduce_sum_int_impl(const void* send_data, void* recv_data,
                                        size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_INT, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with sum operation (long)
    result<void> allreduce_sum_long_impl(const void* send_data, void* recv_data,
                                         size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_LONG, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Reduce to root with sum operation (int)
    result<void> reduce_sum_int_impl(const void* send_data, void* recv_data,
                                     size_type count, rank_t root) {
#if DTL_ENABLE_MPI
        int result = MPI_Reduce(send_data, recv_data, static_cast<int>(count),
                                MPI_INT, MPI_SUM, root, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Reduce failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count; (void)root;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Extended Reduction Operations (min, max, prod)
    // ------------------------------------------------------------------------

    /// @brief Allreduce with min operation (int)
    result<void> allreduce_min_int_impl(const void* send_data, void* recv_data,
                                        size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_INT, MPI_MIN, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (MIN) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with min operation (long)
    result<void> allreduce_min_long_impl(const void* send_data, void* recv_data,
                                         size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_LONG, MPI_MIN, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (MIN) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with min operation (double)
    result<void> allreduce_min_impl(const void* send_data, void* recv_data,
                                    size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_DOUBLE, MPI_MIN, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (MIN) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with max operation (int)
    result<void> allreduce_max_int_impl(const void* send_data, void* recv_data,
                                        size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_INT, MPI_MAX, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (MAX) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with max operation (long)
    result<void> allreduce_max_long_impl(const void* send_data, void* recv_data,
                                         size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_LONG, MPI_MAX, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (MAX) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with max operation (double)
    result<void> allreduce_max_impl(const void* send_data, void* recv_data,
                                    size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_DOUBLE, MPI_MAX, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (MAX) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with product operation (int)
    result<void> allreduce_prod_int_impl(const void* send_data, void* recv_data,
                                         size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_INT, MPI_PROD, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (PROD) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with product operation (long)
    result<void> allreduce_prod_long_impl(const void* send_data, void* recv_data,
                                          size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_LONG, MPI_PROD, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (PROD) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with product operation (double)
    result<void> allreduce_prod_impl(const void* send_data, void* recv_data,
                                     size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_DOUBLE, MPI_PROD, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (PROD) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with logical AND operation (int as bool)
    result<void> allreduce_land_impl(const void* send_data, void* recv_data,
                                     size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_INT, MPI_LAND, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (LAND) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allreduce with logical OR operation (int as bool)
    result<void> allreduce_lor_impl(const void* send_data, void* recv_data,
                                    size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Allreduce(send_data, recv_data, static_cast<int>(count),
                                   MPI_INT, MPI_LOR, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allreduce (LOR) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Variable-Size Collective Operations (V1.1)
    // ------------------------------------------------------------------------

    /// @brief Variable-size gather (gatherv)
    /// @param send_data Local data to send
    /// @param send_count Number of bytes to send from this rank
    /// @param recv_data Receive buffer (significant only at root)
    /// @param recv_counts Array of counts to receive from each rank (at root)
    /// @param displs Array of displacements in recv_data (at root)
    /// @param elem_size Element size in bytes
    /// @param root Root rank
    result<void> gatherv_impl(const void* send_data, size_type send_count,
                              void* recv_data, const int* recv_counts,
                              const int* displs, size_type elem_size, rank_t root) {
#if DTL_ENABLE_MPI
        // Convert element counts and displacements to byte counts for MPI_BYTE
        std::vector<int> byte_recv_counts(static_cast<size_t>(size_));
        std::vector<int> byte_displs(static_cast<size_t>(size_));

        for (int i = 0; i < size_; ++i) {
            byte_recv_counts[static_cast<size_t>(i)] =
                recv_counts[i] * static_cast<int>(elem_size);
            byte_displs[static_cast<size_t>(i)] =
                displs[i] * static_cast<int>(elem_size);
        }

        int result = MPI_Gatherv(
            send_data, static_cast<int>(send_count * elem_size), MPI_BYTE,
            recv_data, byte_recv_counts.data(), byte_displs.data(), MPI_BYTE,
            root, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Gatherv failed");
        }
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_counts; (void)displs; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Variable-size scatter (scatterv)
    /// @param send_data Send buffer (significant only at root)
    /// @param send_counts Array of counts to send to each rank (at root)
    /// @param displs Array of displacements in send_data (at root)
    /// @param recv_data Local receive buffer
    /// @param recv_count Number of bytes to receive at this rank
    /// @param elem_size Element size in bytes
    /// @param root Root rank
    result<void> scatterv_impl(const void* send_data, const int* send_counts,
                               const int* displs, void* recv_data,
                               size_type recv_count, size_type elem_size, rank_t root) {
#if DTL_ENABLE_MPI
        // Convert element counts and displacements to byte counts for MPI_BYTE
        std::vector<int> byte_send_counts(static_cast<size_t>(size_));
        std::vector<int> byte_displs(static_cast<size_t>(size_));

        for (int i = 0; i < size_; ++i) {
            byte_send_counts[static_cast<size_t>(i)] =
                send_counts[i] * static_cast<int>(elem_size);
            byte_displs[static_cast<size_t>(i)] =
                displs[i] * static_cast<int>(elem_size);
        }

        int result = MPI_Scatterv(
            send_data, byte_send_counts.data(), byte_displs.data(), MPI_BYTE,
            recv_data, static_cast<int>(recv_count * elem_size), MPI_BYTE,
            root, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Scatterv failed");
        }
        return {};
#else
        (void)send_data; (void)send_counts; (void)displs;
        (void)recv_data; (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Variable-size allgather (allgatherv)
    /// @param send_data Local data to send
    /// @param send_count Number of elements to send from this rank
    /// @param recv_data Receive buffer (all ranks)
    /// @param recv_counts Array of element counts to receive from each rank
    /// @param displs Array of element displacements in recv_data
    /// @param elem_size Element size in bytes
    result<void> allgatherv_impl(const void* send_data, size_type send_count,
                                 void* recv_data, const int* recv_counts,
                                 const int* displs, size_type elem_size) {
#if DTL_ENABLE_MPI
        // Convert element counts and displacements to byte counts for MPI_BYTE
        // (same pattern as alltoallv_impl)
        std::vector<int> byte_recv_counts(static_cast<size_t>(size_));
        std::vector<int> byte_displs(static_cast<size_t>(size_));

        for (int i = 0; i < size_; ++i) {
            byte_recv_counts[static_cast<size_t>(i)] =
                recv_counts[i] * static_cast<int>(elem_size);
            byte_displs[static_cast<size_t>(i)] =
                displs[i] * static_cast<int>(elem_size);
        }

        int result = MPI_Allgatherv(
            send_data, static_cast<int>(send_count * elem_size), MPI_BYTE,
            recv_data, byte_recv_counts.data(), byte_displs.data(), MPI_BYTE,
            comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Allgatherv failed");
        }
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_counts; (void)displs; (void)elem_size;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Variable-size all-to-all (alltoallv) - CRITICAL for redistribute()
    /// @param send_data Send buffer
    /// @param send_counts Array of counts to send to each rank
    /// @param send_displs Array of displacements in send_data
    /// @param recv_data Receive buffer
    /// @param recv_counts Array of counts to receive from each rank
    /// @param recv_displs Array of displacements in recv_data
    /// @param elem_size Element size in bytes
    result<void> alltoallv_impl(const void* send_data, const int* send_counts,
                                const int* send_displs, void* recv_data,
                                const int* recv_counts, const int* recv_displs,
                                size_type elem_size) {
#if DTL_ENABLE_MPI
        // For byte-level transfer, we need to multiply counts and displs by elem_size
        // However, MPI_Alltoallv expects counts already in bytes when using MPI_BYTE
        // The caller should provide counts in elements; we convert here
        const auto comm_size = static_cast<size_t>(size_);
        std::vector<int> byte_send_counts(comm_size);
        std::vector<int> byte_send_displs(comm_size);
        std::vector<int> byte_recv_counts(comm_size);
        std::vector<int> byte_recv_displs(comm_size);

        for (rank_t i = 0; i < size_; ++i) {
            const auto idx = static_cast<size_t>(i);
            byte_send_counts[idx] = send_counts[i] * static_cast<int>(elem_size);
            byte_send_displs[idx] = send_displs[i] * static_cast<int>(elem_size);
            byte_recv_counts[idx] = recv_counts[i] * static_cast<int>(elem_size);
            byte_recv_displs[idx] = recv_displs[i] * static_cast<int>(elem_size);
        }

        int result = MPI_Alltoallv(
            send_data, byte_send_counts.data(), byte_send_displs.data(), MPI_BYTE,
            recv_data, byte_recv_counts.data(), byte_recv_displs.data(), MPI_BYTE,
            comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Alltoallv failed");
        }
        return {};
#else
        (void)send_data; (void)send_counts; (void)send_displs;
        (void)recv_data; (void)recv_counts; (void)recv_displs; (void)elem_size;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Scan Operations (V1.1)
    // ------------------------------------------------------------------------

    /// @brief Inclusive prefix sum (scan) - int
    result<void> scan_sum_int_impl(const void* send_data, void* recv_data,
                                   size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Scan(send_data, recv_data, static_cast<int>(count),
                              MPI_INT, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Scan (SUM) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Inclusive prefix sum (scan) - long
    result<void> scan_sum_long_impl(const void* send_data, void* recv_data,
                                    size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Scan(send_data, recv_data, static_cast<int>(count),
                              MPI_LONG, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Scan (SUM) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Inclusive prefix sum (scan) - double
    result<void> scan_sum_impl(const void* send_data, void* recv_data,
                               size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Scan(send_data, recv_data, static_cast<int>(count),
                              MPI_DOUBLE, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Scan (SUM) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Exclusive prefix sum (exscan) - int
    result<void> exscan_sum_int_impl(const void* send_data, void* recv_data,
                                     size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Exscan(send_data, recv_data, static_cast<int>(count),
                                MPI_INT, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Exscan (SUM) failed");
        }
        // Note: On rank 0, recv_data is undefined after MPI_Exscan
        // The caller should handle this case
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Exclusive prefix sum (exscan) - long
    result<void> exscan_sum_long_impl(const void* send_data, void* recv_data,
                                      size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Exscan(send_data, recv_data, static_cast<int>(count),
                                MPI_LONG, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Exscan (SUM) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Exclusive prefix sum (exscan) - double
    result<void> exscan_sum_impl(const void* send_data, void* recv_data,
                                 size_type count) {
#if DTL_ENABLE_MPI
        int result = MPI_Exscan(send_data, recv_data, static_cast<int>(count),
                                MPI_DOUBLE, MPI_SUM, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Exscan (SUM) failed");
        }
        return {};
#else
        (void)send_data; (void)recv_data; (void)count;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Send Mode Variants (Phase 12 / V1.2 — Task 12.10)
    // ------------------------------------------------------------------------

    /// @brief Synchronous blocking send (MPI_Ssend)
    /// @details Completes only when the matching receive has begun.
    result<void> ssend_impl(const void* data, size_type count,
                            size_type elem_size, rank_t dest, int tag) {
#if DTL_ENABLE_MPI
        int result = MPI_Ssend(data, static_cast<int>(count * elem_size),
                               MPI_BYTE, dest, tag, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Ssend failed");
        }
        return {};
#else
        (void)data; (void)count; (void)elem_size; (void)dest; (void)tag;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Ready-mode blocking send (MPI_Rsend)
    /// @details Caller guarantees that matching receive is already posted.
    result<void> rsend_impl(const void* data, size_type count,
                            size_type elem_size, rank_t dest, int tag) {
#if DTL_ENABLE_MPI
        int result = MPI_Rsend(data, static_cast<int>(count * elem_size),
                               MPI_BYTE, dest, tag, comm_);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Rsend failed");
        }
        return {};
#else
        (void)data; (void)count; (void)elem_size; (void)dest; (void)tag;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Non-blocking synchronous send (MPI_Issend)
    result<request_handle> issend_impl(const void* data, size_type count,
                                       size_type elem_size, rank_t dest, int tag) {
#if DTL_ENABLE_MPI
        auto* req = new MPI_Request;
        int result = MPI_Issend(data, static_cast<int>(count * elem_size),
                                MPI_BYTE, dest, tag, comm_, req);
        if (result != MPI_SUCCESS) {
            delete req;
            return make_error<request_handle>(status_code::communication_error,
                                              "MPI_Issend failed");
        }
        return request_handle{static_cast<void*>(req)};
#else
        (void)data; (void)count; (void)elem_size; (void)dest; (void)tag;
        return make_error<request_handle>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Non-blocking ready-mode send (MPI_Irsend)
    result<request_handle> irsend_impl(const void* data, size_type count,
                                       size_type elem_size, rank_t dest, int tag) {
#if DTL_ENABLE_MPI
        auto* req = new MPI_Request;
        int result = MPI_Irsend(data, static_cast<int>(count * elem_size),
                                MPI_BYTE, dest, tag, comm_, req);
        if (result != MPI_SUCCESS) {
            delete req;
            return make_error<request_handle>(status_code::communication_error,
                                              "MPI_Irsend failed");
        }
        return request_handle{static_cast<void*>(req)};
#else
        (void)data; (void)count; (void)elem_size; (void)dest; (void)tag;
        return make_error<request_handle>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Probe Operations (Phase 5 / V1.2.2 — Task 5.2)
    // ------------------------------------------------------------------------

    /// @brief Blocking probe for incoming message (MPI_Probe)
    /// @details Blocks until a matching message is available, then returns its status.
    /// @param source Source rank (or any_source for wildcard)
    /// @param tag Message tag (or any_tag for wildcard)
    /// @return Result containing message status
    result<message_status> probe_impl(rank_t source, int tag) {
#if DTL_ENABLE_MPI
        MPI_Status mpi_status;
        int result = MPI_Probe(source, tag, comm_, &mpi_status);
        if (result != MPI_SUCCESS) {
            return make_error<message_status>(status_code::communication_error,
                                              "MPI_Probe failed");
        }
        int count = 0;
        MPI_Get_count(&mpi_status, MPI_BYTE, &count);
        message_status ms;
        ms.source = mpi_status.MPI_SOURCE;
        ms.tag = mpi_status.MPI_TAG;
        ms.count = static_cast<size_type>(count);
        ms.cancelled = false;
        ms.error = mpi_status.MPI_ERROR;
        return ms;
#else
        (void)source; (void)tag;
        return make_error<message_status>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Non-blocking probe for incoming message (MPI_Iprobe)
    /// @details Tests if a matching message is available without blocking.
    /// @param source Source rank (or any_source for wildcard)
    /// @param tag Message tag (or any_tag for wildcard)
    /// @return Result containing {message_available, status}. If no message, returns {false, {}}
    result<std::pair<bool, message_status>> iprobe_impl(rank_t source, int tag) {
#if DTL_ENABLE_MPI
        int flag = 0;
        MPI_Status mpi_status;
        int result = MPI_Iprobe(source, tag, comm_, &flag, &mpi_status);
        if (result != MPI_SUCCESS) {
            return make_error<std::pair<bool, message_status>>(
                status_code::communication_error, "MPI_Iprobe failed");
        }
        if (!flag) {
            return std::pair<bool, message_status>{false, message_status{}};
        }
        int count = 0;
        MPI_Get_count(&mpi_status, MPI_BYTE, &count);
        message_status ms;
        ms.source = mpi_status.MPI_SOURCE;
        ms.tag = mpi_status.MPI_TAG;
        ms.count = static_cast<size_type>(count);
        ms.cancelled = false;
        ms.error = mpi_status.MPI_ERROR;
        return std::pair<bool, message_status>{true, ms};
#else
        (void)source; (void)tag;
        return make_error<std::pair<bool, message_status>>(
            status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Request Management
    // ------------------------------------------------------------------------

    /// @brief Wait for any request to complete (MPI_Waitany)
    /// @param requests Array of MPI_Request handles (as request_handle)
    /// @param count Number of requests
    /// @return Index of completed request
    result<size_type> waitany_impl(request_handle* requests, size_type count) {
#if DTL_ENABLE_MPI
        if (count == 0) {
            return make_error<size_type>(status_code::invalid_argument,
                                         "No requests to wait on");
        }
        std::vector<MPI_Request> mpi_reqs(count);
        for (size_type i = 0; i < count; ++i) {
            if (requests[i].valid()) {
                mpi_reqs[i] = *static_cast<MPI_Request*>(requests[i].handle);
            } else {
                mpi_reqs[i] = MPI_REQUEST_NULL;
            }
        }
        int index = MPI_UNDEFINED;
        MPI_Status status;
        int result = MPI_Waitany(static_cast<int>(count), mpi_reqs.data(), &index, &status);
        if (result != MPI_SUCCESS) {
            return make_error<size_type>(status_code::communication_error,
                                         "MPI_Waitany failed");
        }
        if (index == MPI_UNDEFINED) {
            return make_error<size_type>(status_code::invalid_argument,
                                         "All requests were MPI_REQUEST_NULL");
        }
        // Clean up the completed request
        auto idx = static_cast<size_type>(index);
        if (requests[idx].valid()) {
            delete static_cast<MPI_Request*>(requests[idx].handle);
            requests[idx].handle = nullptr;
        }
        return idx;
#else
        (void)requests; (void)count;
        return make_error<size_type>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Send-receive with replace (MPI_Sendrecv_replace)
    result<void> sendrecv_replace_impl(void* buf, size_type count,
                                        rank_t dest, int sendtag,
                                        rank_t source, int recvtag) {
#if DTL_ENABLE_MPI
        MPI_Status status;
        int result = MPI_Sendrecv_replace(buf, static_cast<int>(count),
                                           MPI_BYTE, dest, sendtag,
                                           source, recvtag, comm_, &status);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Sendrecv_replace failed");
        }
        return {};
#else
        (void)buf; (void)count; (void)dest; (void)sendtag;
        (void)source; (void)recvtag;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // MPI-Specific Methods
    // ------------------------------------------------------------------------

#if DTL_ENABLE_MPI
    /// @brief Get the underlying MPI communicator
    [[nodiscard]] MPI_Comm native_handle() const noexcept { return comm_; }

    /// @brief Create a sub-communicator from a subset of ranks
    /// @param ranks World ranks to include
    /// @return New communicator or error
    result<std::unique_ptr<mpi_communicator>> split(const std::vector<rank_t>& ranks) {
        // Create MPI group from current communicator
        MPI_Group world_group, new_group;
        MPI_Comm_group(comm_, &world_group);

        std::vector<int> int_ranks(ranks.begin(), ranks.end());
        MPI_Group_incl(world_group, static_cast<int>(int_ranks.size()),
                       int_ranks.data(), &new_group);

        MPI_Comm new_comm;
        MPI_Comm_create(comm_, new_group, &new_comm);

        MPI_Group_free(&world_group);
        MPI_Group_free(&new_group);

        if (new_comm == MPI_COMM_NULL) {
            return std::unique_ptr<mpi_communicator>(nullptr);
        }

        return std::make_unique<mpi_communicator>(new_comm, true);
    }

    /// @brief Split communicator by color
    /// @param color Color for grouping (ranks with same color in same comm)
    /// @param key Ordering key within color
    /// @return New communicator or error
    result<std::unique_ptr<mpi_communicator>> split_by_color(int color, int key) {
        MPI_Comm new_comm;
        int result = MPI_Comm_split(comm_, color, key, &new_comm);
        if (result != MPI_SUCCESS) {
            return make_error<std::unique_ptr<mpi_communicator>>(
                status_code::communication_error, "MPI_Comm_split failed");
        }
        return std::make_unique<mpi_communicator>(new_comm, true);
    }
#endif

private:
    rank_t rank_ = no_rank;
    rank_t size_ = 0;
#if DTL_ENABLE_MPI
    MPI_Comm comm_ = MPI_COMM_NULL;
    bool owns_comm_ = false;
#endif
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the world communicator (wraps MPI_COMM_WORLD)
/// @return World communicator
/// @details Uses lazy re-initialization to handle the case where the static
///          singleton is first accessed before MPI_Init_thread() completes.
[[nodiscard]] inline mpi_communicator& world_communicator() {
#if DTL_ENABLE_MPI
    static mpi_communicator world_comm(MPI_COMM_WORLD, false);
    static std::atomic<bool> reinit_checked{false};
    if (!reinit_checked) {
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Finalized(&finalized);
        }
        if (initialized && !finalized) {
            if (world_comm.size() <= 1) {
                world_comm = mpi_communicator(MPI_COMM_WORLD, false);
            }
            reinit_checked = true;
        }
    }
    return world_comm;
#else
    static mpi_communicator world_comm{};
    return world_comm;
#endif
}

/// @brief Get a self communicator (wraps MPI_COMM_SELF)
/// @return Self communicator
/// @details Uses lazy re-initialization to handle the case where the static
///          singleton is first accessed before MPI_Init_thread() completes.
[[nodiscard]] inline mpi_communicator& self_communicator() {
#if DTL_ENABLE_MPI
    static mpi_communicator self_comm(MPI_COMM_SELF, false);
    static std::atomic<bool> reinit_checked{false};
    if (!reinit_checked) {
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Finalized(&finalized);
        }
        if (initialized && !finalized) {
            self_comm = mpi_communicator(MPI_COMM_SELF, false);
            reinit_checked = true;
        }
    }
    return self_comm;
#else
    static mpi_communicator self_comm{};
    return self_comm;
#endif
}

}  // namespace mpi
}  // namespace dtl
