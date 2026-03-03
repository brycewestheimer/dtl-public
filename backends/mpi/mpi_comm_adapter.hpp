// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_comm_adapter.hpp
/// @brief Concept-compliant MPI communicator adapter
/// @details Wraps mpi_communicator with void-returning methods that throw on error.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/error/status.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/communication/message_status.hpp>
#include <backends/mpi/mpi_communicator.hpp>

#include <stdexcept>
#include <memory>
#include <type_traits>
#include <utility>

namespace dtl {
namespace mpi {

// ============================================================================
// Communication Error Exception
// ============================================================================

/// @brief Exception thrown when MPI communication fails
class communication_error : public std::runtime_error {
public:
    /// @brief Construct with message
    explicit communication_error(const std::string& msg)
        : std::runtime_error(msg) {}

    /// @brief Construct with status
    explicit communication_error(const status& s)
        : std::runtime_error(s.to_string()) {}
};

// ============================================================================
// MPI Communicator Adapter
// ============================================================================

/// @brief Concept-compliant adapter for mpi_communicator
/// @details Provides void-returning methods that throw communication_error on failure.
///          Satisfies Communicator, CollectiveCommunicator, and ReducingCommunicator concepts.
class mpi_comm_adapter {
public:
    using size_type = dtl::size_type;

    /// @brief Construct adapter wrapping world communicator
    mpi_comm_adapter()
        : impl_(&world_communicator()) {}

    /// @brief Construct adapter wrapping specific communicator
    /// @param comm Reference to mpi_communicator (must outlive adapter)
    explicit mpi_comm_adapter(mpi_communicator& comm)
        : impl_(&comm) {}

    /// @brief Construct adapter owning a communicator shared_ptr
    /// @param comm Shared communicator instance
    explicit mpi_comm_adapter(std::shared_ptr<mpi_communicator> comm)
        : impl_(comm.get())
        , owned_impl_(std::move(comm)) {}

    // ------------------------------------------------------------------------
    // Query Operations (Communicator concept)
    // ------------------------------------------------------------------------

    /// @brief Get this process's rank
    [[nodiscard]] rank_t rank() const noexcept {
        return impl_->rank();
    }

    /// @brief Get total number of ranks
    [[nodiscard]] rank_t size() const noexcept {
        return impl_->size();
    }

    // ------------------------------------------------------------------------
    // Point-to-Point Communication (Communicator concept)
    // ------------------------------------------------------------------------

    /// @brief Blocking send
    /// @param buf Buffer to send
    /// @param count Number of bytes
    /// @param dest Destination rank
    /// @param tag Message tag
    /// @throws communication_error on MPI failure
    void send(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->send_impl(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("MPI send failed to rank " + std::to_string(dest));
        }
    }

    /// @brief Blocking receive
    /// @param buf Buffer to receive into
    /// @param count Number of bytes
    /// @param source Source rank
    /// @param tag Message tag
    /// @throws communication_error on MPI failure
    void recv(void* buf, size_type count, rank_t source, int tag) {
        auto result = impl_->recv_impl(buf, count, 1, source, tag);
        if (!result) {
            throw communication_error("MPI recv failed from rank " + std::to_string(source));
        }
    }

    /// @brief Non-blocking send
    /// @param buf Buffer to send
    /// @param count Number of bytes
    /// @param dest Destination rank
    /// @param tag Message tag
    /// @return Request handle for completion tracking
    /// @throws communication_error on MPI failure
    [[nodiscard]] request_handle isend(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->isend_impl(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("MPI isend failed to rank " + std::to_string(dest));
        }
        return *result;
    }

    /// @brief Non-blocking receive
    /// @param buf Buffer to receive into
    /// @param count Number of bytes
    /// @param source Source rank
    /// @param tag Message tag
    /// @return Request handle for completion tracking
    /// @throws communication_error on MPI failure
    [[nodiscard]] request_handle irecv(void* buf, size_type count, rank_t source, int tag) {
        auto result = impl_->irecv_impl(buf, count, 1, source, tag);
        if (!result) {
            throw communication_error("MPI irecv failed from rank " + std::to_string(source));
        }
        return *result;
    }

    /// @brief Wait for non-blocking operation to complete
    /// @param req Request handle to wait on
    /// @throws communication_error on MPI failure
    void wait(request_handle& req) {
        auto result = impl_->wait_impl(req);
        if (!result) {
            throw communication_error("MPI wait failed");
        }
    }

    /// @brief Test if non-blocking operation completed
    /// @param req Request handle to test
    /// @return true if operation completed
    /// @throws communication_error on MPI failure
    [[nodiscard]] bool test(request_handle& req) {
        auto result = impl_->test_impl(req);
        if (!result) {
            throw communication_error("MPI test failed");
        }
        return *result;
    }

    /// @brief Wait for any non-blocking operation to complete
    /// @param requests Array of request handles
    /// @param count Number of requests
    /// @return Index of completed request
    /// @throws communication_error on MPI failure
    size_type waitany(request_handle* requests, size_type count) {
        auto result = impl_->waitany_impl(requests, count);
        if (!result) {
            throw communication_error("MPI waitany failed");
        }
        return *result;
    }

    /// @brief Send-receive with in-place replace
    /// @param buf Buffer for both send and receive
    /// @param count Number of bytes
    /// @param dest Destination rank
    /// @param sendtag Send tag
    /// @param source Source rank
    /// @param recvtag Receive tag
    /// @throws communication_error on MPI failure
    void sendrecv_replace(void* buf, size_type count,
                           rank_t dest, int sendtag,
                           rank_t source, int recvtag) {
        auto result = impl_->sendrecv_replace_impl(buf, count, dest, sendtag, source, recvtag);
        if (!result) {
            throw communication_error("MPI sendrecv_replace failed");
        }
    }

    // ------------------------------------------------------------------------
    // Send Mode Variants
    // ------------------------------------------------------------------------

    /// @brief Synchronous blocking send (MPI_Ssend)
    /// @details Completes only when the matching receive has begun.
    /// @param buf Buffer to send
    /// @param count Number of bytes
    /// @param dest Destination rank
    /// @param tag Message tag
    /// @throws communication_error on MPI failure
    void ssend(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->ssend_impl(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("MPI ssend failed to rank " + std::to_string(dest));
        }
    }

    /// @brief Ready-mode blocking send (MPI_Rsend)
    /// @details Caller guarantees that matching receive is already posted.
    /// @param buf Buffer to send
    /// @param count Number of bytes
    /// @param dest Destination rank
    /// @param tag Message tag
    /// @throws communication_error on MPI failure
    void rsend(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->rsend_impl(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("MPI rsend failed to rank " + std::to_string(dest));
        }
    }

    /// @brief Non-blocking synchronous send (MPI_Issend)
    /// @details Non-blocking version of synchronous send.
    /// @param buf Buffer to send
    /// @param count Number of bytes
    /// @param dest Destination rank
    /// @param tag Message tag
    /// @return Request handle for completion tracking
    /// @throws communication_error on MPI failure
    [[nodiscard]] request_handle issend(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->issend_impl(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("MPI issend failed to rank " + std::to_string(dest));
        }
        return *result;
    }

    /// @brief Non-blocking ready-mode send (MPI_Irsend)
    /// @details Non-blocking version of ready-mode send.
    /// @param buf Buffer to send
    /// @param count Number of bytes
    /// @param dest Destination rank
    /// @param tag Message tag
    /// @return Request handle for completion tracking
    /// @throws communication_error on MPI failure
    [[nodiscard]] request_handle irsend(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->irsend_impl(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("MPI irsend failed to rank " + std::to_string(dest));
        }
        return *result;
    }

    // ------------------------------------------------------------------------
    // Probe Operations (Phase 5 / V1.2.2 — Task 5.2)
    // ------------------------------------------------------------------------

    /// @brief Blocking probe for incoming message
    /// @details Blocks until a matching message is available, returns its status.
    /// @param source Source rank (or any_source for wildcard)
    /// @param tag Message tag (or any_tag for wildcard)
    /// @return Message status information
    /// @throws communication_error on MPI failure
    message_status probe(rank_t source, int tag) {
        auto result = impl_->probe_impl(source, tag);
        if (!result) {
            throw communication_error("MPI probe failed");
        }
        return *result;
    }

    /// @brief Non-blocking probe for incoming message
    /// @details Tests if a matching message is available without blocking.
    /// @param source Source rank (or any_source for wildcard)
    /// @param tag Message tag (or any_tag for wildcard)
    /// @return Pair {message_available, status}. If no message, status is default-initialized.
    /// @throws communication_error on MPI failure
    std::pair<bool, message_status> iprobe(rank_t source, int tag) {
        auto result = impl_->iprobe_impl(source, tag);
        if (!result) {
            throw communication_error("MPI iprobe failed");
        }
        return *result;
    }

    // ------------------------------------------------------------------------
    // Collective Operations (CollectiveCommunicator concept)
    // ------------------------------------------------------------------------

    /// @brief Barrier synchronization
    /// @throws communication_error on MPI failure
    void barrier() {
        auto result = impl_->barrier();
        if (!result) {
            throw communication_error("MPI barrier failed");
        }
    }

    /// @brief Broadcast data from root to all ranks
    /// @param buf Buffer (input at root, output at others)
    /// @param count Number of bytes
    /// @param root Root rank
    /// @throws communication_error on MPI failure
    void broadcast(void* buf, size_type count, rank_t root) {
        auto result = impl_->broadcast_impl(buf, count, 1, root);
        if (!result) {
            throw communication_error("MPI broadcast failed");
        }
    }

    /// @brief Scatter data from root to all ranks
    /// @param sendbuf Send buffer (significant only at root)
    /// @param recvbuf Receive buffer
    /// @param count Number of bytes per rank
    /// @param root Root rank
    /// @throws communication_error on MPI failure
    void scatter(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->scatter_impl(sendbuf, count, recvbuf, count, 1, root);
        if (!result) {
            throw communication_error("MPI scatter failed");
        }
    }

    /// @brief Gather data from all ranks to root
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer (significant only at root)
    /// @param count Number of bytes per rank
    /// @param root Root rank
    /// @throws communication_error on MPI failure
    void gather(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->gather_impl(sendbuf, count, recvbuf, count, 1, root);
        if (!result) {
            throw communication_error("MPI gather failed");
        }
    }

    /// @brief Gather data from all ranks to all ranks
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer
    /// @param count Number of bytes per rank
    /// @throws communication_error on MPI failure
    void allgather(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allgather_impl(sendbuf, count, recvbuf, count, 1);
        if (!result) {
            throw communication_error("MPI allgather failed");
        }
    }

    /// @brief All-to-all exchange
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer
    /// @param count Number of bytes per rank
    /// @throws communication_error on MPI failure
    void alltoall(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->alltoall_impl(sendbuf, recvbuf, count, 1);
        if (!result) {
            throw communication_error("MPI alltoall failed");
        }
    }

    // ------------------------------------------------------------------------
    // Reduction Operations (ReducingCommunicator concept)
    // ------------------------------------------------------------------------

    /// @brief Reduce with sum operation
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer (significant only at root)
    /// @param count Number of elements (double)
    /// @param root Root rank
    /// @throws communication_error on MPI failure
    void reduce_sum(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->reduce_sum_impl(sendbuf, recvbuf, count, root);
        if (!result) {
            throw communication_error("MPI reduce_sum failed");
        }
    }

    /// @brief Allreduce with sum operation (double)
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer
    /// @param count Number of elements (double)
    /// @throws communication_error on MPI failure
    void allreduce_sum(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_sum_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_sum failed");
        }
    }

    /// @brief Allreduce with sum operation (int)
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer
    /// @param count Number of elements (int)
    /// @throws communication_error on MPI failure
    void allreduce_sum_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_sum_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_sum_int failed");
        }
    }

    /// @brief Allreduce with sum operation (long)
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer
    /// @param count Number of elements (long)
    /// @throws communication_error on MPI failure
    void allreduce_sum_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_sum_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_sum_long failed");
        }
    }

    /// @brief Reduce to root with sum operation (int)
    /// @param sendbuf Send buffer
    /// @param recvbuf Receive buffer (significant only at root)
    /// @param count Number of elements (int)
    /// @param root Root rank
    /// @throws communication_error on MPI failure
    void reduce_sum_int(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->reduce_sum_int_impl(sendbuf, recvbuf, count, root);
        if (!result) {
            throw communication_error("MPI reduce_sum_int failed");
        }
    }

    /// @brief Type-dispatched allreduce sum (template)
    /// @tparam T Value type (int, long, double supported)
    /// @param local Local value to contribute
    /// @return Global sum across all ranks
    template <typename T>
    T allreduce_sum_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_sum_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_sum_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_sum(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            // Convert float to double for MPI
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            // Other integral types: convert via long
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_sum_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            // Floating point types: convert via double
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Extended Reduction Operations (min, max, prod, logical)
    // ------------------------------------------------------------------------

    /// @brief Allreduce with min operation (int)
    void allreduce_min_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_min_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_min_int failed");
        }
    }

    /// @brief Allreduce with min operation (long)
    void allreduce_min_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_min_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_min_long failed");
        }
    }

    /// @brief Allreduce with min operation (double)
    void allreduce_min(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_min_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_min failed");
        }
    }

    /// @brief Allreduce with max operation (int)
    void allreduce_max_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_max_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_max_int failed");
        }
    }

    /// @brief Allreduce with max operation (long)
    void allreduce_max_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_max_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_max_long failed");
        }
    }

    /// @brief Allreduce with max operation (double)
    void allreduce_max(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_max_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_max failed");
        }
    }

    /// @brief Allreduce with product operation (int)
    void allreduce_prod_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_prod_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_prod_int failed");
        }
    }

    /// @brief Allreduce with product operation (long)
    void allreduce_prod_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_prod_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_prod_long failed");
        }
    }

    /// @brief Allreduce with product operation (double)
    void allreduce_prod(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_prod_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_prod failed");
        }
    }

    /// @brief Allreduce with logical AND operation
    void allreduce_land(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_land_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_land failed");
        }
    }

    /// @brief Allreduce with logical OR operation
    void allreduce_lor(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_lor_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI allreduce_lor failed");
        }
    }

    /// @brief Type-dispatched allreduce min (template)
    /// @tparam T Value type (int, long, double supported)
    /// @param local Local value to contribute
    /// @return Global min across all ranks
    template <typename T>
    T allreduce_min_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_min_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_min_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_min(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_min(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_min_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_min(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    /// @brief Type-dispatched allreduce max (template)
    /// @tparam T Value type (int, long, double supported)
    /// @param local Local value to contribute
    /// @return Global max across all ranks
    template <typename T>
    T allreduce_max_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_max_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_max_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_max(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_max(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_max_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_max(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    /// @brief Type-dispatched allreduce product (template)
    /// @tparam T Value type (int, long, double supported)
    /// @param local Local value to contribute
    /// @return Global product across all ranks
    template <typename T>
    T allreduce_prod_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_prod_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_prod_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_prod(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_prod(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_prod_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_prod(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    /// @brief Type-dispatched allreduce logical AND (template)
    /// @param local Local boolean value
    /// @return Global logical AND across all ranks
    bool allreduce_land_value(bool local) {
        int i_local = local ? 1 : 0;
        int i_result{};
        allreduce_land(&i_local, &i_result, 1);
        return i_result != 0;
    }

    /// @brief Type-dispatched allreduce logical OR (template)
    /// @param local Local boolean value
    /// @return Global logical OR across all ranks
    bool allreduce_lor_value(bool local) {
        int i_local = local ? 1 : 0;
        int i_result{};
        allreduce_lor(&i_local, &i_result, 1);
        return i_result != 0;
    }

    /// @brief Type-dispatched reduce to root (template)
    /// @tparam T Value type
    /// @param local Local value to contribute
    /// @param root Root rank that receives the result
    /// @return Global sum (valid only on root rank)
    template <typename T>
    T reduce_sum_to_root(T local, rank_t root) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            reduce_sum_int(&local, &result, 1, root);
        } else if constexpr (std::is_same_v<T, double>) {
            reduce_sum(&local, &result, 1, root);
        } else if constexpr (std::is_integral_v<T>) {
            int i_local = static_cast<int>(local);
            int i_result{};
            reduce_sum_int(&i_local, &i_result, 1, root);
            result = static_cast<T>(i_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            reduce_sum(&d_local, &d_result, 1, root);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Variable-Size Collective Operations (V1.1)
    // ------------------------------------------------------------------------

    /// @brief Variable-size gather (gatherv)
    /// @param sendbuf Local data to send
    /// @param sendcount Number of elements to send
    /// @param recvbuf Receive buffer (significant at root)
    /// @param recvcounts Array of counts from each rank (at root)
    /// @param displs Displacements in recvbuf (at root)
    /// @param elem_size Element size in bytes
    /// @param root Root rank
    /// @throws communication_error on MPI failure
    void gatherv(const void* sendbuf, size_type sendcount,
                 void* recvbuf, const int* recvcounts, const int* displs,
                 size_type elem_size, rank_t root) {
        auto result = impl_->gatherv_impl(sendbuf, sendcount, recvbuf,
                                          recvcounts, displs, elem_size, root);
        if (!result) {
            throw communication_error("MPI gatherv failed");
        }
    }

    /// @brief Variable-size scatter (scatterv)
    /// @param sendbuf Send buffer (significant at root)
    /// @param sendcounts Array of counts to each rank (at root)
    /// @param displs Displacements in sendbuf (at root)
    /// @param recvbuf Local receive buffer
    /// @param recvcount Number of elements to receive
    /// @param elem_size Element size in bytes
    /// @param root Root rank
    /// @throws communication_error on MPI failure
    void scatterv(const void* sendbuf, const int* sendcounts, const int* displs,
                  void* recvbuf, size_type recvcount, size_type elem_size, rank_t root) {
        auto result = impl_->scatterv_impl(sendbuf, sendcounts, displs,
                                           recvbuf, recvcount, elem_size, root);
        if (!result) {
            throw communication_error("MPI scatterv failed");
        }
    }

    /// @brief Variable-size allgather (allgatherv)
    /// @param sendbuf Local data to send
    /// @param sendcount Number of elements to send
    /// @param recvbuf Receive buffer
    /// @param recvcounts Array of counts from each rank
    /// @param displs Displacements in recvbuf
    /// @param elem_size Element size in bytes
    /// @throws communication_error on MPI failure
    void allgatherv(const void* sendbuf, size_type sendcount,
                    void* recvbuf, const int* recvcounts, const int* displs,
                    size_type elem_size) {
        auto result = impl_->allgatherv_impl(sendbuf, sendcount, recvbuf,
                                             recvcounts, displs, elem_size);
        if (!result) {
            throw communication_error("MPI allgatherv failed");
        }
    }

    /// @brief Variable-size all-to-all (alltoallv) - CRITICAL for redistribute()
    /// @param sendbuf Send buffer
    /// @param sendcounts Array of counts to send to each rank
    /// @param sdispls Displacements in sendbuf for each rank
    /// @param recvbuf Receive buffer
    /// @param recvcounts Array of counts to receive from each rank
    /// @param rdispls Displacements in recvbuf for each rank
    /// @param elem_size Element size in bytes
    /// @throws communication_error on MPI failure
    void alltoallv(const void* sendbuf, const int* sendcounts, const int* sdispls,
                   void* recvbuf, const int* recvcounts, const int* rdispls,
                   size_type elem_size) {
        auto result = impl_->alltoallv_impl(sendbuf, sendcounts, sdispls,
                                            recvbuf, recvcounts, rdispls, elem_size);
        if (!result) {
            throw communication_error("MPI alltoallv failed");
        }
    }

    // ------------------------------------------------------------------------
    // Scan Operations (V1.1)
    // ------------------------------------------------------------------------

    /// @brief Inclusive prefix sum (scan) - int
    void scan_sum_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->scan_sum_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI scan_sum_int failed");
        }
    }

    /// @brief Inclusive prefix sum (scan) - long
    void scan_sum_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->scan_sum_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI scan_sum_long failed");
        }
    }

    /// @brief Inclusive prefix sum (scan) - double
    void scan_sum(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->scan_sum_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI scan_sum failed");
        }
    }

    /// @brief Exclusive prefix sum (exscan) - int
    /// @note On rank 0, recvbuf is undefined after this call
    void exscan_sum_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->exscan_sum_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI exscan_sum_int failed");
        }
    }

    /// @brief Exclusive prefix sum (exscan) - long
    void exscan_sum_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->exscan_sum_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI exscan_sum_long failed");
        }
    }

    /// @brief Exclusive prefix sum (exscan) - double
    void exscan_sum(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->exscan_sum_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("MPI exscan_sum failed");
        }
    }

    /// @brief Type-dispatched inclusive scan (template)
    template <typename T>
    T scan_sum_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            scan_sum_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            scan_sum_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            scan_sum(&local, &result, 1);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            scan_sum_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            scan_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    /// @brief Type-dispatched exclusive scan (template)
    /// @note Returns 0 on rank 0
    template <typename T>
    T exscan_sum_value(T local) {
        T result{};
        if (rank() == 0) {
            // MPI_Exscan leaves rank 0 result undefined; we define it as 0
            result = T{};
            // Still need to participate in the collective
            if constexpr (std::is_same_v<T, int>) {
                exscan_sum_int(&local, &result, 1);
            } else if constexpr (std::is_same_v<T, long>) {
                exscan_sum_long(&local, &result, 1);
            } else if constexpr (std::is_same_v<T, double>) {
                exscan_sum(&local, &result, 1);
            } else if constexpr (std::is_integral_v<T>) {
                long l_local = static_cast<long>(local);
                long l_result{};
                exscan_sum_long(&l_local, &l_result, 1);
            } else {
                double d_local = static_cast<double>(local);
                double d_result{};
                exscan_sum(&d_local, &d_result, 1);
            }
            return T{};  // Rank 0 gets 0
        }

        if constexpr (std::is_same_v<T, int>) {
            exscan_sum_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            exscan_sum_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            exscan_sum(&local, &result, 1);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            exscan_sum_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            exscan_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------------

    /// @brief Check if this is the root rank
    [[nodiscard]] bool is_root() const noexcept {
        return rank() == 0;
    }

    /// @brief Get underlying mpi_communicator
    [[nodiscard]] mpi_communicator& underlying() noexcept {
        return *impl_;
    }

    /// @brief Get underlying mpi_communicator (const)
    [[nodiscard]] const mpi_communicator& underlying() const noexcept {
        return *impl_;
    }

    // ------------------------------------------------------------------------
    // Communicator Management
    // ------------------------------------------------------------------------

#if DTL_ENABLE_MPI
    /// @brief Split communicator by color
    /// @param color Color for grouping (ranks with same color in same group)
    /// @param key Ordering key within color group (default 0)
    /// @return New adapter wrapping the split communicator
    /// @throws communication_error on MPI failure
    [[nodiscard]] mpi_comm_adapter split(int color, int key = 0) {
        auto result = impl_->split_by_color(color, key);
        if (!result) {
            throw communication_error("MPI communicator split failed");
        }
        return mpi_comm_adapter(std::move(*result));
    }
#endif

private:
    /// @brief Construct adapter owning a communicator (used by split)
    explicit mpi_comm_adapter(std::unique_ptr<mpi_communicator> owned)
        : impl_(owned.get())
        , owned_impl_(std::move(owned)) {}

    mpi_communicator* impl_;
    std::shared_ptr<mpi_communicator> owned_impl_;  // For split-created adapters
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Communicator<mpi_comm_adapter>,
              "mpi_comm_adapter must satisfy Communicator concept");
static_assert(CollectiveCommunicator<mpi_comm_adapter>,
              "mpi_comm_adapter must satisfy CollectiveCommunicator concept");
static_assert(ReducingCommunicator<mpi_comm_adapter>,
              "mpi_comm_adapter must satisfy ReducingCommunicator concept");

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the world communicator adapter
/// @return Adapter wrapping MPI_COMM_WORLD
[[nodiscard]] inline mpi_comm_adapter world_adapter() {
    return mpi_comm_adapter{};
}

}  // namespace mpi
}  // namespace dtl
