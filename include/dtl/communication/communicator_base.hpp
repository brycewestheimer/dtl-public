// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file communicator_base.hpp
/// @brief Base communicator interface and utilities
/// @details Provides common base types for communicator implementations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/communication/message_status.hpp>

#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include <utility>

namespace dtl {

// ============================================================================
// Communicator Properties
// ============================================================================

/// @brief Properties describing a communicator
struct communicator_properties {
    /// @brief Number of ranks in the communicator
    rank_t size = 1;

    /// @brief This process's rank
    rank_t rank = 0;

    /// @brief Whether communicator is inter-communicator
    bool is_inter = false;

    /// @brief Whether communicator is a duplicate/derived
    bool is_derived = false;

    /// @brief Name of the communicator (if set)
    const char* name = nullptr;
};

// ============================================================================
// Communicator Base
// ============================================================================

/// @brief Abstract base class for communicators
/// @details Provides common interface for all communicator implementations.
class communicator_base {
public:
    using size_type = dtl::size_type;

    /// @brief Virtual destructor
    virtual ~communicator_base() = default;

    /// @brief Get this process's rank
    [[nodiscard]] virtual rank_t rank() const noexcept = 0;

    /// @brief Get number of ranks
    [[nodiscard]] virtual rank_t size() const noexcept = 0;

    /// @brief Get communicator properties
    [[nodiscard]] virtual communicator_properties properties() const noexcept = 0;

    /// @brief Check if this is the root rank (rank 0)
    [[nodiscard]] bool is_root() const noexcept {
        return rank() == 0;
    }

    /// @brief Check if this rank is the specified rank
    [[nodiscard]] bool is_rank(rank_t r) const noexcept {
        return rank() == r;
    }

protected:
    communicator_base() = default;
    communicator_base(const communicator_base&) = default;
    communicator_base& operator=(const communicator_base&) = default;
    communicator_base(communicator_base&&) = default;
    communicator_base& operator=(communicator_base&&) = default;
};

// ============================================================================
// Null Communicator
// ============================================================================

/// @brief Null communicator for single-process execution
/// @details Always rank 0 of size 1, all operations are no-ops.
class null_communicator {
public:
    using size_type = dtl::size_type;

    /// @brief Get this process's rank (always 0)
    [[nodiscard]] static constexpr rank_t rank() noexcept {
        return 0;
    }

    /// @brief Get number of ranks (always 1)
    [[nodiscard]] static constexpr rank_t size() noexcept {
        return 1;
    }

    /// @brief Blocking send (no-op in single process)
    void send(const void* /*buf*/, size_type /*count*/,
              rank_t /*dest*/, int /*tag*/) noexcept {
        // No-op: can't send to self in null communicator
    }

    /// @brief Blocking receive (no-op)
    void recv(void* /*buf*/, size_type /*count*/,
              rank_t /*source*/, int /*tag*/) noexcept {
        // No-op
    }

    /// @brief Non-blocking send
    [[nodiscard]] request_handle isend(const void* /*buf*/, size_type /*count*/,
                                       rank_t /*dest*/, int /*tag*/) noexcept {
        return request_handle{};
    }

    /// @brief Non-blocking receive
    [[nodiscard]] request_handle irecv(void* /*buf*/, size_type /*count*/,
                                       rank_t /*source*/, int /*tag*/) noexcept {
        return request_handle{};
    }

    /// @brief Wait for request (no-op)
    void wait(request_handle& /*req*/) noexcept {}

    /// @brief Test request completion (always true)
    [[nodiscard]] static bool test(request_handle& /*req*/) noexcept {
        return true;
    }

    /// @brief Synchronous send (delegates to send for single process)
    void ssend(const void* /*buf*/, size_type /*count*/,
               rank_t /*dest*/, int /*tag*/) noexcept {
        // No-op: delegates to send() semantics for single process
    }

    /// @brief Ready-mode send (delegates to send for single process)
    void rsend(const void* /*buf*/, size_type /*count*/,
               rank_t /*dest*/, int /*tag*/) noexcept {
        // No-op: delegates to send() semantics for single process
    }

    /// @brief Non-blocking synchronous send (delegates to isend for single process)
    [[nodiscard]] request_handle issend(const void* /*buf*/, size_type /*count*/,
                                        rank_t /*dest*/, int /*tag*/) noexcept {
        return request_handle{};
    }

    /// @brief Non-blocking ready-mode send (delegates to isend for single process)
    [[nodiscard]] request_handle irsend(const void* /*buf*/, size_type /*count*/,
                                        rank_t /*dest*/, int /*tag*/) noexcept {
        return request_handle{};
    }

    /// @brief Blocking probe for incoming message (single-rank: returns default status)
    /// @details In single-rank execution, no messages are ever pending from other ranks.
    /// @param source Source rank (ignored)
    /// @param tag Message tag (ignored)
    /// @return Default message status (source=0, tag=0, count=0)
    message_status probe([[maybe_unused]] rank_t source, [[maybe_unused]] int tag) noexcept {
        message_status status;
        status.source = 0;
        status.tag = 0;
        status.count = 0;
        return status;
    }

    /// @brief Non-blocking probe for incoming message (single-rank: always returns false)
    /// @details In single-rank execution, no messages are ever pending from other ranks.
    /// @param source Source rank (ignored)
    /// @param tag Message tag (ignored)
    /// @return {false, {}} - no message ever pending
    std::pair<bool, message_status> iprobe([[maybe_unused]] rank_t source,
                                           [[maybe_unused]] int tag) noexcept {
        return std::pair<bool, message_status>{false, message_status{}};
    }

    /// @brief Wait for any request (always returns 0 for single process)
    size_type waitany(request_handle* /*requests*/, size_type count) noexcept {
        (void)count;
        return 0;
    }

    /// @brief Send-receive replace (no-op for single process)
    void sendrecv_replace(void* /*buf*/, size_type /*count*/,
                           rank_t /*dest*/, int /*sendtag*/,
                           rank_t /*source*/, int /*recvtag*/) noexcept {
        // No-op: single-rank, data stays in buffer
    }

    /// @brief Barrier (no-op for single process)
    void barrier() noexcept {}

    /// @brief Broadcast (no-op for single process)
    void broadcast(void* /*buf*/, size_type /*count*/, rank_t /*root*/) noexcept {}

    /// @brief Scatter (copies data for single process)
    void scatter(const void* send_buf, void* recv_buf,
                 size_type count, rank_t /*root*/) noexcept {
        std::memcpy(recv_buf, send_buf, count);
    }

    /// @brief Gather (copies data for single process)
    void gather(const void* send_buf, void* recv_buf,
                size_type count, rank_t /*root*/) noexcept {
        std::memcpy(recv_buf, send_buf, count);
    }

    /// @brief Allgather (copies data for single process)
    void allgather(const void* send_buf, void* recv_buf,
                   size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count);
    }

    /// @brief Alltoall (copies data for single process)
    void alltoall(const void* send_buf, void* recv_buf,
                  size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count);
    }

    /// @brief Reduce sum (copies data for single process)
    /// @note Uses element size parameter to handle any arithmetic type correctly.
    void reduce_sum(const void* send_buf, void* recv_buf,
                    size_type count, rank_t /*root*/, size_type elem_size) noexcept {
        std::memcpy(recv_buf, send_buf, count * elem_size);
    }

    /// @brief Reduce sum (copies data for single process, double — legacy overload)
    void reduce_sum(const void* send_buf, void* recv_buf,
                    size_type count, rank_t /*root*/) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(double));
    }

    /// @brief Allreduce sum (copies data for single process)
    /// @note Uses element size parameter to handle any arithmetic type correctly.
    void allreduce_sum(const void* send_buf, void* recv_buf,
                       size_type count, size_type elem_size) noexcept {
        std::memcpy(recv_buf, send_buf, count * elem_size);
    }

    /// @brief Allreduce sum (copies data for single process, double — legacy overload)
    void allreduce_sum(const void* send_buf, void* recv_buf,
                       size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(double));
    }

    /// @brief Allreduce sum (int)
    void allreduce_sum_int(const void* send_buf, void* recv_buf,
                           size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce sum (long)
    void allreduce_sum_long(const void* send_buf, void* recv_buf,
                            size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    /// @brief Reduce sum to root (int)
    void reduce_sum_int(const void* send_buf, void* recv_buf,
                        size_type count, rank_t /*root*/) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    // ------------------------------------------------------------------------
    // Extended Reduction Operations (min, max, prod, logical)
    // ------------------------------------------------------------------------

    /// @brief Allreduce min (int)
    void allreduce_min_int(const void* send_buf, void* recv_buf,
                           size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce min (long)
    void allreduce_min_long(const void* send_buf, void* recv_buf,
                            size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    /// @brief Allreduce min (double)
    void allreduce_min(const void* send_buf, void* recv_buf,
                       size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(double));
    }

    /// @brief Allreduce max (int)
    void allreduce_max_int(const void* send_buf, void* recv_buf,
                           size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce max (long)
    void allreduce_max_long(const void* send_buf, void* recv_buf,
                            size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    /// @brief Allreduce max (double)
    void allreduce_max(const void* send_buf, void* recv_buf,
                       size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(double));
    }

    /// @brief Allreduce product (int)
    void allreduce_prod_int(const void* send_buf, void* recv_buf,
                            size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce product (long)
    void allreduce_prod_long(const void* send_buf, void* recv_buf,
                             size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    /// @brief Allreduce product (double)
    void allreduce_prod(const void* send_buf, void* recv_buf,
                        size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(double));
    }

    /// @brief Allreduce logical AND
    void allreduce_land(const void* send_buf, void* recv_buf,
                        size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce logical OR
    void allreduce_lor(const void* send_buf, void* recv_buf,
                       size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    // ------------------------------------------------------------------------
    // Bitwise Reduction Operations
    // ------------------------------------------------------------------------

    /// @brief Allreduce bitwise AND (int)
    void allreduce_band_int(const void* send_buf, void* recv_buf,
                            size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce bitwise AND (long)
    void allreduce_band_long(const void* send_buf, void* recv_buf,
                             size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    /// @brief Allreduce bitwise OR (int)
    void allreduce_bor_int(const void* send_buf, void* recv_buf,
                           size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce bitwise OR (long)
    void allreduce_bor_long(const void* send_buf, void* recv_buf,
                            size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    /// @brief Allreduce bitwise XOR (int)
    void allreduce_bxor_int(const void* send_buf, void* recv_buf,
                            size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Allreduce bitwise XOR (long)
    void allreduce_bxor_long(const void* send_buf, void* recv_buf,
                             size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    // ------------------------------------------------------------------------
    // Template Convenience Reduction Methods
    // ------------------------------------------------------------------------

    /// @brief Type-dispatched allreduce sum (single-rank identity)
    template <typename T>
    T allreduce_sum_value(T value) noexcept { return value; }

    /// @brief Type-dispatched allreduce min (single-rank identity)
    template <typename T>
    T allreduce_min_value(T value) noexcept { return value; }

    /// @brief Type-dispatched allreduce max (single-rank identity)
    template <typename T>
    T allreduce_max_value(T value) noexcept { return value; }

    /// @brief Type-dispatched allreduce product (single-rank identity)
    template <typename T>
    T allreduce_prod_value(T value) noexcept { return value; }

    /// @brief Type-dispatched allreduce logical AND (single-rank identity)
    bool allreduce_land_value(bool value) noexcept { return value; }

    /// @brief Type-dispatched allreduce logical OR (single-rank identity)
    bool allreduce_lor_value(bool value) noexcept { return value; }

    /// @brief Type-dispatched reduce to root (single-rank identity)
    template <typename T>
    T reduce_sum_to_root(T value, rank_t /*root*/) noexcept { return value; }

    // ------------------------------------------------------------------------
    // Scan Operations
    // ------------------------------------------------------------------------

    /// @brief Inclusive prefix sum (int) — single-rank identity
    void scan_sum_int(const void* send_buf, void* recv_buf,
                      size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(int));
    }

    /// @brief Inclusive prefix sum (long) — single-rank identity
    void scan_sum_long(const void* send_buf, void* recv_buf,
                       size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(long));
    }

    /// @brief Inclusive prefix sum (double) — single-rank identity
    void scan_sum(const void* send_buf, void* recv_buf,
                  size_type count) noexcept {
        std::memcpy(recv_buf, send_buf, count * sizeof(double));
    }

    /// @brief Exclusive prefix sum (int) — single-rank: zero
    void exscan_sum_int(const void* /*send_buf*/, void* recv_buf,
                        size_type count) noexcept {
        std::memset(recv_buf, 0, count * sizeof(int));
    }

    /// @brief Exclusive prefix sum (long) — single-rank: zero
    void exscan_sum_long(const void* /*send_buf*/, void* recv_buf,
                         size_type count) noexcept {
        std::memset(recv_buf, 0, count * sizeof(long));
    }

    /// @brief Exclusive prefix sum (double) — single-rank: zero
    void exscan_sum(const void* /*send_buf*/, void* recv_buf,
                    size_type count) noexcept {
        std::memset(recv_buf, 0, count * sizeof(double));
    }

    /// @brief Type-dispatched inclusive scan (single-rank identity)
    template <typename T>
    T scan_sum_value(T value) noexcept { return value; }

    /// @brief Type-dispatched exclusive scan (single-rank: returns zero)
    template <typename T>
    T exscan_sum_value(T /*local*/) noexcept { return T{}; }

    // ------------------------------------------------------------------------
    // Non-Sum Scan Operations
    // ------------------------------------------------------------------------

    /// @brief Type-dispatched inclusive scan min (single-rank identity)
    template <typename T>
    T scan_min_value(T value) noexcept { return value; }

    /// @brief Type-dispatched inclusive scan max (single-rank identity)
    template <typename T>
    T scan_max_value(T value) noexcept { return value; }

    /// @brief Type-dispatched inclusive scan product (single-rank identity)
    template <typename T>
    T scan_prod_value(T value) noexcept { return value; }

    /// @brief Type-dispatched exclusive scan min (single-rank: identity element)
    template <typename T>
    T exscan_min_value(T /*value*/) noexcept {
        if constexpr (std::numeric_limits<T>::has_infinity) {
            return std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::max();
        }
    }

    /// @brief Type-dispatched exclusive scan max (single-rank: identity element)
    template <typename T>
    T exscan_max_value(T /*value*/) noexcept {
        if constexpr (std::numeric_limits<T>::has_infinity) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::lowest();
        }
    }

    /// @brief Type-dispatched exclusive scan product (single-rank: identity = 1)
    template <typename T>
    T exscan_prod_value(T /*value*/) noexcept { return T{1}; }

    // ------------------------------------------------------------------------
    // Variable-Size Collective Operations
    // ------------------------------------------------------------------------

    /// @brief Variable-size gather (single-rank: memcpy)
    void gatherv(const void* send_buf, size_type sendcount,
                 void* recv_buf, const int* /*recvcounts*/, const int* /*displs*/,
                 size_type elem_size, rank_t /*root*/) noexcept {
        std::memcpy(recv_buf, send_buf, sendcount * elem_size);
    }

    /// @brief Variable-size scatter (single-rank: memcpy)
    void scatterv(const void* send_buf, const int* /*sendcounts*/, const int* /*displs*/,
                  void* recv_buf, size_type recvcount, size_type elem_size,
                  rank_t /*root*/) noexcept {
        std::memcpy(recv_buf, send_buf, recvcount * elem_size);
    }

    /// @brief Variable-size allgather (single-rank: memcpy)
    void allgatherv(const void* send_buf, size_type sendcount,
                    void* recv_buf, const int* /*recvcounts*/, const int* /*displs*/,
                    size_type elem_size) noexcept {
        std::memcpy(recv_buf, send_buf, sendcount * elem_size);
    }

    /// @brief Variable-size all-to-all (single-rank: memcpy)
    void alltoallv(const void* send_buf, const int* sendcounts, const int* /*sdispls*/,
                   void* recv_buf, const int* /*recvcounts*/, const int* /*rdispls*/,
                   size_type elem_size) noexcept {
        std::memcpy(recv_buf, send_buf, static_cast<size_type>(sendcounts[0]) * elem_size);
    }

    // ------------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------------

    /// @brief Check if this is the root rank (always true for single-rank)
    [[nodiscard]] static constexpr bool is_root() noexcept {
        return true;
    }
};

// ============================================================================
// Communicator Handle
// ============================================================================

/// @brief Type-erased handle to any communicator
/// @details Provides virtual dispatch for communicator operations.
class communicator_handle {
public:
    /// @brief Construct with null communicator
    communicator_handle() : impl_(std::make_unique<model<null_communicator>>(null_communicator{})) {}

    /// @brief Construct with specific communicator
    template <typename Comm>
        requires Communicator<Comm>
    explicit communicator_handle(Comm comm)
        : impl_(std::make_unique<model<Comm>>(std::move(comm))) {}

    /// @brief Get rank
    [[nodiscard]] rank_t rank() const noexcept {
        return impl_->rank();
    }

    /// @brief Get size
    [[nodiscard]] rank_t size() const noexcept {
        return impl_->size();
    }

private:
    struct concept_base {
        virtual ~concept_base() = default;
        [[nodiscard]] virtual rank_t rank() const noexcept = 0;
        [[nodiscard]] virtual rank_t size() const noexcept = 0;
    };

    template <typename Comm>
    struct model final : concept_base {
        explicit model(Comm c) : comm(std::move(c)) {}
        [[nodiscard]] rank_t rank() const noexcept override { return comm.rank(); }
        [[nodiscard]] rank_t size() const noexcept override { return comm.size(); }
        Comm comm;
    };

    std::unique_ptr<concept_base> impl_;
};

// ============================================================================
// Communicator Utilities
// ============================================================================

/// @brief Get a string describing the communicator
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @return Description string
template <Communicator Comm>
[[nodiscard]] std::string communicator_description(const Comm& comm) {
    return "Communicator(rank=" + std::to_string(comm.rank()) +
           ", size=" + std::to_string(comm.size()) + ")";
}

/// @brief Check if rank is valid for communicator
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @param r Rank to check
/// @return true if valid
template <Communicator Comm>
[[nodiscard]] bool is_valid_rank(const Comm& comm, rank_t r) noexcept {
    return r >= 0 && r < comm.size();
}

/// @brief Calculate number of elements per rank for even distribution
/// @param total Total number of elements
/// @param num_ranks Number of ranks
/// @param rank Current rank
/// @return Number of elements for this rank
[[nodiscard]] inline size_type elements_per_rank(size_type total,
                                                  rank_t num_ranks,
                                                  rank_t rank) noexcept {
    size_type base = total / static_cast<size_type>(num_ranks);
    size_type remainder = total % static_cast<size_type>(num_ranks);
    return base + (static_cast<size_type>(rank) < remainder ? 1 : 0);
}

/// @brief Calculate offset for a rank in even distribution
/// @param total Total number of elements
/// @param num_ranks Number of ranks
/// @param rank Current rank
/// @return Starting offset for this rank
[[nodiscard]] inline size_type rank_offset(size_type total,
                                           rank_t num_ranks,
                                           rank_t rank) noexcept {
    size_type base = total / static_cast<size_type>(num_ranks);
    size_type remainder = total % static_cast<size_type>(num_ranks);
    size_type r = static_cast<size_type>(rank);
    if (r < remainder) {
        return r * (base + 1);
    } else {
        return remainder * (base + 1) + (r - remainder) * base;
    }
}

}  // namespace dtl
