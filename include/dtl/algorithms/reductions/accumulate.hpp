// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file accumulate.hpp
/// @brief Distributed accumulate algorithm
/// @details Sequential accumulation for non-commutative operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/async.hpp>

#include <functional>
#include <type_traits>
#include <string>
#include <vector>
#include <cstring>

// Forward declare Communicator concept for disambiguation
#include <dtl/backend/concepts/communicator.hpp>

// Futures for async variants
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

namespace dtl {

namespace detail {

/// @brief Send a value via communicator (handles trivial and string types)
template <typename T, typename Comm>
void accumulate_send(Comm& comm, const T& value, rank_t dest, int tag) {
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm.send(&value, sizeof(T), dest, tag);
    } else if constexpr (std::is_same_v<T, std::string>) {
        // For strings: send size, then data
        size_t size = value.size();
        comm.send(&size, sizeof(size), dest, tag);
        if (size > 0) {
            comm.send(value.data(), size, dest, tag + 1);
        }
    } else {
        // For other non-trivial types, attempt trivial send (may fail at runtime)
        comm.send(&value, sizeof(T), dest, tag);
    }
}

/// @brief Receive a value via communicator (handles trivial and string types)
template <typename T, typename Comm>
void accumulate_recv(Comm& comm, T& value, rank_t src, int tag) {
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm.recv(&value, sizeof(T), src, tag);
    } else if constexpr (std::is_same_v<T, std::string>) {
        // For strings: recv size, then data
        size_t size = 0;
        comm.recv(&size, sizeof(size), src, tag);
        if (size > 0) {
            std::vector<char> buffer(size);
            comm.recv(buffer.data(), size, src, tag + 1);
            value.assign(buffer.data(), size);
        } else {
            value.clear();
        }
    } else {
        // For other non-trivial types, attempt trivial recv (may fail at runtime)
        comm.recv(&value, sizeof(T), src, tag);
    }
}

/// @brief Broadcast a value via communicator (handles trivial and string types)
template <typename T, typename Comm>
void accumulate_broadcast(Comm& comm, T& value, rank_t root) {
    if constexpr (std::is_trivially_copyable_v<T>) {
        comm.broadcast(&value, sizeof(T), root);
    } else if constexpr (std::is_same_v<T, std::string>) {
        // For strings: broadcast size, then data
        size_t size = value.size();
        comm.broadcast(&size, sizeof(size), root);
        if (comm.rank() != root) {
            value.resize(size);
        }
        if (size > 0) {
            // Need mutable buffer for non-root ranks
            std::vector<char> buffer(size);
            if (comm.rank() == root) {
                std::memcpy(buffer.data(), value.data(), size);
            }
            comm.broadcast(buffer.data(), size, root);
            if (comm.rank() != root) {
                value.assign(buffer.data(), size);
            }
        }
    } else {
        // For other non-trivial types, attempt trivial broadcast (may fail at runtime)
        comm.broadcast(&value, sizeof(T), root);
    }
}

}  // namespace detail

// ============================================================================
// Distributed accumulate with communicator (Phase 6 Task 6.1)
// ============================================================================

/// @brief Sequential accumulation with MPI communicator (for non-commutative operations)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value
/// @param op Binary operation (need not be commutative)
/// @param comm The MPI communicator adapter
/// @return Accumulated result in global index order (same on all ranks)
///
/// @par Difference from reduce:
/// - accumulate: Sequential, preserves order, for non-commutative ops
/// - reduce: Parallel, requires associative+commutative ops
///
/// @par Complexity:
/// O(n) total operations, O(p) communication rounds.
/// Much more expensive than reduce for large p.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// Data is accumulated in strict global index order (rank 0, then rank 1, etc.).
///
/// @warning This algorithm is inherently sequential across ranks.
///          Prefer reduce when operation is commutative.
///
/// @par Example:
/// @code
/// // String concatenation (non-commutative)
/// mpi::mpi_comm_adapter comm;
/// distributed_vector<std::string> words(100, comm);
/// std::string sentence = dtl::accumulate(dtl::seq{}, words,
///     std::string{},
///     [](std::string a, const std::string& b) { return a + " " + b; },
///     comm);
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
T accumulate([[maybe_unused]] ExecutionPolicy&& policy,
             const Container& container,
             T init,
             BinaryOp op,
             Comm& comm) {
    // Handle empty container
    if (container.global_size() == 0) {
        return init;
    }

    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    // Phase 1: Accumulate local partition
    T local_result = init;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        local_result = op(local_result, *it);
    }

    // Phase 2: Sequential accumulation across ranks using MPI_Exscan pattern
    // Each rank receives accumulated value from previous ranks, combines with local, sends to next

    if (num_ranks == 1) {
        return local_result;
    }

    // For accumulate, we need to preserve order, so we use a chain pattern:
    // Rank 0 starts, sends to rank 1, rank 1 combines and sends to rank 2, etc.

    T accumulated_so_far = init;

    if (my_rank > 0) {
        // Receive accumulated result from previous rank
        detail::accumulate_recv(comm, accumulated_so_far, my_rank - 1, 0);
    }

    // Combine with local result
    // Note: For rank 0, this effectively computes op(init, local_result)
    // For other ranks, we want to first apply accumulated_so_far to local elements
    // But since we already computed local_result = op(init, local_elements),
    // we need: op(accumulated_so_far, local_result - init contribution)
    // For simplicity with generic ops, we recompute from accumulated_so_far
    T combined = accumulated_so_far;
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        combined = op(combined, *it);
    }

    if (my_rank < num_ranks - 1) {
        // Send combined result to next rank
        detail::accumulate_send(comm, combined, my_rank + 1, 0);
    }

    // Phase 3: Broadcast final result from last rank
    T final_result = combined;
    detail::accumulate_broadcast(comm, final_result, num_ranks - 1);

    return final_result;
}

/// @brief Accumulate with communicator using default addition
template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
T accumulate(ExecutionPolicy&& policy, const Container& container, T init, Comm& comm) {
    return accumulate(std::forward<ExecutionPolicy>(policy), container, init, std::plus<>{}, comm);
}

// ============================================================================
// Distributed accumulate (standalone - no communicator)
// ============================================================================

/// @brief Sequential accumulation (for non-commutative operations)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value
/// @param op Binary operation (need not be commutative)
/// @return Accumulated result in global index order
///
/// @par Difference from reduce:
/// - accumulate: Sequential, preserves order, for non-commutative ops
/// - reduce: Parallel, requires associative+commutative ops
///
/// @par Complexity:
/// O(n) total operations, O(p) communication rounds.
/// Much more expensive than reduce for large p.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// Data is accumulated in strict global index order.
///
/// @warning This algorithm is inherently sequential across ranks.
///          Prefer reduce when operation is commutative.
///
/// @par Example:
/// @code
/// // String concatenation (non-commutative)
/// distributed_vector<std::string> words(100, ctx);
/// std::string sentence = dtl::accumulate(dtl::seq{}, words,
///     std::string{},
///     [](std::string a, const std::string& b) { return a + " " + b; });
/// @endcode
///
/// @note This overload is for standalone (single-rank) usage. For multi-rank
///       usage, pass a communicator as the last argument.
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
T accumulate([[maybe_unused]] ExecutionPolicy&& policy,
             const Container& container,
             T init,
             BinaryOp op) {
    detail::require_collective_comm_or_single_rank(container, "dtl::accumulate");

    // First: accumulate local partition
    T local_result = init;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        local_result = op(local_result, *it);
    }

    // No communicator - return local result only
    // For distributed accumulation with MPI, use the overload with communicator parameter
    return local_result;
}

/// @brief Accumulate with default addition
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value
/// @return Sum of init and all elements in order
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
T accumulate(ExecutionPolicy&& policy, const Container& container, T init) {
    return accumulate(std::forward<ExecutionPolicy>(policy), container, init,
                      std::plus<>{});
}

/// @brief Accumulate with default execution and addition
template <typename Container, typename T>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("accumulate(container, init) is local-only for multi-rank containers; use accumulate(..., comm) for collective ordered semantics or local_accumulate(...) for rank-local semantics")
T accumulate(const Container& container, T init) {
    return accumulate(seq{}, container, init);
}

// ============================================================================
// Local-only accumulate (no communication)
// ============================================================================

/// @brief Accumulate local partition only
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param container The distributed container
/// @param init Initial value
/// @param op Binary operation
/// @return Local accumulated result
template <typename Container, typename T, typename BinaryOp>
    requires DistributedContainer<Container>
T local_accumulate(const Container& container, T init, BinaryOp op) {
    T result = init;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        result = op(result, *it);
    }
    return result;
}

/// @brief Local accumulate with default addition
template <typename Container, typename T>
    requires DistributedContainer<Container>
T local_accumulate(const Container& container, T init) {
    return local_accumulate(container, init, std::plus<>{});
}

// ============================================================================
// Async accumulate
// ============================================================================

/// @brief Asynchronously accumulate
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param container The distributed container
/// @param init Initial value
/// @param op Binary operation
/// @return Future containing result
template <typename Container, typename T, typename BinaryOp>
    requires DistributedContainer<Container>
auto async_accumulate(const Container& container, T init, BinaryOp op)
    -> futures::distributed_future<T> {
    auto promise = std::make_shared<futures::distributed_promise<T>>();
    auto future = promise->get_future();

    try {
        auto result = accumulate(seq{}, container, init, std::move(op));
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
