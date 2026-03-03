// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file communicator.hpp
/// @brief Communicator concept for distributed communication
/// @details Defines requirements for point-to-point and collective communication.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <span>

namespace dtl {

// ============================================================================
// Request Handle Type
// ============================================================================

/// @brief Type-erased request handle for non-blocking operations
/// @details Used to track pending communication operations.
struct request_handle {
    /// @brief Internal handle (implementation-defined)
    void* handle = nullptr;

    /// @brief Check if this request is valid
    [[nodiscard]] bool valid() const noexcept { return handle != nullptr; }
};

// ============================================================================
// Communicator Concept
// ============================================================================

/// @brief Core communicator concept for point-to-point communication
/// @details Defines minimum requirements for a distributed communicator.
///
/// @par Required Types:
/// - size_type: Type for counts/sizes
///
/// @par Required Operations:
/// - rank(): Get this process's rank
/// - size(): Get total number of ranks
/// - send(): Blocking send
/// - recv(): Blocking receive
/// - isend(): Non-blocking send
/// - irecv(): Non-blocking receive
/// - wait(): Wait for non-blocking operation
/// - test(): Test if non-blocking operation completed
template <typename T>
concept Communicator = requires(T& comm,
                                const T& ccomm,
                                void* buf,
                                const void* cbuf,
                                size_type count,
                                rank_t rank,
                                int tag,
                                request_handle& req) {
    // Type aliases
    typename T::size_type;

    // Query operations (const)
    { ccomm.rank() } -> std::same_as<rank_t>;
    { ccomm.size() } -> std::same_as<rank_t>;

    // Blocking point-to-point
    { comm.send(cbuf, count, rank, tag) } -> std::same_as<void>;
    { comm.recv(buf, count, rank, tag) } -> std::same_as<void>;

    // Non-blocking point-to-point
    { comm.isend(cbuf, count, rank, tag) } -> std::same_as<request_handle>;
    { comm.irecv(buf, count, rank, tag) } -> std::same_as<request_handle>;

    // Request completion
    { comm.wait(req) } -> std::same_as<void>;
    { comm.test(req) } -> std::same_as<bool>;
};

// ============================================================================
// Collective Communicator Concept
// ============================================================================

/// @brief Extended communicator with collective operations
/// @details Adds barrier, broadcast, scatter, gather to base communicator.
template <typename T>
concept CollectiveCommunicator = Communicator<T> &&
    requires(T& comm,
             void* buf,
             const void* cbuf,
             size_type count,
             rank_t root) {
    // Synchronization
    { comm.barrier() } -> std::same_as<void>;

    // Data movement
    { comm.broadcast(buf, count, root) } -> std::same_as<void>;
    { comm.scatter(cbuf, buf, count, root) } -> std::same_as<void>;
    { comm.gather(cbuf, buf, count, root) } -> std::same_as<void>;
    { comm.allgather(cbuf, buf, count) } -> std::same_as<void>;
    { comm.alltoall(cbuf, buf, count) } -> std::same_as<void>;
};

// ============================================================================
// Reducing Communicator Concept
// ============================================================================

/// @brief Communicator with reduction operations
/// @details Adds reduce and allreduce operations.
template <typename T>
concept ReducingCommunicator = CollectiveCommunicator<T> &&
    requires(T& comm,
             const void* sendbuf,
             void* recvbuf,
             size_type count,
             rank_t root) {
    // Reduction operations (using built-in sum operation for concept check)
    { comm.reduce_sum(sendbuf, recvbuf, count, root) } -> std::same_as<void>;
    { comm.allreduce_sum(sendbuf, recvbuf, count) } -> std::same_as<void>;
};

// ============================================================================
// Async Communicator Concept
// ============================================================================

/// @brief Communicator with non-blocking collective operations
template <typename T>
concept AsyncCommunicator = CollectiveCommunicator<T> &&
    requires(T& comm,
             void* buf,
             const void* cbuf,
             size_type count,
             rank_t root) {
    // Non-blocking collectives
    { comm.ibarrier() } -> std::same_as<request_handle>;
    { comm.ibroadcast(buf, count, root) } -> std::same_as<request_handle>;
    { comm.iscatter(cbuf, buf, count, root) } -> std::same_as<request_handle>;
    { comm.igather(cbuf, buf, count, root) } -> std::same_as<request_handle>;
};

// ============================================================================
// Communicator Tag Types
// ============================================================================

/// @brief Tag for MPI-style communicator implementations
struct mpi_communicator_tag {};

/// @brief Tag for shared-memory communicator implementations
struct shared_memory_communicator_tag {};

/// @brief Tag for NCCL-style GPU communicator implementations
struct gpu_communicator_tag {};

}  // namespace dtl
