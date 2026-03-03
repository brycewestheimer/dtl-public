// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rma_communicator.hpp
/// @brief RmaCommunicator concept for one-sided communication
/// @details Defines requirements for Remote Memory Access (RMA) operations
///          enabling one-sided communication where the target process does not
///          participate in data transfer.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <concepts>

namespace dtl {

// ============================================================================
// Window Handle Type
// ============================================================================

/// @brief Type-erased window handle for RMA operations
/// @details Used to track memory windows exposed for one-sided access.
struct window_handle {
    /// @brief Internal handle (implementation-defined)
    void* handle = nullptr;

    /// @brief Check if this window is valid
    [[nodiscard]] constexpr bool valid() const noexcept { return handle != nullptr; }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const window_handle& other) const noexcept {
        return handle == other.handle;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const window_handle& other) const noexcept {
        return handle != other.handle;
    }
};

// ============================================================================
// RMA Lock Modes
// ============================================================================

/// @brief Lock mode for passive-target RMA synchronization
/// @details Controls access to a target window during lock/unlock epochs.
enum class rma_lock_mode {
    /// @brief Exclusive access (no concurrent access allowed)
    exclusive,
    /// @brief Shared access (concurrent shared locks allowed)
    shared
};

// ============================================================================
// RMA Reduction Operations
// ============================================================================

/// @brief Reduction operations for RMA accumulate operations
/// @details Defines atomic operations that can be performed during RMA accumulate.
enum class rma_reduce_op {
    /// @brief Sum: accumulate += origin
    sum,
    /// @brief Product: accumulate *= origin
    prod,
    /// @brief Minimum: accumulate = min(accumulate, origin)
    min,
    /// @brief Maximum: accumulate = max(accumulate, origin)
    max,
    /// @brief Bitwise AND: accumulate &= origin
    band,
    /// @brief Bitwise OR: accumulate |= origin
    bor,
    /// @brief Bitwise XOR: accumulate ^= origin
    bxor,
    /// @brief Replace: accumulate = origin
    replace,
    /// @brief No-op (used for fetch without modification)
    no_op
};

// ============================================================================
// RmaCommunicator Concept
// ============================================================================

/// @brief Concept for communicators with RMA (one-sided) operations
/// @details Extends Communicator with remote memory access capabilities.
///
/// @par Required Operations:
/// - create_window(): Create a memory window for RMA
/// - free_window(): Release a memory window
/// - put(): Write data to a remote window
/// - get(): Read data from a remote window
/// - fence(): Active-target synchronization
/// - flush(): Ensure operations to target complete
/// - flush_all(): Ensure all operations complete
///
/// @par Synchronization Models:
/// RMA supports two synchronization models:
/// 1. Active-target (fence): All ranks call fence collectively
/// 2. Passive-target (lock/unlock): Target does not participate
///
/// @tparam T The communicator type to check
template <typename T>
concept RmaCommunicator = Communicator<T> &&
    requires(T& comm, const T& ccomm,
             void* base, void* buf,
             const void* cbuf,
             size_type size, size_type offset,
             rank_t target, window_handle& win) {
    // Window management
    { comm.create_window(base, size) } -> std::same_as<window_handle>;
    { comm.free_window(win) } -> std::same_as<void>;

    // One-sided data transfer
    { comm.put(cbuf, size, target, offset, win) } -> std::same_as<void>;
    { comm.get(buf, size, target, offset, win) } -> std::same_as<void>;

    // Active-target synchronization
    { comm.fence(win) } -> std::same_as<void>;

    // Remote completion
    { comm.flush(target, win) } -> std::same_as<void>;
    { comm.flush_all(win) } -> std::same_as<void>;
};

// ============================================================================
// Passive Target RmaCommunicator Concept
// ============================================================================

/// @brief Concept for RMA communicators with passive-target synchronization
/// @details Adds lock/unlock operations for passive-target access.
///
/// @par Additional Operations:
/// - lock(): Acquire lock on target window
/// - unlock(): Release lock on target window
/// - lock_all(): Acquire shared lock on all windows
/// - unlock_all(): Release locks on all windows
template <typename T>
concept PassiveTargetRmaCommunicator = RmaCommunicator<T> &&
    requires(T& comm, rank_t target, window_handle& win, rma_lock_mode mode) {
    // Single-target locking
    { comm.lock(target, mode, win) } -> std::same_as<void>;
    { comm.unlock(target, win) } -> std::same_as<void>;

    // All-target locking (for global access patterns)
    { comm.lock_all(win) } -> std::same_as<void>;
    { comm.unlock_all(win) } -> std::same_as<void>;
};

// ============================================================================
// Atomic RMA Communicator Concept
// ============================================================================

/// @brief Concept for RMA communicators with atomic operations
/// @details Adds atomic accumulate and fetch-and-op operations.
///
/// @par Required Operations:
/// - accumulate(): Atomically combine values
/// - fetch_and_op(): Fetch old value and apply operation
/// - compare_and_swap(): Atomic compare-and-swap
template <typename T>
concept AtomicRmaCommunicator = RmaCommunicator<T> &&
    requires(T& comm,
             const void* origin, void* result,
             size_type size, rank_t target, size_type offset,
             window_handle& win, rma_reduce_op op) {
    // Atomic accumulate
    { comm.accumulate(origin, size, target, offset, op, win) } -> std::same_as<void>;

    // Fetch and atomic operation
    { comm.fetch_and_op(origin, result, size, target, offset, op, win) } -> std::same_as<void>;

    // Atomic compare-and-swap
    { comm.compare_and_swap(origin, origin, result, size, target, offset, win) }
        -> std::same_as<void>;
};

// ============================================================================
// Full RMA Communicator Concept
// ============================================================================

/// @brief Concept for communicators with full RMA support
/// @details Combines passive-target and atomic RMA capabilities.
template <typename T>
concept FullRmaCommunicator = PassiveTargetRmaCommunicator<T> && AtomicRmaCommunicator<T>;

// ============================================================================
// Capability Detection Traits
// ============================================================================

/// @brief Trait to detect RMA capability
/// @tparam C The communicator type
template <typename C>
inline constexpr bool supports_rma_v = RmaCommunicator<C>;

/// @brief Trait to detect passive-target RMA capability
/// @tparam C The communicator type
template <typename C>
inline constexpr bool supports_passive_target_rma_v = PassiveTargetRmaCommunicator<C>;

/// @brief Trait to detect atomic RMA capability
/// @tparam C The communicator type
template <typename C>
inline constexpr bool supports_atomic_rma_v = AtomicRmaCommunicator<C>;

/// @brief Trait to detect full RMA capability
/// @tparam C The communicator type
template <typename C>
inline constexpr bool supports_full_rma_v = FullRmaCommunicator<C>;

// ============================================================================
// RMA Communicator Tag Type
// ============================================================================

/// @brief Tag for RMA-capable communicator implementations
struct rma_communicator_tag {};

}  // namespace dtl
