// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file memory_transfer.hpp
/// @brief Memory transfer concept for cross-space copies
/// @details Defines requirements for copying data between memory spaces.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/backend/concepts/event.hpp>

#include <concepts>

namespace dtl {

// ============================================================================
// Transfer Direction
// ============================================================================

/// @brief Direction of memory transfer
enum class transfer_direction {
    host_to_host,      ///< Copy within host memory
    host_to_device,    ///< Copy from host to device
    device_to_host,    ///< Copy from device to host
    device_to_device,  ///< Copy within device memory
    unknown            ///< Direction determined at runtime
};

// ============================================================================
// Memory Transfer Concept
// ============================================================================

/// @brief Core memory transfer concept
/// @details Defines requirements for copying between memory spaces.
///
/// @par Required Operations:
/// - copy(): Synchronous copy
/// - direction(): Get transfer direction
template <typename T>
concept MemoryTransfer = requires(T& transfer, const T& ctransfer,
                                  void* dst, const void* src, size_type count) {
    // Synchronous copy
    { transfer.copy(dst, src, count) } -> std::same_as<void>;

    // Query
    { ctransfer.direction() } -> std::same_as<transfer_direction>;
};

// ============================================================================
// Async Memory Transfer Concept
// ============================================================================

/// @brief Memory transfer with async support
/// @details Adds non-blocking copy operations.
template <typename T>
concept AsyncMemoryTransfer = MemoryTransfer<T> &&
    requires(T& transfer, void* dst, const void* src, size_type count) {
    // Async copy returns an event
    { transfer.async_copy(dst, src, count) };  // Returns event-like type

    // Synchronization
    { transfer.synchronize() } -> std::same_as<void>;
};

// ============================================================================
// Bidirectional Transfer Concept
// ============================================================================

/// @brief Transfer supporting both directions
template <typename T>
concept BidirectionalTransfer = MemoryTransfer<T> &&
    requires(T& transfer, void* dst, const void* src, size_type count) {
    // Copy in specific direction
    { transfer.copy_to_device(dst, src, count) } -> std::same_as<void>;
    { transfer.copy_to_host(dst, src, count) } -> std::same_as<void>;
};

// ============================================================================
// Memory Transfer Factory Concept
// ============================================================================

/// @brief Factory for creating memory transfers
template <typename Factory, typename SrcSpace, typename DstSpace>
concept MemoryTransferFactory = MemorySpace<SrcSpace> && MemorySpace<DstSpace> &&
    requires(Factory& factory, SrcSpace& src_space, DstSpace& dst_space) {
    // Create transfer for space pair
    { factory.create_transfer(src_space, dst_space) };
};

// ============================================================================
// Standard Memory Transfer Operations
// ============================================================================

/// @brief Copy memory synchronously
/// @tparam Transfer Transfer type
/// @param transfer The transfer object
/// @param dst Destination pointer
/// @param src Source pointer
/// @param count Number of bytes to copy
template <MemoryTransfer Transfer>
void memory_copy(Transfer& transfer, void* dst, const void* src, size_type count) {
    transfer.copy(dst, src, count);
}

/// @brief Copy memory asynchronously
/// @tparam Transfer Async transfer type
/// @param transfer The transfer object
/// @param dst Destination pointer
/// @param src Source pointer
/// @param count Number of bytes to copy
/// @return Event representing completion
template <AsyncMemoryTransfer Transfer>
auto memory_copy_async(Transfer& transfer, void* dst, const void* src, size_type count) {
    return transfer.async_copy(dst, src, count);
}

// ============================================================================
// Memory Transfer Traits
// ============================================================================

/// @brief Traits for memory transfer types
template <typename Transfer>
struct memory_transfer_traits {
    /// @brief Whether transfer supports async operations
    static constexpr bool supports_async = false;

    /// @brief Whether transfer can be done by memcpy
    static constexpr bool is_trivial = false;

    /// @brief Whether transfer requires staging
    static constexpr bool requires_staging = false;
};

// ============================================================================
// Helper Functions
// ============================================================================

/// @brief Determine transfer direction between spaces
/// @details Uses memory_space_traits to classify source and destination
///          spaces and returns the appropriate transfer_direction enum.
///          Returns transfer_direction::unknown for unclassified spaces
///          (e.g., user-defined or mock spaces without traits specialization).
/// @tparam SrcSpace Source memory space
/// @tparam DstSpace Destination memory space
/// @return Transfer direction
template <MemorySpace SrcSpace, MemorySpace DstSpace>
[[nodiscard]] constexpr transfer_direction get_transfer_direction() noexcept {
    using src_traits = memory_space_traits<SrcSpace>;
    using dst_traits = memory_space_traits<DstSpace>;

    constexpr bool src_host = src_traits::is_host_space;
    constexpr bool src_device = src_traits::is_device_space;
    constexpr bool dst_host = dst_traits::is_host_space;
    constexpr bool dst_device = dst_traits::is_device_space;

    if constexpr (src_host && dst_host) {
        return transfer_direction::host_to_host;
    } else if constexpr (src_host && dst_device) {
        return transfer_direction::host_to_device;
    } else if constexpr (src_device && dst_host) {
        return transfer_direction::device_to_host;
    } else if constexpr (src_device && dst_device) {
        return transfer_direction::device_to_device;
    } else {
        // Unified spaces or unclassified user-defined spaces
        return transfer_direction::unknown;
    }
}

/// @brief Check if transfer between spaces is supported
/// @details A transfer is supported when the spaces are compatible (same
///          domain or one is unified), or when an explicit cross-domain
///          transfer path exists (host-to-device or device-to-host).
///          Returns false only for truly unsupported combinations, which
///          currently do not exist in the DTL type system, so this always
///          returns true. Kept for forward-compatibility with spaces that
///          may have restricted transfer support (e.g., RDMA-only regions).
/// @tparam SrcSpace Source memory space
/// @tparam DstSpace Destination memory space
template <MemorySpace SrcSpace, MemorySpace DstSpace>
[[nodiscard]] constexpr bool transfer_supported() noexcept {
    // All currently defined space combinations support transfers.
    // Cross-domain transfers are handled by staging through host memory
    // or GPU-aware MPI. Reserve this predicate for future restrictions.
    return true;
}

}  // namespace dtl
