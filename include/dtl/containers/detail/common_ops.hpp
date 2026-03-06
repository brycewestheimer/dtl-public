// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file common_ops.hpp
/// @brief Common container operations extracted from distributed containers
/// @details Provides free-function templates for operations that are identical
///          across distributed_vector, distributed_array, and distributed_tensor:
///          barrier, fence, sync, and sync-state queries. These are parameterized
///          on the container type so that each container can delegate to them
///          without duplicating the implementations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/sync_domain.hpp>
#include <dtl/error/result.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/device_view.hpp>
#include <dtl/views/global_view.hpp>
#include <dtl/views/segmented_view.hpp>

#include <atomic>

namespace dtl {
namespace detail {

// ============================================================================
// Standalone Barrier / Fence (no communicator)
// ============================================================================

/// @brief Standalone barrier (no-op without communicator)
/// @return Always succeeds in standalone mode
[[nodiscard]] inline result<void> standalone_barrier() noexcept {
    return result<void>::success();
}

/// @brief Standalone memory fence (seq_cst)
/// @return Always succeeds
[[nodiscard]] inline result<void> standalone_fence() noexcept {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return result<void>::success();
}

// ============================================================================
// Sync State Delegation
// ============================================================================

/// @brief Synchronize a container (barrier + mark clean)
/// @tparam Container Any container with barrier() and sync_state_ref()
/// @param container The container to sync
/// @return Result indicating success or error
template <typename Container>
[[nodiscard]] result<void> sync_container(Container& container) {
    auto barrier_result = container.barrier();
    if (!barrier_result) {
        return barrier_result;
    }
    container.sync_state_ref().mark_clean();
    return result<void>::success();
}

// ============================================================================
// Distribution Query Helpers
// ============================================================================

/// @brief Check rank/size consistency invariant
/// @tparam Container Container with rank() and num_ranks()
/// @param container The container to check
/// @return true if rank < num_ranks (or both are 0)
template <typename Container>
[[nodiscard]] constexpr bool rank_invariant_holds(const Container& container) noexcept {
    return container.rank() < container.num_ranks() || container.num_ranks() == 0;
}

// ============================================================================
// View Construction Helpers
// ============================================================================

/// @brief Construct a local_view from a container's local data and partition
/// @tparam Container Container type with local_data(), partition(), rank()
/// @param container The container
/// @return local_view<T> over the container's local partition
///
/// @note This helper captures the 4-argument local_view construction pattern
///       used by distributed_vector and distributed_array. distributed_tensor
///       uses a 2-argument pattern (no rank/offset), so it does not use this.
template <typename Container>
[[nodiscard]] auto make_local_view(Container& container) noexcept
    requires requires { Container::is_host_accessible(); } &&
             (Container::is_host_accessible()) {
    using T = typename Container::value_type;
    using view_type = dtl::local_view<T>;
    return view_type{container.local_data(), container.local_size(),
                     container.rank(), container.partition().local_offset()};
}

/// @brief Construct a const local_view from a container
template <typename Container>
[[nodiscard]] auto make_const_local_view(const Container& container) noexcept
    requires requires { Container::is_host_accessible(); } &&
             (Container::is_host_accessible()) {
    using T = typename Container::value_type;
    using view_type = dtl::local_view<const T>;
    return view_type{container.local_data(), container.local_size(),
                     container.rank(), container.partition().local_offset()};
}

/// @brief Construct a device_view from a GPU-capable container
template <typename Container>
[[nodiscard]] auto make_device_view(Container& container) noexcept
    requires requires(Container& c) { c.device_view(); } {
    return container.device_view();
}

/// @brief Construct a const device_view from a GPU-capable container
template <typename Container>
[[nodiscard]] auto make_device_view(const Container& container) noexcept
    requires requires(const Container& c) { c.device_view(); } {
    return container.device_view();
}

/// @brief Construct a global_view from a container
template <typename Container>
[[nodiscard]] auto make_global_view(Container& container) noexcept {
    using view_type = dtl::global_view<Container>;
    return view_type{container};
}

/// @brief Construct a const global_view from a container
template <typename Container>
[[nodiscard]] auto make_const_global_view(const Container& container) noexcept {
    using view_type = dtl::global_view<const Container>;
    return view_type{container};
}

/// @brief Construct a segmented_view from a container
template <typename Container>
[[nodiscard]] auto make_segmented_view(Container& container) noexcept {
    using view_type = dtl::segmented_view<Container>;
    return view_type{container};
}

/// @brief Construct a const segmented_view from a container
template <typename Container>
[[nodiscard]] auto make_const_segmented_view(const Container& container) noexcept {
    using view_type = dtl::segmented_view<const Container>;
    return view_type{container};
}

}  // namespace detail
}  // namespace dtl
