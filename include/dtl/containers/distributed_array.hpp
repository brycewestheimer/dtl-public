// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_array.hpp
/// @brief Fixed-size distributed array container (std::array analog)
/// @details Distributed 1D container with compile-time fixed size.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/fwd.hpp>
#include <dtl/error/result.hpp>
#include <dtl/policies/policies.hpp>
#include <dtl/index/partition_map.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/device_view.hpp>
#include <dtl/views/global_view.hpp>
#include <dtl/views/segmented_view.hpp>
#include <dtl/memory/default_allocator.hpp>
#include <dtl/core/sync_domain.hpp>
#include <dtl/core/runtime_device_context.hpp>
#include <dtl/containers/detail/device_affinity.hpp>
#include <dtl/handle/handle.hpp>

#include <array>
#include <algorithm>
#include <atomic>
#include <limits>

namespace dtl {

/// @brief Distributed array container with fixed compile-time size
/// @tparam T Element type (must satisfy Transportable)
/// @tparam N Compile-time array size (total elements across all ranks)
/// @tparam Policies... Policy pack (partition, placement, consistency, execution, error)
/// @details A distributed 1D container with fixed size known at compile time.
///          Unlike distributed_vector, the size cannot be changed after construction.
///          This mirrors the relationship between std::array and std::vector.
///
/// @par Key Design Points:
/// - Size N is a compile-time constant template parameter
/// - NO resize(), clear(), or push_back() - arrays are fixed size
/// - Local operations via local_view() never communicate
/// - Global access via global_view() returns remote_ref<T> for all elements
/// - Segmented iteration via segmented_view() for efficient distributed algorithms
///
/// @par Example Usage:
/// @code
/// // Create array with compile-time size
/// dtl::distributed_array<int, 1000> arr(4, 1);  // 1000 elements, 4 ranks, I'm rank 1
///
/// // Local operations (no communication)
/// auto local = arr.local_view();
/// for (auto& elem : local) {
///     elem = compute(elem);
/// }
///
/// // Compile-time size access
/// static_assert(arr.extent == 1000);
/// @endcode
template <typename T, size_type N, typename... Policies>
class distributed_array {
public:
    // ========================================================================
    // Type Aliases (needed before extent for proper declaration order)
    // ========================================================================

    /// @brief Size type
    using size_type = dtl::size_type;

    // ========================================================================
    // Compile-time constants
    // ========================================================================

    /// @brief Compile-time array extent (total elements)
    static constexpr size_type extent = N;

    // ========================================================================
    // Policy Types
    // ========================================================================

    /// @brief Extracted policy set from variadic parameters
    using policies = make_policy_set<Policies...>;

    /// @brief Partition policy for this container
    using partition_policy = typename policies::partition_policy;

    /// @brief Placement policy for this container
    using placement_policy = typename policies::placement_policy;

    /// @brief Consistency policy for this container
    using consistency_policy = typename policies::consistency_policy;

    /// @brief Execution policy for this container
    using execution_policy = typename policies::execution_policy;

    /// @brief Error policy for this container
    using error_policy = typename policies::error_policy;

    // ========================================================================
    // Type Aliases (continued)
    // ========================================================================

    /// @brief Element type
    using value_type = T;

    /// @brief Allocator type (selected based on placement policy)
    using allocator_type = select_allocator_t<T, placement_policy>;

    /// @brief Storage type (std::vector with selected allocator for runtime local size)
    using storage_type = std::vector<T, allocator_type>;

    /// @brief Difference type
    using difference_type = std::ptrdiff_t;

    /// @brief Reference type (for local access)
    using reference = T&;

    /// @brief Const reference type
    using const_reference = const T&;

    /// @brief Pointer type
    using pointer = T*;

    /// @brief Const pointer type
    using const_pointer = const T*;

    /// @brief Partition map type
    using partition_map_type = partition_map<partition_policy>;

    /// @brief Local view type
    using local_view_type = dtl::local_view<T>;

    /// @brief Const local view type
    using const_local_view_type = dtl::local_view<const T>;

    /// @brief Global view type
    using global_view_type = dtl::global_view<distributed_array>;

    /// @brief Const global view type
    using const_global_view_type = dtl::global_view<const distributed_array>;

    /// @brief Segmented view type
    using segmented_view_type = dtl::segmented_view<distributed_array>;

    /// @brief Const segmented view type
    using const_segmented_view_type = dtl::segmented_view<const distributed_array>;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (single rank, all elements local)
    distributed_array() noexcept
        : partition_{N, 1, 0}
        , my_rank_{0}
        , num_ranks_{1}
        , local_data_(partition_.local_size())
        , comm_handle_(handle::comm_handle::local()) {}

    // -------------------------------------------------------------------------
    // Context-Based Constructors (V1.3.0 - Preferred)
    // -------------------------------------------------------------------------

    /// @brief Construct with context
    /// @param ctx The execution context providing rank/size
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    explicit distributed_array(const Ctx& ctx)
        : partition_{N, ctx.size(), ctx.rank()}
        , my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , local_data_(partition_.local_size())
        , device_id_(compute_device_id_from_ctx(ctx))
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    /// @brief Construct with initial value and context
    /// @param value Initial value for all elements
    /// @param ctx The execution context providing rank/size
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    distributed_array(const T& value, const Ctx& ctx)
        : partition_{N, ctx.size(), ctx.rank()}
        , my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , local_data_(partition_.local_size(), value)
        , device_id_(compute_device_id_from_ctx(ctx))
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    // -------------------------------------------------------------------------
    // Legacy Constructors (Deprecated - will be removed in V2.0.0)
    // -------------------------------------------------------------------------

    /// @brief Construct with distribution info
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    /// @deprecated Use distributed_array(ctx) instead
    [[deprecated("Use distributed_array(ctx) instead - will be removed in V2.0.0")]]
    distributed_array(rank_t num_ranks, rank_t my_rank)
        : partition_{N, num_ranks, my_rank}
        , my_rank_{my_rank}
        , num_ranks_{num_ranks}
        , local_data_(partition_.local_size())
        , comm_handle_(num_ranks > 1
            ? handle::comm_handle::unbound(my_rank, num_ranks)
            : handle::comm_handle::local()) {}

    /// @brief Construct with distribution info and initial value
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    /// @param value Initial value for all elements
    /// @deprecated Use distributed_array(value, ctx) instead
    [[deprecated("Use distributed_array(value, ctx) instead - will be removed in V2.0.0")]]
    distributed_array(rank_t num_ranks, rank_t my_rank, const T& value)
        : partition_{N, num_ranks, my_rank}
        , my_rank_{my_rank}
        , num_ranks_{num_ranks}
        , local_data_(partition_.local_size(), value)
        , comm_handle_(num_ranks > 1
            ? handle::comm_handle::unbound(my_rank, num_ranks)
            : handle::comm_handle::local()) {}

    /// @brief Factory function returning result
    /// @param num_ranks Number of ranks
    /// @param my_rank This rank's ID
    /// @return result<distributed_array>
    static result<distributed_array> create(rank_t num_ranks, rank_t my_rank) {
        try {
            struct legacy_context {
                rank_t my_rank;
                rank_t num_ranks;

                [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
                [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
            };

            return result<distributed_array>::success_in_place(
                legacy_context{my_rank, num_ranks});
        } catch (const std::exception& e) {
            return result<distributed_array>::failure(
                status{status_code::allocation_failed});
        }
    }

    // ========================================================================
    // Size Queries (mostly constexpr)
    // ========================================================================

    /// @brief Get total global size (always N)
    [[nodiscard]] static constexpr size_type size() noexcept {
        return N;
    }

    /// @brief Alias for size() (global size, always N)
    [[nodiscard]] static constexpr size_type global_size() noexcept {
        return N;
    }

    /// @brief Get local size on this rank
    [[nodiscard]] size_type local_size() const noexcept {
        return partition_.local_size();
    }

    /// @brief Get local size for a specific rank
    /// @param r The rank to query
    [[nodiscard]] size_type local_size_for_rank(rank_t r) const noexcept {
        return partition_.local_size(r);
    }

    /// @brief Check if globally empty (always false if N > 0)
    [[nodiscard]] static constexpr bool empty() noexcept {
        return N == 0;
    }

    /// @brief Get maximum possible size (always N for arrays)
    [[nodiscard]] static constexpr size_type max_size() noexcept {
        return N;
    }

    // ========================================================================
    // View Accessors
    // ========================================================================

    /// @brief Get local view (STL-compatible, no communication)
    [[nodiscard]] local_view_type local_view() noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_view_type{local_data_.data(), local_data_.size(),
                              my_rank_, partition_.local_offset()};
    }

    /// @brief Get const local view
    [[nodiscard]] const_local_view_type local_view() const noexcept
        requires (placement_policy::is_host_accessible()) {
        return const_local_view_type{local_data_.data(), local_data_.size(),
                                     my_rank_, partition_.local_offset()};
    }

    /// @brief Get device view for GPU-capable placements
    [[nodiscard]] auto device_view() noexcept
        requires (DeviceStorable<T> && placement_policy::is_device_accessible()) {
        return dtl::make_device_view(local_data_.data(), local_data_.size(), device_id_);
    }

    /// @brief Get const device view for GPU-capable placements
    [[nodiscard]] auto device_view() const noexcept
        requires (DeviceStorable<T> && placement_policy::is_device_accessible()) {
        return dtl::make_device_view(local_data_.data(), local_data_.size(), device_id_);
    }

    /// @brief Get global view (returns remote_ref for all elements)
    [[nodiscard]] global_view_type global_view() noexcept {
        return global_view_type{*this};
    }

    /// @brief Get const global view
    [[nodiscard]] const_global_view_type global_view() const noexcept {
        return const_global_view_type{*this};
    }

    /// @brief Get segmented view (for distributed iteration)
    [[nodiscard]] segmented_view_type segmented_view() noexcept {
        return segmented_view_type{*this};
    }

    /// @brief Get const segmented view
    [[nodiscard]] const_segmented_view_type segmented_view() const noexcept {
        return const_segmented_view_type{*this};
    }

    // ========================================================================
    // Local Element Access
    // ========================================================================

    /// @brief Access local element by local index
    /// @param local_idx Index within local partition
    [[nodiscard]] reference local(size_type local_idx) noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_data_[local_idx];
    }

    /// @brief Access local element by local index (const)
    [[nodiscard]] const_reference local(size_type local_idx) const noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_data_[local_idx];
    }

    /// @brief Get pointer to local data
    [[nodiscard]] pointer local_data() noexcept {
        return local_data_.data();
    }

    /// @brief Get const pointer to local data
    [[nodiscard]] const_pointer local_data() const noexcept {
        return local_data_.data();
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// @brief Collective barrier (all ranks must call)
    /// @return Result indicating success or error
    /// @note In standalone mode, this is a no-op
    result<void> barrier() {
        return comm_handle_.barrier();
    }

    [[nodiscard]] handle::comm_handle communicator_handle() const {
        return comm_handle_;
    }

    /// @brief Memory fence for consistency
    /// @return Result indicating success or error
    result<void> fence() {
        // Standalone mode: just memory fence
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return result<void>::success();
    }

    // ========================================================================
    // Fill Operation
    // ========================================================================

    /// @brief Fill all local elements with a value
    /// @param value The value to fill with
    void fill(const T& value) {
        std::fill(local_data_.begin(), local_data_.end(), value);
        sync_state_.mark_local_modified();
    }

    /// @brief Check whether local storage size matches partition metadata
    [[nodiscard]] bool structural_metadata_consistent() const noexcept {
        return local_data_.size() == partition_.local_size();
    }

    /// @brief Atomically replace local partition storage for fixed-size arrays
    /// @param new_local_data Replacement local partition payload
    /// @return Result indicating success or metadata mismatch
    result<void> replace_local_partition(storage_type new_local_data) {
        if (new_local_data.size() != partition_.local_size()) {
            return result<void>::failure(status{
                status_code::invalid_argument,
                no_rank,
                "replace_local_partition size mismatch with array partition metadata"});
        }

        local_data_.swap(new_local_data);
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    /// @brief Swap local storage with caller-provided storage (same metadata)
    /// @param external_storage External storage buffer
    /// @return Result indicating success or metadata mismatch
    result<void> swap_local_storage(storage_type& external_storage) {
        if (external_storage.size() != partition_.local_size()) {
            return result<void>::failure(status{
                status_code::invalid_argument,
                no_rank,
                "swap_local_storage size mismatch with array partition metadata"});
        }

        local_data_.swap(external_storage);
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    // ========================================================================
    // Distribution Queries
    // ========================================================================

    /// @brief Get number of ranks
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return num_ranks_;
    }

    /// @brief Get current rank
    [[nodiscard]] rank_t rank() const noexcept {
        return my_rank_;
    }

    /// @brief Check if a global index is local
    [[nodiscard]] bool is_local(index_t global_idx) const noexcept {
        return partition_.is_local(global_idx);
    }

    /// @brief Get owner rank for a global index
    [[nodiscard]] rank_t owner(index_t global_idx) const noexcept {
        return partition_.owner(global_idx);
    }

    /// @brief Convert global index to local index
    /// @param global_idx The global index to convert
    /// @return The local index
    /// @pre is_local(global_idx) must be true
    [[nodiscard]] index_t to_local(index_t global_idx) const noexcept {
        return partition_.to_local(global_idx);
    }

    /// @brief Convert local index to global index
    /// @param local_idx The local index to convert
    /// @return The global index
    [[nodiscard]] index_t to_global(index_t local_idx) const noexcept {
        return partition_.to_global(local_idx);
    }

    /// @brief Get global offset of local partition
    [[nodiscard]] index_t global_offset() const noexcept {
        return partition_.local_offset();
    }

    // ========================================================================
    // Partition Map Access
    // ========================================================================

    /// @brief Get the partition map
    [[nodiscard]] const partition_map_type& partition() const noexcept {
        return partition_;
    }

    // ========================================================================
    // Placement Policy Queries
    // ========================================================================

    // ========================================================================
    // Sync State
    // ========================================================================

    /// @brief Check if container has local modifications
    [[nodiscard]] bool is_dirty() const noexcept {
        return sync_state_.is_dirty();
    }

    /// @brief Check if container is clean
    [[nodiscard]] bool is_clean() const noexcept {
        return sync_state_.is_clean();
    }

    /// @brief Get sync state reference
    [[nodiscard]] const sync_state& sync_state_ref() const noexcept {
        return sync_state_;
    }

    /// @brief Get sync state reference (mutable)
    [[nodiscard]] sync_state& sync_state_ref() noexcept {
        return sync_state_;
    }

    /// @brief Mark container as clean
    void mark_clean() noexcept {
        sync_state_.mark_clean();
    }

    /// @brief Mark container as locally modified
    void mark_local_modified() noexcept {
        sync_state_.mark_local_modified();
    }

    /// @brief Synchronize the container (barrier + mark clean)
    /// @return Result indicating success or error
    result<void> sync() {
        auto barrier_result = barrier();
        if (!barrier_result) {
            return barrier_result;
        }
        sync_state_.mark_clean();
        return result<void>::success();
    }

    // ========================================================================
    // Placement Policy Queries
    // ========================================================================

    /// @brief Check if memory is accessible from host CPU
    /// @return true if host accessible, false otherwise
    [[nodiscard]] static constexpr bool is_host_accessible() noexcept {
        return placement_policy::is_host_accessible();
    }

    /// @brief Check if memory is accessible from device (GPU)
    /// @return true if device accessible, false otherwise
    [[nodiscard]] static constexpr bool is_device_accessible() noexcept {
        return placement_policy::is_device_accessible();
    }

    /// @brief Get the device ID this container is affiliated with
    /// @return Device ID (>= 0 for device memory, -1 for host, -2 for unified)
    [[nodiscard]] constexpr int device_id() const noexcept {
        return device_id_;
    }

    /// @brief Check if container has device affinity
    /// @return true if container is on a specific device
    [[nodiscard]] constexpr bool has_device_affinity() const noexcept {
        return device_id_ >= 0;
    }

    /// @brief Get the allocator instance
    [[nodiscard]] allocator_type get_allocator() const noexcept {
        return local_data_.get_allocator();
    }

private:
    /// @brief Compute device ID from placement policy (compile-time policies)
    [[nodiscard]] static constexpr int compute_device_id_from_policy() noexcept {
        if constexpr (requires { placement_policy::device_id(); }) {
            return placement_policy::device_id();
        } else if constexpr (placement_policy::is_host_accessible() &&
                             !placement_policy::is_device_accessible()) {
            return detail::no_device_affinity;  // Host memory
        } else {
            return detail::no_device_affinity;  // Default to no affinity
        }
    }

    /// @brief Compute device ID from context (runtime device policies)
    template <typename Ctx>
    [[nodiscard]] static int compute_device_id_from_ctx([[maybe_unused]] const Ctx& ctx) noexcept {
        if constexpr (is_runtime_device_policy_v<placement_policy>) {
            auto device_id = detail::ctx_gpu_device_id(ctx);
            return device_id.value_or(detail::no_device_affinity);
        } else if constexpr (requires { placement_policy::device_id(); }) {
            return placement_policy::device_id();
        } else if constexpr (placement_policy::is_host_accessible() &&
                             !placement_policy::is_device_accessible()) {
            return detail::no_device_affinity;  // Host memory
        } else {
            return detail::no_device_affinity;  // Default to no affinity
        }
    }

    partition_map_type partition_;
    rank_t my_rank_;
    rank_t num_ranks_;
    storage_type local_data_;  // Local storage using placement-selected allocator
    sync_state sync_state_;    // Sync state tracking
    int device_id_ = compute_device_id_from_policy();  // Device affinity
    handle::comm_handle comm_handle_;
};

// =============================================================================
// Type Trait Specializations
// =============================================================================

template <typename T, size_type N, typename... Policies>
struct is_distributed_container<distributed_array<T, N, Policies...>> : std::true_type {};

/// @brief Type trait for distributed_array
template <typename T>
struct is_distributed_array : std::false_type {};

template <typename T, size_type N, typename... Policies>
struct is_distributed_array<distributed_array<T, N, Policies...>> : std::true_type {};

/// @brief Helper variable template
template <typename T>
inline constexpr bool is_distributed_array_v = is_distributed_array<T>::value;

}  // namespace dtl
