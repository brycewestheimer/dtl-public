// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_vector.hpp
/// @brief Primary distributed sequence container (std::vector analog)
/// @details Distributed 1D container with STL-like interface.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/fwd.hpp>
#include <dtl/core/sync_domain.hpp>
#include <dtl/core/runtime_device_context.hpp>
#include <dtl/error/result.hpp>
#include <dtl/policies/policies.hpp>
#include <dtl/index/partition_map.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/device_view.hpp>
#include <dtl/views/global_view.hpp>
#include <dtl/views/segmented_view.hpp>
#include <dtl/memory/default_allocator.hpp>
#include <dtl/containers/detail/storage.hpp>
#include <dtl/containers/detail/device_affinity.hpp>

#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <dtl/handle/handle.hpp>

namespace dtl {

/// @brief Distributed vector container
/// @tparam T Element type (must satisfy Transportable)
/// @tparam Policies... Policy pack (partition, placement, consistency, execution, error)
/// @details A distributed 1D container that distributes elements across ranks
///          according to the partition policy. Provides STL-like interface
///          with explicit distinction between local and global operations.
///
/// @par Key Design Points:
/// - Local operations via local_view() never communicate
/// - Global access via global_view() returns remote_ref<T> for all elements
/// - Segmented iteration via segmented_view() for efficient distributed algorithms
/// - All collective operations are explicit (resize, redistribute, barrier)
///
/// @par Deliberately Omitted std::vector Operations:
/// The following std::vector operations are intentionally NOT provided because
/// they cannot be implemented efficiently in a distributed context:
/// - `push_back()` / `emplace_back()` -- would require global rebalancing or
///   create load imbalance; use `resize()` + local assignment instead
/// - `insert()` / `erase()` -- would require O(n) global element shifting;
///   use `redistribute()` for bulk data movement
///   implementation detail that varies per rank; use `local_size()` for the
///   meaningful per-rank metric
///
/// @par Example Usage:
/// @code
/// // Single-rank convenience (no context needed)
/// dtl::distributed_vector<int> vec(1000);        // 1000 default-initialized elements
/// dtl::distributed_vector<int> vec(1000, 42);    // 1000 elements, all set to 42
///
/// // Multi-rank with context
/// dtl::distributed_vector<int> vec(1000, ctx);   // Distributed across ctx.size() ranks
///
/// // Local operations (no communication)
/// auto local = vec.local_view();
/// for (auto& elem : local) {
///     elem = compute(elem);
/// }
///
/// // Collective barrier
/// vec.barrier();
///
/// // Global access (may communicate)
/// auto ref = vec.global_view()[500];
/// if (ref.is_local()) {
///     int val = ref.get().value();  // No communication
/// } else {
///     int val = ref.get().value();  // Would communicate (returns error in standalone)
/// }
/// @endcode
template <typename T, typename... Policies>
class distributed_vector {
public:
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
    // Type Aliases
    // ========================================================================

    /// @brief Element type
    using value_type = T;

    /// @brief Allocator type (selected based on placement policy)
    using allocator_type = select_allocator_t<T, placement_policy>;

    /// @brief Storage type selected by placement policy
    using storage_type = detail::select_storage_t<T, placement_policy>;

    /// @brief Size type
    using size_type = dtl::size_type;

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
    using global_view_type = dtl::global_view<distributed_vector>;

    /// @brief Const global view type
    using const_global_view_type = dtl::global_view<const distributed_vector>;

    /// @brief Segmented view type
    using segmented_view_type = dtl::segmented_view<distributed_vector>;

    /// @brief Const segmented view type
    using const_segmented_view_type = dtl::segmented_view<const distributed_vector>;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (empty vector, single rank)
    distributed_vector() noexcept
        : partition_{0, 1, 0}
        , my_rank_{0}
        , num_ranks_{1}
        , comm_handle_(handle::comm_handle::local()) {}

    // -------------------------------------------------------------------------
    // Single-Rank Convenience Constructors
    // -------------------------------------------------------------------------

    /// @brief Construct for single-rank use (no context needed, default value)
    /// @param global_size Total number of elements
    /// @details Creates a single-rank distributed vector where all elements are
    ///          local. Useful for prototyping, testing, and sequential programs
    ///          that want to use DTL's container interface without MPI.
    explicit distributed_vector(size_type global_size)
        : partition_{global_size, 1, 0}
        , my_rank_{0}
        , num_ranks_{1}
        , local_data_(make_storage(global_size))
        , comm_handle_(handle::comm_handle::local()) {}

    /// @brief Construct for single-rank use (no context needed)
    /// @param global_size Total number of elements
    /// @param value Initial value for all elements
    /// @note Equivalent to distributed_vector(global_size, value, single_rank_context{})
    distributed_vector(size_type global_size, const T& value)
        requires (placement_policy::is_host_accessible())
        : partition_{global_size, 1, 0}
        , my_rank_{0}
        , num_ranks_{1}
        , local_data_(make_filled_storage(global_size, value))
        , comm_handle_(handle::comm_handle::local()) {}

    // -------------------------------------------------------------------------
    // Context-Based Constructors (V1.3.0 - Preferred)
    // -------------------------------------------------------------------------

    /// @brief Construct with global size and context
    /// @param global_size Total number of elements across all ranks
    /// @param ctx The execution context providing rank/size
    /// @details This is the preferred constructor for distributed vectors.
    ///          The context provides type-safe access to communication domains.
    ///          For device_only_runtime placement, the device ID is extracted
    ///          from the context's cuda_domain or hip_domain.
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    explicit distributed_vector(size_type global_size, const Ctx& ctx)
        : partition_{global_size, ctx.size(), ctx.rank()}
        , my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , local_data_(make_storage(partition_.local_size(), compute_device_id_from_ctx(ctx)))
        , device_id_(compute_device_id_from_ctx(ctx))
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    /// @brief Construct with global size, initial value, and context
    /// @param global_size Total number of elements across all ranks
    /// @param value Initial value for all elements
    /// @param ctx The execution context providing rank/size
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    distributed_vector(size_type global_size, const T& value, const Ctx& ctx)
        requires (placement_policy::is_host_accessible())
        : partition_{global_size, ctx.size(), ctx.rank()}
        , my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , local_data_(make_filled_storage(
              partition_.local_size(), value, compute_device_id_from_ctx(ctx)))
        , device_id_(compute_device_id_from_ctx(ctx))
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    // -------------------------------------------------------------------------
    // Legacy Constructors (Deprecated - will be removed in V2.0.0)
    // -------------------------------------------------------------------------

    /// @brief Construct with global size and distribution info
    /// @param global_size Total number of elements across all ranks
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    /// @deprecated Use distributed_vector(global_size, ctx) instead
    [[deprecated("Use distributed_vector(global_size, ctx) instead - will be removed in V2.0.0")]]
    distributed_vector(size_type global_size, rank_t num_ranks, rank_t my_rank)
        : partition_{global_size, num_ranks, my_rank}
        , my_rank_{my_rank}
        , num_ranks_{num_ranks}
        , local_data_(make_storage(partition_.local_size()))
        , comm_handle_(num_ranks > 1
            ? handle::comm_handle::unbound(my_rank, num_ranks)
            : handle::comm_handle::local()) {}

    /// @brief Construct with global size, distribution info, and initial value
    /// @param global_size Total number of elements across all ranks
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    /// @param value Initial value for all elements
    /// @deprecated Use distributed_vector(global_size, value, ctx) instead
    [[deprecated("Use distributed_vector(global_size, value, ctx) instead - will be removed in V2.0.0")]]
    distributed_vector(size_type global_size, rank_t num_ranks, rank_t my_rank, const T& value)
        requires (placement_policy::is_host_accessible())
        : partition_{global_size, num_ranks, my_rank}
        , my_rank_{my_rank}
        , num_ranks_{num_ranks}
        , local_data_(make_filled_storage(partition_.local_size(), value))
        , comm_handle_(num_ranks > 1
            ? handle::comm_handle::unbound(my_rank, num_ranks)
            : handle::comm_handle::local()) {}

    /// @brief Factory function returning result
    /// @param global_size Total number of elements
    /// @param num_ranks Number of ranks
    /// @param my_rank This rank's ID
    /// @return result<distributed_vector>
    static result<distributed_vector> create(size_type global_size, rank_t num_ranks, rank_t my_rank) {
        try {
            struct legacy_context {
                rank_t my_rank;
                rank_t num_ranks;

                [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
                [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
            };

            return result<distributed_vector>::success_in_place(
                global_size, legacy_context{my_rank, num_ranks});
        } catch (const std::exception& e) {
            return result<distributed_vector>::failure(
                status{status_code::allocation_failed});
        }
    }

    // ========================================================================
    // Size Queries
    // ========================================================================

    /// @brief Get total global size (sum of local sizes across all ranks)
    /// @return The total number of elements across ALL ranks, not just this rank.
    ///         For the number of elements stored locally, use local_size().
    /// @note Unlike std::vector::size(), this returns the distributed total.
    ///       For single-rank use, size() == local_size().
    [[nodiscard]] size_type size() const noexcept {
        return partition_.global_size();
    }

    /// @brief Alias for size() (global size)
    [[nodiscard]] size_type global_size() const noexcept {
        return partition_.global_size();
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

    /// @brief Check if globally empty
    [[nodiscard]] bool empty() const noexcept {
        return partition_.global_size() == 0;
    }

    /// @brief Get maximum possible size
    [[nodiscard]] size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max();
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
        // const_global_view_type = global_view<const distributed_vector>
        // The view constructor takes Container& which binds to const distributed_vector&
        return const_global_view_type{*this};
    }

    /// @brief Get segmented view (for distributed iteration)
    [[nodiscard]] segmented_view_type segmented_view() noexcept {
        return segmented_view_type{*this};
    }

    /// @brief Get const segmented view
    [[nodiscard]] const_segmented_view_type segmented_view() const noexcept {
        // const_segmented_view_type = segmented_view<const distributed_vector>
        // The view constructor takes Container& which binds to const distributed_vector&
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
    // Structural Operations (Collective - Invalidate Views)
    // ========================================================================

    /// @brief Resize the vector (collective operation)
    /// @param new_size New global size
    /// @return Result indicating success or error
    /// @warning Invalidates all views
    result<void> resize(size_type new_size) {
        if (num_ranks_ > 1 && !comm_handle_.has_collective_path()) {
            return result<void>::failure(status{
                status_code::invalid_state,
                my_rank_,
                "distributed_vector::resize requires a communicator-backed collective path when num_ranks()>1"});
        }

        try {
            partition_ = partition_map_type{new_size, num_ranks_, my_rank_};
            local_data_.resize(partition_.local_size());
            sync_state_.mark_global_dirty();
            return result<void>::success();
        } catch (const std::exception& e) {
            return result<void>::failure(
                status{status_code::allocation_failed});
        }
    }

    /// @brief Resize with value (collective operation)
    /// @param new_size New global size
    /// @param value Value for new elements
    /// @return Result indicating success or error
    result<void> resize(size_type new_size, const T& value)
        requires (placement_policy::is_host_accessible()) {
        if (num_ranks_ > 1 && !comm_handle_.has_collective_path()) {
            return result<void>::failure(status{
                status_code::invalid_state,
                my_rank_,
                "distributed_vector::resize(new_size, value) requires a communicator-backed collective path when num_ranks()>1"});
        }

        try {
            size_type old_local_size = partition_.local_size();
            partition_ = partition_map_type{new_size, num_ranks_, my_rank_};
            size_type new_local_size = partition_.local_size();
            local_data_.resize(new_local_size);
            // Fill new elements with value
            if (new_local_size > old_local_size) {
                std::fill(local_data_.begin() + static_cast<difference_type>(old_local_size),
                         local_data_.end(), value);
            }
            sync_state_.mark_global_dirty();
            return result<void>::success();
        } catch (const std::exception& e) {
            return result<void>::failure(
                status{status_code::allocation_failed});
        }
    }

    /// @brief Clear all elements (collective operation)
    /// @return Result indicating success or error
    result<void> clear() {
        if (num_ranks_ > 1 && !comm_handle_.has_collective_path()) {
            return result<void>::failure(status{
                status_code::invalid_state,
                my_rank_,
                "distributed_vector::clear requires a communicator-backed collective path when num_ranks()>1"});
        }

        local_data_.clear();
        partition_ = partition_map_type{0, num_ranks_, my_rank_};
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    /// @brief Check whether local storage size matches partition metadata
    [[nodiscard]] bool structural_metadata_consistent() const noexcept {
        return local_data_.size() == partition_.local_size();
    }

    /// @brief Atomically replace local partition storage while preserving global size
    /// @param new_local_data Replacement local partition payload
    /// @return Result indicating success or metadata mismatch
    result<void> replace_local_partition(storage_type new_local_data) {
        if (new_local_data.size() != partition_.local_size()) {
            return result<void>::failure(status{
                status_code::invalid_argument,
                no_rank,
                "replace_local_partition size mismatch with current partition metadata"});
        }

        local_data_.swap(new_local_data);
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    /// @brief Atomically replace local partition and update global size metadata
    /// @param new_local_data Replacement local partition payload
    /// @param new_global_size New global element count
    /// @return Result indicating success or metadata mismatch
    result<void> replace_local_partition(storage_type new_local_data,
                                         size_type new_global_size) {
        partition_map_type new_partition{new_global_size, num_ranks_, my_rank_};
        if (new_local_data.size() != new_partition.local_size()) {
            return result<void>::failure(status{
                status_code::invalid_argument,
                no_rank,
                "replace_local_partition size mismatch for new global size"});
        }

        partition_ = std::move(new_partition);
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
                "swap_local_storage size mismatch with current partition metadata"});
        }

        local_data_.swap(external_storage);
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    /// @brief Redistribute with new partition policy (collective)
    /// @tparam NewPartition New partition policy type
    /// @return Result indicating success or error
    /// @warning This is a collective operation - all ranks must participate
    /// @warning Invalidates all views
    /// @note For multi-rank redistribution, use redistribute_with_comm() instead
    ///
    /// @par Algorithm (requires communicator):
    /// 1. Compute send counts for each destination rank under new partition
    /// 2. Exchange counts via alltoall to determine receive counts
    /// 3. Compute displacements
    /// 4. Exchange data via alltoallv
    /// 5. Update local state with received data
    template <typename NewPartition>
    DTL_DEPRECATED_MSG("redistribute() without communicator is stubbed for multi-rank containers; use redistribute_with_comm<NewPartition>(comm)")
    result<void> redistribute() {
        // For single-rank case, partition policy change is a no-op for data
        // All elements remain local regardless of partition policy
        if (num_ranks_ <= 1) {
            return result<void>::success();
        }

        // Multi-rank redistribution requires a communicator for data exchange.
        // Use redistribute_with_comm<NewPartition>(comm) instead.
        return result<void>::failure(
            status{status_code::not_implemented, no_rank,
                   "Multi-rank redistribution requires communicator; "
                   "use redistribute_with_comm<NewPartition>(comm)"});
    }

    /// @brief Redistribute with new partition using communicator
    /// @tparam NewPartition New partition policy type
    /// @tparam Communicator Communicator type with alltoallv support
    /// @param comm Communicator for data exchange
    /// @return Result indicating success or error
    /// @note Cross-policy redistribution is currently unsupported because
    ///       partition metadata is encoded in the container's compile-time
    ///       partition policy. Use a container instantiated with the target
    ///       partition policy for now.
    template <typename NewPartition, typename Communicator>
    result<void> redistribute_with_comm(Communicator& comm) {
        if constexpr (!placement_policy::is_host_accessible()) {
            return result<void>::failure(
                status{
                    status_code::not_supported,
                    no_rank,
                    "redistribute_with_comm is not yet supported for device-only placement"});
        }

        if (num_ranks_ <= 1) {
            return result<void>::success();
        }

        if constexpr (!std::is_same_v<NewPartition, partition_policy>) {
            return result<void>::failure(
                status{
                    status_code::not_supported,
                    no_rank,
                    "Cross-policy redistribute_with_comm is not supported for "
                    "distributed_vector; instantiate the container with the "
                    "target partition policy"});
        }

        try {
            // Build new partition map
            using new_partition_map_type = partition_map<NewPartition>;
            new_partition_map_type new_partition{partition_.global_size(),
                                                  num_ranks_, my_rank_};

            // Compute send buffers by destination rank
            std::vector<std::vector<T>> send_buffers(static_cast<size_type>(num_ranks_));
            for (size_type i = 0; i < local_data_.size(); ++i) {
                index_t global_idx = partition_.to_global(static_cast<index_t>(i));
                rank_t dest = new_partition.owner(global_idx);
                send_buffers[static_cast<size_type>(dest)].push_back(local_data_[i]);
            }

            // Compute send counts and displacements
            std::vector<int> send_counts(static_cast<size_type>(num_ranks_));
            std::vector<int> send_displs(static_cast<size_type>(num_ranks_));
            for (rank_t r = 0; r < num_ranks_; ++r) {
                send_counts[static_cast<size_type>(r)] =
                    static_cast<int>(send_buffers[static_cast<size_type>(r)].size());
            }
            std::exclusive_scan(send_counts.begin(), send_counts.end(),
                               send_displs.begin(), 0);

            // Exchange counts
            std::vector<int> recv_counts(static_cast<size_type>(num_ranks_));
            comm.alltoall(send_counts.data(), recv_counts.data(), sizeof(int));

            // Compute receive displacements
            std::vector<int> recv_displs(static_cast<size_type>(num_ranks_));
            std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                               recv_displs.begin(), 0);

            // Flatten send buffer
            size_type total_send = static_cast<size_type>(send_displs.back() + send_counts.back());
            std::vector<T> send_flat;
            send_flat.reserve(total_send);
            for (auto& buf : send_buffers) {
                send_flat.insert(send_flat.end(), buf.begin(), buf.end());
            }

            // Allocate receive buffer
            size_type new_local_size = new_partition.local_size();
            storage_type new_local_data = make_storage(new_local_size, device_id_);

            // Exchange data
            comm.alltoallv(send_flat.data(), send_counts.data(), send_displs.data(),
                          new_local_data.data(), recv_counts.data(), recv_displs.data(),
                          sizeof(T));

            // Update local state
            local_data_ = std::move(new_local_data);
            partition_ = std::move(new_partition);
            sync_state_.mark_global_dirty();

            return result<void>::success();
        } catch (const std::exception& e) {
            return result<void>::failure(
                status{status_code::operation_failed, std::string("Redistribute failed: ") + e.what()});
        }
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

    /// @brief Synchronize halo regions (for stencil computations)
    /// @return Result indicating success or error
    /// @note Without communicator, marks halo synced without data exchange
    result<void> sync_halo() {
        // Without communicator, just mark as synced (single-rank or standalone mode)
        if (sync_state_.domain() == sync_domain::halo) {
            sync_state_.mark_halo_synced();
        }
        return result<void>::success();
    }

    /// @brief Synchronize halo regions with explicit communicator
    /// @tparam Communicator Communicator type with send/recv support
    /// @param comm Communicator for neighbor exchange
    /// @param halo_width Number of boundary elements to exchange per side
    /// @param left_halo_recv Optional receive buffer for left-neighbor halo data
    /// @param right_halo_recv Optional receive buffer for right-neighbor halo data
    /// @return Result indicating success or error
    /// @details Exchanges boundary values with neighbor ranks:
    ///          - Sends first halo_width elements to left neighbor (rank-1)
    ///          - Sends last halo_width elements to right neighbor (rank+1)
    ///          - Receives halo data from neighbors into halo buffers
    /// @note For full halo support, caller should pre-allocate halo buffers.
    ///       This method exchanges boundary values that can be used for stencil ops.
    template <typename Communicator>
    result<void> sync_halo_with_comm(Communicator& comm, size_type halo_width,
                                     T* left_halo_recv = nullptr,
                                     T* right_halo_recv = nullptr) {
        if constexpr (!placement_policy::is_host_accessible()) {
            return result<void>::failure(status{
                status_code::not_supported,
                no_rank,
                "sync_halo_with_comm is not supported for device-only placement"});
        }

        // Single rank or no halo: no-op
        if (num_ranks_ <= 1 || halo_width == 0 || local_data_.empty()) {
            if (sync_state_.domain() == sync_domain::halo) {
                sync_state_.mark_halo_synced();
            }
            return result<void>::success();
        }

        // Clamp halo width to available data
        halo_width = std::min(halo_width, local_data_.size());

        try {
            const int tag_left = 0;   // Tag for left-bound messages
            const int tag_right = 1;  // Tag for right-bound messages

            // Determine neighbors (use -1 for "no neighbor")
            const rank_t left_neighbor = (my_rank_ > 0) ? my_rank_ - 1 : no_rank;
            const rank_t right_neighbor = (my_rank_ < num_ranks_ - 1) ? my_rank_ + 1 : no_rank;

            // Temporary buffers if caller didn't provide storage
            std::vector<T> left_recv_buf;
            std::vector<T> right_recv_buf;

            if (left_neighbor != no_rank && left_halo_recv == nullptr) {
                left_recv_buf.resize(halo_width);
                left_halo_recv = left_recv_buf.data();
            }
            if (right_neighbor != no_rank && right_halo_recv == nullptr) {
                right_recv_buf.resize(halo_width);
                right_halo_recv = right_recv_buf.data();
            }

            // Exchange with left neighbor: send our first elements, receive their last elements
            if (left_neighbor != no_rank) {
                // Post non-blocking receive from left neighbor
                auto recv_req = comm.irecv(left_halo_recv, halo_width * sizeof(T),
                                           left_neighbor, tag_right);
                // Send our first halo_width elements to left neighbor
                comm.send(local_data_.data(), halo_width * sizeof(T),
                          left_neighbor, tag_left);
                // Wait for receive to complete
                comm.wait(recv_req);
            }

            // Exchange with right neighbor: send our last elements, receive their first elements
            if (right_neighbor != no_rank) {
                // Post non-blocking receive from right neighbor
                auto recv_req = comm.irecv(right_halo_recv, halo_width * sizeof(T),
                                           right_neighbor, tag_left);
                // Send our last halo_width elements to right neighbor
                comm.send(local_data_.data() + local_data_.size() - halo_width,
                          halo_width * sizeof(T), right_neighbor, tag_right);
                // Wait for receive to complete
                comm.wait(recv_req);
            }

            // Mark halo as synced
            if (sync_state_.domain() == sync_domain::halo) {
                sync_state_.mark_halo_synced();
            }

            return result<void>::success();
        } catch (const std::exception& e) {
            return result<void>::failure(
                status{status_code::operation_failed,
                       std::string("Halo exchange failed: ") + e.what()});
        }
    }

    /// @brief Synchronize halo with in-place boundary extension
    /// @tparam Communicator Communicator type
    /// @param comm Communicator for neighbor exchange
    /// @param halo_width Number of boundary elements
    /// @param left_halo Pre-allocated buffer for left neighbor's data
    /// @param right_halo Pre-allocated buffer for right neighbor's data
    /// @return Result indicating success or error
    /// @details Variant that writes received halo data to caller-provided buffers
    template <typename Communicator>
    result<void> sync_halo_with_comm(Communicator& comm, size_type halo_width,
                                     std::vector<T>& left_halo,
                                     std::vector<T>& right_halo) {
        // Resize buffers if needed
        if (my_rank_ > 0 && left_halo.size() < halo_width) {
            left_halo.resize(halo_width);
        }
        if (my_rank_ < num_ranks_ - 1 && right_halo.size() < halo_width) {
            right_halo.resize(halo_width);
        }

        return sync_halo_with_comm(comm, halo_width,
                                   left_halo.empty() ? nullptr : left_halo.data(),
                                   right_halo.empty() ? nullptr : right_halo.data());
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
        if constexpr (requires { local_data_.get_allocator(); }) {
            return local_data_.get_allocator();
        } else {
            return allocator_type{};
        }
    }

private:
    [[nodiscard]] static storage_type make_storage(size_type count) {
        return make_storage(count, compute_device_id_from_policy());
    }

    [[nodiscard]] static storage_type make_storage(size_type count, int device_id) {
        if constexpr (!placement_policy::is_host_accessible() &&
                      requires { storage_type{count, device_id}; }) {
            return storage_type(count, device_id);
        } else {
            return storage_type(count);
        }
    }

    [[nodiscard]] static storage_type make_filled_storage(
        size_type count,
        const T& value)
        requires (placement_policy::is_host_accessible()) {
        return make_filled_storage(count, value, compute_device_id_from_policy());
    }

    [[nodiscard]] static storage_type make_filled_storage(
        size_type count,
        const T& value,
        int device_id)
        requires (placement_policy::is_host_accessible()) {
        if constexpr (requires { storage_type{count, value}; }) {
            return storage_type{count, value};
        } else {
            auto storage = make_storage(count, device_id);
            std::fill(storage.begin(), storage.end(), value);
            return storage;
        }
    }

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
    /// @details For device_only_runtime, extracts device ID from context's
    ///          cuda_domain or hip_domain. For compile-time policies, uses
    ///          the policy's device_id().
    template <typename Ctx>
    [[nodiscard]] static int compute_device_id_from_ctx([[maybe_unused]] const Ctx& ctx) noexcept {
        // Check if this is a runtime device policy
        if constexpr (is_runtime_device_policy_v<placement_policy>) {
            // Extract device ID from context's GPU domain
            auto device_id = detail::ctx_gpu_device_id(ctx);
            return device_id.value_or(detail::no_device_affinity);
        } else if constexpr (requires { placement_policy::device_id(); }) {
            // Compile-time device policy
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

template <typename T, typename... Policies>
struct is_distributed_container<distributed_vector<T, Policies...>> : std::true_type {};

template <typename T, typename... Policies>
struct is_distributed_vector<distributed_vector<T, Policies...>> : std::true_type {};

}  // namespace dtl
