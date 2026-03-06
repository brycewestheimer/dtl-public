// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_tensor.hpp
/// @brief N-dimensional distributed tensor container
/// @details MVP-critical: ND/tensor is core functionality.
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
#include <dtl/views/remote_ref.hpp>
#include <dtl/memory/default_allocator.hpp>
#include <dtl/core/sync_domain.hpp>
#include <dtl/core/runtime_device_context.hpp>
#include <dtl/containers/detail/device_affinity.hpp>
#include <dtl/handle/handle.hpp>

#include <array>
#include <atomic>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <span>

namespace dtl {

// Note: nd_index<N> is defined in dtl/core/types.hpp
// Note: nd_extent<N> is defined in dtl/core/types.hpp

// =============================================================================
// Layout Policies
// =============================================================================

/// @brief Row-major (C-style) layout - last dimension varies fastest
struct row_major {
    /// @brief Linearize ND index to 1D
    template <size_type Rank>
    [[nodiscard]] static constexpr index_t linearize(
        const nd_index<Rank>& idx, const nd_extent<Rank>& extents) noexcept {
        index_t result = 0;
        index_t stride = 1;
        for (size_type d = Rank; d > 0; --d) {
            result += idx[d - 1] * stride;
            stride *= static_cast<index_t>(extents[d - 1]);
        }
        return result;
    }

    /// @brief Delinearize 1D index to ND
    template <size_type Rank>
    [[nodiscard]] static constexpr nd_index<Rank> delinearize(
        index_t linear, const nd_extent<Rank>& extents) noexcept {
        nd_index<Rank> result{};
        for (size_type d = Rank; d > 0; --d) {
            result[d - 1] = linear % static_cast<index_t>(extents[d - 1]);
            linear /= static_cast<index_t>(extents[d - 1]);
        }
        return result;
    }

    /// @brief Compute total size from extents
    template <size_type Rank>
    [[nodiscard]] static constexpr size_type size(const nd_extent<Rank>& extents) noexcept {
        size_type result = 1;
        for (size_type d = 0; d < Rank; ++d) {
            result *= extents[d];
        }
        return result;
    }
};

/// @brief Column-major (Fortran-style) layout - first dimension varies fastest
struct column_major {
    /// @brief Linearize ND index to 1D
    template <size_type Rank>
    [[nodiscard]] static constexpr index_t linearize(
        const nd_index<Rank>& idx, const nd_extent<Rank>& extents) noexcept {
        index_t result = 0;
        index_t stride = 1;
        for (size_type d = 0; d < Rank; ++d) {
            result += idx[d] * stride;
            stride *= static_cast<index_t>(extents[d]);
        }
        return result;
    }

    /// @brief Delinearize 1D index to ND
    template <size_type Rank>
    [[nodiscard]] static constexpr nd_index<Rank> delinearize(
        index_t linear, const nd_extent<Rank>& extents) noexcept {
        nd_index<Rank> result{};
        for (size_type d = 0; d < Rank; ++d) {
            result[d] = linear % static_cast<index_t>(extents[d]);
            linear /= static_cast<index_t>(extents[d]);
        }
        return result;
    }

    /// @brief Compute total size from extents
    template <size_type Rank>
    [[nodiscard]] static constexpr size_type size(const nd_extent<Rank>& extents) noexcept {
        return row_major::size(extents);  // Same computation
    }
};

// =============================================================================
// ND Partition Map
// =============================================================================

/// @brief Partition map for ND tensors (partitions along one dimension)
/// @tparam PartitionPolicy The 1D partition policy to use
/// @tparam Rank Number of dimensions
template <typename PartitionPolicy, size_type Rank>
class nd_partition_map {
public:
    /// @brief Construct ND partition map
    /// @param extents Global extents
    /// @param partition_dim Dimension along which to partition (0 = first)
    /// @param num_ranks Number of ranks
    /// @param my_rank This rank's ID
    constexpr nd_partition_map(const nd_extent<Rank>& extents, size_type partition_dim,
                               rank_t num_ranks, rank_t my_rank) noexcept
        : global_extents_{extents}
        , partition_dim_{partition_dim}
        , num_ranks_{num_ranks}
        , my_rank_{my_rank}
        , partition_1d_{extents[partition_dim], num_ranks, my_rank} {
        // Compute local extents
        local_extents_ = extents;
        local_extents_[partition_dim_] = partition_1d_.local_size();

        // Compute local offset (in the partitioned dimension)
        local_offset_ = {};
        local_offset_[partition_dim_] = partition_1d_.local_offset();
    }

    // =========================================================================
    // Extent Queries
    // =========================================================================

    /// @brief Get global extents
    [[nodiscard]] constexpr const nd_extent<Rank>& global_extents() const noexcept {
        return global_extents_;
    }

    /// @brief Get local extents for this rank
    [[nodiscard]] constexpr const nd_extent<Rank>& local_extents() const noexcept {
        return local_extents_;
    }

    /// @brief Get global extent for a dimension
    [[nodiscard]] constexpr size_type global_extent(size_type d) const noexcept {
        return global_extents_[d];
    }

    /// @brief Get local extent for a dimension
    [[nodiscard]] constexpr size_type local_extent(size_type d) const noexcept {
        return local_extents_[d];
    }

    /// @brief Get total global size
    [[nodiscard]] constexpr size_type global_size() const noexcept {
        return row_major::size(global_extents_);
    }

    /// @brief Get local size
    [[nodiscard]] constexpr size_type local_size() const noexcept {
        return row_major::size(local_extents_);
    }

    /// @brief Get local size for a specific rank
    [[nodiscard]] constexpr size_type local_size_for_rank(rank_t r) const noexcept {
        nd_extent<Rank> r_extents = global_extents_;
        r_extents[partition_dim_] = partition_1d_.local_size(r);
        return row_major::size(r_extents);
    }

    // =========================================================================
    // Ownership and Index Translation
    // =========================================================================

    /// @brief Get owner rank for global ND index
    [[nodiscard]] constexpr rank_t owner(const nd_index<Rank>& global_idx) const noexcept {
        return partition_1d_.owner(global_idx[partition_dim_]);
    }

    /// @brief Check if global ND index is local
    [[nodiscard]] constexpr bool is_local(const nd_index<Rank>& global_idx) const noexcept {
        return partition_1d_.is_local(global_idx[partition_dim_]);
    }

    /// @brief Convert global ND index to local ND index
    [[nodiscard]] constexpr nd_index<Rank> to_local(const nd_index<Rank>& global_idx) const noexcept {
        nd_index<Rank> result = global_idx;
        result[partition_dim_] = partition_1d_.to_local(global_idx[partition_dim_]);
        return result;
    }

    /// @brief Convert local ND index to global ND index
    [[nodiscard]] constexpr nd_index<Rank> to_global(const nd_index<Rank>& local_idx) const noexcept {
        nd_index<Rank> result = local_idx;
        result[partition_dim_] = partition_1d_.to_global(local_idx[partition_dim_]);
        return result;
    }

    /// @brief Get the global offset of local data
    [[nodiscard]] constexpr const nd_index<Rank>& local_offset() const noexcept {
        return local_offset_;
    }

    /// @brief Get the partition dimension
    [[nodiscard]] constexpr size_type partition_dim() const noexcept {
        return partition_dim_;
    }

    /// @brief Get number of ranks
    [[nodiscard]] constexpr rank_t num_ranks() const noexcept {
        return num_ranks_;
    }

    /// @brief Get this rank
    [[nodiscard]] constexpr rank_t my_rank() const noexcept {
        return my_rank_;
    }

private:
    nd_extent<Rank> global_extents_;
    nd_extent<Rank> local_extents_;
    nd_index<Rank> local_offset_;
    size_type partition_dim_;
    rank_t num_ranks_;
    rank_t my_rank_;
    partition_map<PartitionPolicy> partition_1d_;
};

// =============================================================================
// Distributed Tensor
// =============================================================================

/// @brief N-dimensional distributed tensor container
/// @tparam T Element type (must satisfy Transportable)
/// @tparam Rank Number of dimensions
/// @tparam Policies... Policy pack (partition, placement, consistency, execution, error)
/// @details A distributed N-dimensional array that partitions data across ranks
///          along one dimension (default: first dimension).
///
/// @par Design Rationale:
/// ND/tensor is MVP-critical because:
/// - Scientific computing heavily uses multi-dimensional arrays
/// - Domain decomposition naturally maps to tensor slicing
/// - Stencil computations require ND halo exchange
///
/// @par Layout:
/// Default layout is row-major (C-style). The last dimension varies fastest.
///
/// @par Partitioning:
/// By default, partitions along dimension 0. This can be configured via
/// the partition_dim constructor parameter.
///
/// @par Example Usage:
/// @code
/// // Create 100x100 matrix distributed across 4 ranks
/// dtl::distributed_tensor<double, 2> matrix({100, 100}, 0, 4, 1);  // 4 ranks, I'm rank 1
///
/// // Access local elements
/// for (size_t i = 0; i < matrix.local_extent(0); ++i) {
///     for (size_t j = 0; j < matrix.local_extent(1); ++j) {
///         matrix.local(i, j) = compute(i, j);
///     }
/// }
/// @endcode
template <typename T, size_type Rank, typename... Policies>
class distributed_tensor {
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

    // ========================================================================
    // Type Aliases
    // ========================================================================

    /// @brief Element type
    using value_type = T;

    /// @brief Allocator type (selected based on placement policy)
    using allocator_type = select_allocator_t<T, placement_policy>;

    /// @brief Storage type (std::vector with selected allocator)
    using storage_type = std::vector<T, allocator_type>;

    /// @brief ND index type
    using index_type = nd_index<Rank>;

    /// @brief ND extent type
    using extent_type = nd_extent<Rank>;

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

    /// @brief Layout policy (row-major by default)
    using layout_type = row_major;

    /// @brief ND partition map type
    using nd_partition_map_type = nd_partition_map<partition_policy, Rank>;

    /// @brief Local view type
    using local_view_type = dtl::local_view<T>;

    /// @brief Const local view type
    using const_local_view_type = dtl::local_view<const T>;

    /// @brief Global view type
    using global_view_type = dtl::global_view<distributed_tensor>;

    /// @brief Const global view type
    using const_global_view_type = dtl::global_view<const distributed_tensor>;

    /// @brief Segmented view type
    using segmented_view_type = dtl::segmented_view<distributed_tensor>;

    /// @brief Const segmented view type
    using const_segmented_view_type = dtl::segmented_view<const distributed_tensor>;

    /// @brief Extents type (required by DistributedTensor concept)
    using extents_type = extent_type;

    /// @brief Number of dimensions
    static constexpr size_type tensor_rank = Rank;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (empty tensor)
    distributed_tensor() noexcept
        : partition_{{}, 0, 1, 0}
        , my_rank_{0}
        , num_ranks_{1}
        , comm_handle_(handle::comm_handle::local()) {}

    // -------------------------------------------------------------------------
    // Context-Based Constructors (V1.3.0 - Preferred)
    // -------------------------------------------------------------------------

    /// @brief Construct with extents and context
    /// @param extents Global extents for each dimension
    /// @param ctx The execution context providing rank/size
    /// @param partition_dim Dimension to partition along (default: 0)
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    explicit distributed_tensor(const extent_type& extents, const Ctx& ctx,
                                 size_type partition_dim = 0)
        : partition_{extents, partition_dim, ctx.size(), ctx.rank()}
        , my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , local_data_(partition_.local_size())
        , device_id_(compute_device_id_from_ctx(ctx))
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    /// @brief Construct with extents, initial value, and context
    /// @param extents Global extents for each dimension
    /// @param value Initial value for all elements
    /// @param ctx The execution context providing rank/size
    /// @param partition_dim Dimension to partition along (default: 0)
    /// @since 0.1.0
    template <typename Ctx>
        requires requires(const Ctx& c) {
            { c.rank() } -> std::convertible_to<rank_t>;
            { c.size() } -> std::convertible_to<rank_t>;
        }
    distributed_tensor(const extent_type& extents, const T& value, const Ctx& ctx,
                       size_type partition_dim = 0)
        : partition_{extents, partition_dim, ctx.size(), ctx.rank()}
        , my_rank_{ctx.rank()}
        , num_ranks_{ctx.size()}
        , local_data_(partition_.local_size(), value)
        , device_id_(compute_device_id_from_ctx(ctx))
        , comm_handle_(handle::make_comm_handle(ctx)) {}

    // -------------------------------------------------------------------------
    // Legacy Constructors (Deprecated - will be removed in V2.0.0)
    // -------------------------------------------------------------------------

    /// @brief Construct with extents and distribution info
    /// @param extents Global extents for each dimension
    /// @param partition_dim Dimension to partition along (default: 0)
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    /// @deprecated Use distributed_tensor(extents, ctx, partition_dim) instead
    [[deprecated("Use distributed_tensor(extents, ctx, partition_dim) instead - will be removed in V2.0.0")]]
    distributed_tensor(const extent_type& extents, size_type partition_dim,
                       rank_t num_ranks, rank_t my_rank)
        : partition_{extents, partition_dim, num_ranks, my_rank}
        , my_rank_{my_rank}
        , num_ranks_{num_ranks}
        , local_data_(partition_.local_size())
        , comm_handle_(num_ranks > 1
            ? handle::comm_handle::unbound(my_rank, num_ranks)
            : handle::comm_handle::local()) {}

    /// @brief Construct with extents (default partition_dim=0)
    /// @param extents Global extents for each dimension
    /// @param num_ranks Number of ranks in the communicator
    /// @param my_rank This rank's ID
    /// @deprecated Use distributed_tensor(extents, ctx) instead
    [[deprecated("Use distributed_tensor(extents, ctx) instead - will be removed in V2.0.0")]]
    distributed_tensor(const extent_type& extents, rank_t num_ranks, rank_t my_rank)
        : distributed_tensor{extents, size_type{0}, num_ranks, my_rank} {}

    /// @brief Construct with extents and initial value
    /// @param extents Global extents for each dimension
    /// @param partition_dim Dimension to partition along
    /// @param num_ranks Number of ranks
    /// @param my_rank This rank's ID
    /// @param value Initial value for all elements
    /// @deprecated Use distributed_tensor(extents, value, ctx, partition_dim) instead
    [[deprecated("Use distributed_tensor(extents, value, ctx, partition_dim) instead - will be removed in V2.0.0")]]
    distributed_tensor(const extent_type& extents, size_type partition_dim,
                       rank_t num_ranks, rank_t my_rank, const T& value)
        : partition_{extents, partition_dim, num_ranks, my_rank}
        , my_rank_{my_rank}
        , num_ranks_{num_ranks}
        , local_data_(partition_.local_size(), value)
        , comm_handle_(num_ranks > 1
            ? handle::comm_handle::unbound(my_rank, num_ranks)
            : handle::comm_handle::local()) {}

    /// @brief Factory function returning result
    static result<distributed_tensor> create(const extent_type& extents,
                                              size_type partition_dim,
                                              rank_t num_ranks, rank_t my_rank) {
        try {
            return result<distributed_tensor>::success(
                distributed_tensor{extents, partition_dim, num_ranks, my_rank});
        } catch (const std::exception& e) {
            return result<distributed_tensor>::failure(
                status{status_code::allocation_failed});
        }
    }

    // ========================================================================
    // Extent Queries
    // ========================================================================

    /// @brief Get rank (number of dimensions)
    [[nodiscard]] static constexpr size_type rank() noexcept {
        return tensor_rank;
    }

    /// @brief Get global extents
    [[nodiscard]] const extent_type& extents() const noexcept {
        return partition_.global_extents();
    }

    /// @brief Get global extents (DistributedTensor concept requirement)
    /// @details Alias for extents() to satisfy the DistributedTensor concept.
    /// @since 0.1.0
    [[nodiscard]] extent_type global_extents() const noexcept {
        return partition_.global_extents();
    }

    /// @brief Get global extent for dimension d
    [[nodiscard]] size_type extent(size_type d) const noexcept {
        return partition_.global_extent(d);
    }

    /// @brief Get local extents for this rank
    [[nodiscard]] const extent_type& local_extents() const noexcept {
        return partition_.local_extents();
    }

    /// @brief Get local extent for dimension d
    [[nodiscard]] size_type local_extent(size_type d) const noexcept {
        return partition_.local_extent(d);
    }

    /// @brief Get total global size (product of all extents)
    [[nodiscard]] size_type size() const noexcept {
        return partition_.global_size();
    }

    /// @brief Alias for size()
    [[nodiscard]] size_type global_size() const noexcept {
        return partition_.global_size();
    }

    /// @brief Get local size (elements on this rank)
    [[nodiscard]] size_type local_size() const noexcept {
        return partition_.local_size();
    }

    /// @brief Get local size for a specific rank
    [[nodiscard]] size_type local_size_for_rank(rank_t r) const noexcept {
        return partition_.local_size_for_rank(r);
    }

    /// @brief Check if globally empty
    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    // ========================================================================
    // Local Element Access (ND Indexing)
    // ========================================================================

    /// @brief Access local element by ND index
    /// @param local_idx Local ND index
    [[nodiscard]] reference local(const index_type& local_idx) noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_data_[static_cast<size_type>(layout_type::linearize(local_idx, partition_.local_extents()))];
    }

    /// @brief Access local element by ND index (const)
    [[nodiscard]] const_reference local(const index_type& local_idx) const noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_data_[static_cast<size_type>(layout_type::linearize(local_idx, partition_.local_extents()))];
    }

    /// @brief Access local element by variadic indices
    template <typename... Indices>
        requires (sizeof...(Indices) == Rank) &&
                 (std::convertible_to<Indices, index_t> && ...)
    [[nodiscard]] reference local(Indices... indices) noexcept
        requires (placement_policy::is_host_accessible()) {
        return local(index_type{static_cast<index_t>(indices)...});
    }

    /// @brief Access local element by variadic indices (const)
    template <typename... Indices>
        requires (sizeof...(Indices) == Rank) &&
                 (std::convertible_to<Indices, index_t> && ...)
    [[nodiscard]] const_reference local(Indices... indices) const noexcept
        requires (placement_policy::is_host_accessible()) {
        return local(index_type{static_cast<index_t>(indices)...});
    }

    /// @brief Access local element by linear index
    [[nodiscard]] reference local_linear(size_type linear_idx) noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_data_[linear_idx];
    }

    /// @brief Access local element by linear index (const)
    [[nodiscard]] const_reference local_linear(size_type linear_idx) const noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_data_[linear_idx];
    }

    /// @brief operator() for ND local access (commonly expected syntax)
    template <typename... Indices>
        requires (sizeof...(Indices) == Rank)
    [[nodiscard]] reference operator()(Indices... indices) noexcept
        requires (placement_policy::is_host_accessible()) {
        return local(indices...);
    }

    /// @brief operator() for ND local access (const)
    template <typename... Indices>
        requires (sizeof...(Indices) == Rank)
    [[nodiscard]] const_reference operator()(Indices... indices) const noexcept
        requires (placement_policy::is_host_accessible()) {
        return local(indices...);
    }

    // ========================================================================
    // Global Element Access (Returns remote_ref)
    // ========================================================================

    /// @brief Access global element by ND index (returns remote_ref)
    /// @param global_idx Global ND index
    [[nodiscard]] remote_ref<T> global(const index_type& global_idx) {
        rank_t owner_rank = partition_.owner(global_idx);
        index_t linear = layout_type::linearize(global_idx, partition_.global_extents());
        if (owner_rank == my_rank_) {
            auto local_idx = partition_.to_local(global_idx);
            pointer ptr = local_data_.data()
                        + static_cast<size_type>(layout_type::linearize(
                              local_idx, partition_.local_extents()));
            return remote_ref<T>{owner_rank, linear, ptr};
        } else {
            return remote_ref<T>{owner_rank, linear, nullptr};
        }
    }

    /// @brief Access global element by ND index (const)
    [[nodiscard]] remote_ref<const T> global(const index_type& global_idx) const {
        rank_t owner_rank = partition_.owner(global_idx);
        index_t linear = layout_type::linearize(global_idx, partition_.global_extents());
        if (owner_rank == my_rank_) {
            auto local_idx = partition_.to_local(global_idx);
            const_pointer ptr = local_data_.data()
                              + static_cast<size_type>(layout_type::linearize(
                                    local_idx, partition_.local_extents()));
            return remote_ref<const T>{owner_rank, linear, ptr};
        } else {
            return remote_ref<const T>{owner_rank, linear, nullptr};
        }
    }

    // ========================================================================
    // Views
    // ========================================================================

    /// @brief Get local view (linear, STL-compatible)
    [[nodiscard]] local_view_type local_view() noexcept
        requires (placement_policy::is_host_accessible()) {
        return local_view_type{local_data_.data(), local_data_.size()};
    }

    /// @brief Get const local view
    [[nodiscard]] const_local_view_type local_view() const noexcept
        requires (placement_policy::is_host_accessible()) {
        return const_local_view_type{local_data_.data(), local_data_.size()};
    }

    /// @brief Get local data span
    [[nodiscard]] std::span<T> local_span() noexcept
        requires (placement_policy::is_host_accessible()) {
        return std::span<T>{local_data_};
    }

    /// @brief Get const local data span
    [[nodiscard]] std::span<const T> local_span() const noexcept
        requires (placement_policy::is_host_accessible()) {
        return std::span<const T>{local_data_};
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

    /// @brief Get pointer to local data
    [[nodiscard]] pointer local_data() noexcept {
        return local_data_.data();
    }

    /// @brief Get const pointer to local data
    [[nodiscard]] const_pointer local_data() const noexcept {
        return local_data_.data();
    }

    /// @brief Get global view (returns remote_ref for all elements)
    /// @details Provides access to all elements in the tensor using linearized
    ///          global indices. Satisfies the DistributedContainer concept.
    /// @since 0.1.0
    [[nodiscard]] global_view_type global_view() noexcept {
        return global_view_type{*this};
    }

    /// @brief Get const global view
    [[nodiscard]] const_global_view_type global_view() const noexcept {
        return const_global_view_type{*this};
    }

    /// @brief Get segmented view (for distributed iteration)
    /// @details Provides iteration over segments (partitions) of the tensor.
    ///          Satisfies the DistributedContainer concept.
    /// @since 0.1.0
    [[nodiscard]] segmented_view_type segmented_view() noexcept {
        return segmented_view_type{*this};
    }

    /// @brief Get const segmented view
    [[nodiscard]] const_segmented_view_type segmented_view() const noexcept {
        return const_segmented_view_type{*this};
    }

    // ========================================================================
    // Distribution Queries
    // ========================================================================

    /// @brief Get number of ranks
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return num_ranks_;
    }

    /// @brief Get this rank's ID
    [[nodiscard]] rank_t my_rank() const noexcept {
        return my_rank_;
    }

    /// @brief Get the partition dimension
    [[nodiscard]] size_type partition_dim() const noexcept {
        return partition_.partition_dim();
    }

    /// @brief Check if global ND index is local
    [[nodiscard]] bool is_local(const index_type& global_idx) const noexcept {
        return partition_.is_local(global_idx);
    }

    /// @brief Check if linearized global index is local
    /// @details Converts linearized index to ND index before checking locality.
    ///          Required for global_view compatibility.
    /// @since 0.1.0
    [[nodiscard]] bool is_local(index_t linear_idx) const noexcept {
        auto nd_idx = layout_type::delinearize(linear_idx, partition_.global_extents());
        return partition_.is_local(nd_idx);
    }

    /// @brief Get owner rank for global ND index
    [[nodiscard]] rank_t owner(const index_type& global_idx) const noexcept {
        return partition_.owner(global_idx);
    }

    /// @brief Get owner rank for linearized global index
    /// @details Converts linearized index to ND index before determining owner.
    ///          Required for global_view compatibility.
    /// @since 0.1.0
    [[nodiscard]] rank_t owner(index_t linear_idx) const noexcept {
        auto nd_idx = layout_type::delinearize(linear_idx, partition_.global_extents());
        return partition_.owner(nd_idx);
    }

    /// @brief Convert global ND index to local ND index
    [[nodiscard]] index_type to_local(const index_type& global_idx) const noexcept {
        return partition_.to_local(global_idx);
    }

    /// @brief Convert linearized global index to linearized local index
    /// @details Converts linearized index to ND, translates to local ND,
    ///          then linearizes again. Required for global_view compatibility.
    /// @since 0.1.0
    [[nodiscard]] index_t to_local(index_t linear_idx) const noexcept {
        auto nd_global = layout_type::delinearize(linear_idx, partition_.global_extents());
        auto nd_local = partition_.to_local(nd_global);
        return layout_type::linearize(nd_local, partition_.local_extents());
    }

    /// @brief Convert local ND index to global ND index
    [[nodiscard]] index_type to_global(const index_type& local_idx) const noexcept {
        return partition_.to_global(local_idx);
    }

    /// @brief Convert linearized local index to linearized global index
    /// @details Required for global_view compatibility.
    /// @since 0.1.0
    [[nodiscard]] index_t to_global(index_t local_linear_idx) const noexcept {
        auto nd_local = layout_type::delinearize(local_linear_idx, partition_.local_extents());
        auto nd_global = partition_.to_global(nd_local);
        return layout_type::linearize(nd_global, partition_.global_extents());
    }

    /// @brief Get the global offset of local data
    [[nodiscard]] const index_type& global_offset() const noexcept {
        return partition_.local_offset();
    }

    /// @brief Get the ND partition map
    [[nodiscard]] const nd_partition_map_type& partition() const noexcept {
        return partition_;
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// @brief Collective barrier
    result<void> barrier() {
        return comm_handle_.barrier();
    }

    [[nodiscard]] handle::comm_handle communicator_handle() const {
        return comm_handle_;
    }

    /// @brief Memory fence
    result<void> fence() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return result<void>::success();
    }

    // ========================================================================
    // Structural Operations
    // ========================================================================

    /// @brief Resize tensor (collective)
    result<void> resize(const extent_type& new_extents) {
        if (num_ranks_ > 1 && !comm_handle_.has_collective_path()) {
            return result<void>::failure(status{
                status_code::invalid_state,
                my_rank_,
                "distributed_tensor::resize requires a communicator-backed collective path when num_ranks()>1"});
        }

        try {
            partition_ = nd_partition_map_type{new_extents, partition_.partition_dim(),
                                               num_ranks_, my_rank_};
            local_data_.resize(partition_.local_size());
            sync_state_.mark_global_dirty();
            return result<void>::success();
        } catch (const std::exception& e) {
            return result<void>::failure(
                status{status_code::allocation_failed});
        }
    }

    /// @brief Check whether local storage size matches tensor partition metadata
    [[nodiscard]] bool structural_metadata_consistent() const noexcept {
        return local_data_.size() == partition_.local_size();
    }

    /// @brief Atomically replace local partition storage while preserving extents
    /// @param new_local_data Replacement local partition payload
    /// @return Result indicating success or metadata mismatch
    result<void> replace_local_partition(storage_type new_local_data) {
        if (new_local_data.size() != partition_.local_size()) {
            return result<void>::failure(status{
                status_code::invalid_argument,
                no_rank,
                "replace_local_partition size mismatch with current tensor partition metadata"});
        }

        local_data_.swap(new_local_data);
        sync_state_.mark_global_dirty();
        return result<void>::success();
    }

    /// @brief Atomically replace local partition storage and update global extents
    /// @param new_local_data Replacement local partition payload
    /// @param new_extents New global extents
    /// @return Result indicating success or metadata mismatch
    result<void> replace_local_partition(storage_type new_local_data,
                                         const extent_type& new_extents) {
        nd_partition_map_type new_partition{new_extents, partition_.partition_dim(),
                                            num_ranks_, my_rank_};
        if (new_local_data.size() != new_partition.local_size()) {
            return result<void>::failure(status{
                status_code::invalid_argument,
                no_rank,
                "replace_local_partition size mismatch for new tensor extents"});
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
                "swap_local_storage size mismatch with current tensor partition metadata"});
        }

        local_data_.swap(external_storage);
        sync_state_.mark_global_dirty();
        return result<void>::success();
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

    nd_partition_map_type partition_;
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

template <typename T, size_type Rank, typename... Policies>
struct is_distributed_container<distributed_tensor<T, Rank, Policies...>> : std::true_type {};

template <typename T, size_type Rank, typename... Policies>
struct is_distributed_tensor<distributed_tensor<T, Rank, Policies...>> : std::true_type {};

// =============================================================================
// Convenience Aliases
// =============================================================================

/// @brief Type alias for 1D tensor
template <typename T, typename... Policies>
using tensor1d = distributed_tensor<T, 1, Policies...>;

/// @brief Type alias for 2D tensor (matrix)
template <typename T, typename... Policies>
using tensor2d = distributed_tensor<T, 2, Policies...>;

/// @brief Type alias for 3D tensor
template <typename T, typename... Policies>
using tensor3d = distributed_tensor<T, 3, Policies...>;

/// @brief Type alias for distributed matrix
template <typename T, typename... Policies>
using distributed_matrix = tensor2d<T, Policies...>;

}  // namespace dtl
