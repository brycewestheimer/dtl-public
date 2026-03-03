// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file segmented_view.hpp
/// @brief Segmented view for efficient distributed iteration
/// @details Primary iteration substrate for distributed algorithms.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/views/local_view.hpp>

#include <iterator>
#include <type_traits>
#include <vector>

namespace dtl {

/// @brief Descriptor for a segment (one rank's portion of distributed data)
/// @tparam T The element type
/// @details Provides metadata about a segment and, for local segments,
///          direct access to the underlying data via local_view.
///
/// @par Design:
/// - Segments are cheap to copy (just metadata + pointer)
/// - Local segments have valid data pointer, remote segments have nullptr
/// - Provides local_view for STL-compatible iteration
template <typename T>
struct segment_descriptor {
    /// @brief Value type
    using value_type = std::remove_cv_t<T>;

    /// @brief Element type (may be const)
    using element_type = T;

    /// @brief Local view type for data access
    using local_view_type = local_view<T>;

    /// @brief The rank that owns this segment
    rank_t rank;

    /// @brief Global offset of the first element in this segment
    index_t global_offset;

    /// @brief The local view providing data access (empty if remote)
    local_view_type data;

    // =========================================================================
    // Query Methods
    // =========================================================================

    /// @brief Get the number of elements in this segment
    [[nodiscard]] constexpr size_type size() const noexcept {
        return data.size();
    }

    /// @brief Check if segment is empty
    [[nodiscard]] constexpr bool empty() const noexcept {
        return data.empty();
    }

    /// @brief Check if this segment is local (has valid data pointer)
    [[nodiscard]] constexpr bool is_local() const noexcept {
        return data.data() != nullptr;
    }

    /// @brief Check if this segment is remote (no local data access)
    [[nodiscard]] constexpr bool is_remote() const noexcept {
        return !is_local();
    }

    // =========================================================================
    // Iteration (delegates to local_view)
    // =========================================================================

    /// @brief Get iterator to beginning (only valid for local segments)
    [[nodiscard]] constexpr auto begin() noexcept {
        return data.begin();
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] constexpr auto begin() const noexcept {
        return data.begin();
    }

    /// @brief Get iterator to end
    [[nodiscard]] constexpr auto end() noexcept {
        return data.end();
    }

    /// @brief Get const iterator to end
    [[nodiscard]] constexpr auto end() const noexcept {
        return data.end();
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    /// @brief Access element by local index (only valid for local segments)
    [[nodiscard]] constexpr T& operator[](size_type idx) noexcept {
        return data[idx];
    }

    /// @brief Access element by local index (const)
    [[nodiscard]] constexpr const T& operator[](size_type idx) const noexcept {
        return data[idx];
    }

    /// @brief Convert local index within segment to global index
    [[nodiscard]] constexpr index_t to_global(size_type local_idx) const noexcept {
        return global_offset + static_cast<index_t>(local_idx);
    }
};

/// @brief Iterator over segments in a segmented view
/// @tparam Container The distributed container type
/// @tparam IsConst Whether this is a const iterator
template <typename Container, bool IsConst = false>
class segment_iterator {
public:
    using container_type = std::conditional_t<IsConst, const Container, Container>;
    using raw_value_type = typename std::remove_const_t<Container>::value_type;
    // Use const value_type if Container is const OR if IsConst is true
    static constexpr bool use_const = IsConst || std::is_const_v<Container>;
    using value_type = std::conditional_t<use_const, const raw_value_type, raw_value_type>;
    using segment_type = segment_descriptor<value_type>;

    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = segment_type*;
    using reference = segment_type;

    // =========================================================================
    // Constructors
    // =========================================================================

    /// @brief Default constructor
    constexpr segment_iterator() noexcept = default;

    /// @brief Construct segment iterator
    /// @param container The distributed container
    /// @param segment_idx Current segment (rank) index
    constexpr segment_iterator(container_type* container, rank_t segment_idx) noexcept
        : container_{container}
        , segment_idx_{segment_idx}
        , offset_cache_{nullptr} {}

    /// @brief Construct segment iterator with offset cache
    /// @param container The distributed container
    /// @param segment_idx Current segment (rank) index
    /// @param offset_cache Pointer to precomputed offset cache (O(1) lookup)
    constexpr segment_iterator(container_type* container, rank_t segment_idx,
                                const index_t* offset_cache) noexcept
        : container_{container}
        , segment_idx_{segment_idx}
        , offset_cache_{offset_cache} {}

    // =========================================================================
    // Iterator Operations
    // =========================================================================

    /// @brief Dereference to get current segment
    [[nodiscard]] segment_type operator*() const {
        return get_segment(segment_idx_);
    }

    /// @brief Pre-increment
    constexpr segment_iterator& operator++() noexcept {
        ++segment_idx_;
        return *this;
    }

    /// @brief Post-increment
    constexpr segment_iterator operator++(int) noexcept {
        segment_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Equality comparison
    [[nodiscard]] constexpr bool operator==(const segment_iterator& other) const noexcept {
        return container_ == other.container_ && segment_idx_ == other.segment_idx_;
    }

    /// @brief Inequality comparison
    [[nodiscard]] constexpr bool operator!=(const segment_iterator& other) const noexcept {
        return !(*this == other);
    }

    /// @brief Get current segment index
    [[nodiscard]] constexpr rank_t segment_index() const noexcept {
        return segment_idx_;
    }

private:
    container_type* container_ = nullptr;
    rank_t segment_idx_ = 0;
    const index_t* offset_cache_ = nullptr;  // Optional precomputed offset cache

    /// @brief Get offset for a rank - O(1) if cache available, O(p) otherwise
    [[nodiscard]] index_t get_offset_for_rank(rank_t rank) const noexcept {
        // Use cached offset if available (O(1))
        if (offset_cache_) {
            return offset_cache_[rank];
        }

        // Fallback: compute offset (O(p)) - for iterators created without cache
        rank_t num_ranks = container_->num_ranks();
        index_t offset = 0;
        for (rank_t r = 0; r < rank && r < num_ranks; ++r) {
            offset += static_cast<index_t>(container_->local_size_for_rank(r));
        }
        return offset;
    }

    /// @brief Construct segment descriptor for a given rank
    [[nodiscard]] segment_type get_segment(rank_t rank) const {
        if (!container_) {
            return segment_type{rank, 0, local_view<value_type>{}};
        }

        rank_t my_rank;
        if constexpr (requires { container_->my_rank(); }) {
            my_rank = container_->my_rank();
        } else {
            my_rank = container_->rank();
        }

        // Use cached offset lookup (O(1) with cache)
        index_t offset = get_offset_for_rank(rank);

        if (rank == my_rank) {
            // Local segment - provide data pointer
            auto* data_ptr = container_->local_data();
            size_type size = container_->local_size();
            return segment_type{
                rank,
                offset,
                local_view<value_type>{data_ptr, size, rank, offset}
            };
        } else {
            // Remote segment - just metadata
            size_type size = container_->local_size_for_rank(rank);
            return segment_type{
                rank,
                offset,
                local_view<value_type>{nullptr, size, rank, offset}
            };
        }
    }
};

/// @brief Segmented view for distributed iteration
/// @tparam Container The distributed container type
/// @details Provides iteration over segments (partitions) of a distributed
///          container. This is the primary iteration substrate for distributed
///          algorithms, enabling predictable bulk operations without hidden
///          communication.
///
/// @par Design Rationale:
/// Segmented iteration is preferred over global iterators because:
/// - Predictable communication patterns
/// - Better data locality
/// - Natural fit for SPMD programming
/// - Enables efficient vectorization within segments
/// - No hidden communication during iteration
///
/// @par Performance:
/// The segmented_view builds an offset cache at construction time (O(p) where
/// p is the number of ranks), enabling O(1) offset lookups during iteration.
/// This changes full iteration complexity from O(p^2) to O(p).
///
/// @par Iteration Model:
/// @code
/// distributed_vector<int> vec(10000, ctx);
/// for (auto segment : vec.segmented_view()) {
///     if (segment.is_local()) {
///         // Process local segment with full STL compatibility
///         std::transform(segment.begin(), segment.end(),
///                        segment.begin(), [](int x) { return x * 2; });
///     }
/// }
/// // Collective barrier after processing all segments
/// vec.barrier();
/// @endcode
///
/// @par Segment Properties:
/// - Each segment corresponds to one rank's partition
/// - Segments are visited in rank order (0, 1, 2, ...)
/// - Local segment provides direct memory access via local_view
/// - Remote segments have nullptr data (metadata only)
/// - num_segments() == num_ranks()
template <typename Container>
class segmented_view {
public:
    /// @brief Value type
    using value_type = typename Container::value_type;

    /// @brief Element type (const-qualified if Container is const)
    using element_type = std::conditional_t<
        std::is_const_v<Container>,
        const value_type,
        value_type>;

    /// @brief Segment descriptor type (respects Container constness)
    using segment_type = segment_descriptor<element_type>;

    /// @brief Const segment descriptor type
    using const_segment_type = segment_descriptor<const value_type>;

    /// @brief Iterator type (iterates over segments)
    using iterator = segment_iterator<Container, false>;

    /// @brief Const iterator type
    using const_iterator = segment_iterator<Container, true>;

    /// @brief Size type
    using size_type = dtl::size_type;

    // =========================================================================
    // Constructors
    // =========================================================================

    /// @brief Construct from container reference
    /// @param container The distributed container to view
    /// @details Builds offset cache at construction (O(p)) for O(1) lookup
    explicit segmented_view(Container& container) noexcept
        : container_{&container} {
        build_offset_cache();
    }

    // =========================================================================
    // Iterators
    // =========================================================================

    /// @brief Get iterator to first segment
    [[nodiscard]] iterator begin() noexcept {
        return iterator{container_, 0, offset_cache_.data()};
    }

    /// @brief Get const iterator to first segment
    [[nodiscard]] const_iterator begin() const noexcept {
        return const_iterator{container_, 0, offset_cache_.data()};
    }

    /// @brief Get const iterator to first segment
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return const_iterator{container_, 0, offset_cache_.data()};
    }

    /// @brief Get iterator past last segment
    [[nodiscard]] iterator end() noexcept {
        return iterator{container_, num_segments(), offset_cache_.data()};
    }

    /// @brief Get const iterator past last segment
    [[nodiscard]] const_iterator end() const noexcept {
        return const_iterator{container_, num_segments(), offset_cache_.data()};
    }

    /// @brief Get const iterator past last segment
    [[nodiscard]] const_iterator cend() const noexcept {
        return const_iterator{container_, num_segments(), offset_cache_.data()};
    }

    // =========================================================================
    // Segment Access
    // =========================================================================

    /// @brief Get number of segments (equal to number of ranks)
    [[nodiscard]] rank_t num_segments() const noexcept {
        return container_->num_ranks();
    }

    /// @brief Get the local segment (current rank's partition)
    [[nodiscard]] segment_type local_segment() noexcept {
        rank_t current_rank;
        if constexpr (requires { container_->my_rank(); }) {
            current_rank = container_->my_rank();
        } else {
            current_rank = container_->rank();
        }
        return *iterator{container_, current_rank, offset_cache_.data()};
    }

    /// @brief Get the local segment (const)
    [[nodiscard]] const_segment_type local_segment() const noexcept {
        rank_t current_rank;
        if constexpr (requires { container_->my_rank(); }) {
            current_rank = container_->my_rank();
        } else {
            current_rank = container_->rank();
        }
        return *const_iterator{container_, current_rank, offset_cache_.data()};
    }

    /// @brief Get segment for a specific rank
    /// @param rank The rank whose segment to get
    /// @return The segment for that rank
    [[nodiscard]] segment_type segment_for_rank(rank_t rank) noexcept {
        return *iterator{container_, rank, offset_cache_.data()};
    }

    /// @brief Get segment for a specific rank (const)
    [[nodiscard]] const_segment_type segment_for_rank(rank_t rank) const noexcept {
        return *const_iterator{container_, rank, offset_cache_.data()};
    }

    /// @brief Get precomputed offset for a rank (O(1))
    /// @param rank The rank to query
    /// @return Global offset of that rank's segment
    [[nodiscard]] index_t offset_for_rank(rank_t rank) const noexcept {
        return offset_cache_[static_cast<size_type>(rank)];
    }

    // =========================================================================
    // Size Queries
    // =========================================================================

    /// @brief Get total element count across all segments
    [[nodiscard]] size_type total_size() const noexcept {
        return container_->size();
    }

    /// @brief Get local segment size
    [[nodiscard]] size_type local_size() const noexcept {
        return container_->local_size();
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /// @brief Apply function to local segment only
    /// @tparam F Function type (element -> void)
    /// @param f Function to apply to each element
    template <typename F>
    void for_each_local(F&& f) {
        // Single-rank short-circuit: skip segment machinery entirely
        if (container_->num_ranks() == 1) {
            auto* data_ptr = container_->local_data();
            const size_type sz = container_->local_size();
            for (size_type i = 0; i < sz; ++i) {
                std::forward<F>(f)(data_ptr[i]);
            }
            return;
        }
        auto seg = local_segment();
        if (seg.is_local()) {
            for (auto& elem : seg) {
                std::forward<F>(f)(elem);
            }
        }
    }

    /// @brief Apply function to each local segment
    /// @tparam F Function type (segment -> void)
    /// @param f Function to apply to each segment
    /// @note Only processes segments that are local to this rank
    template <typename F>
    void for_each_segment(F&& f) {
        // Single-rank short-circuit: exactly one segment, always local
        if (container_->num_ranks() == 1) {
            auto seg = local_segment();
            std::forward<F>(f)(seg);
            return;
        }
        for (auto seg : *this) {
            if (seg.is_local()) {
                std::forward<F>(f)(seg);
            }
        }
    }

    /// @brief Get reference to the underlying container
    [[nodiscard]] Container& container() noexcept {
        return *container_;
    }

    /// @brief Get const reference to the underlying container
    [[nodiscard]] const Container& container() const noexcept {
        return *container_;
    }

private:
    Container* container_;
    std::vector<index_t> offset_cache_;  // Precomputed offsets for O(1) lookup

    /// @brief Build offset cache at construction (O(p) one-time cost)
    /// @details Precomputes global offset for each rank's segment, enabling
    ///          O(1) offset lookups during iteration instead of O(p) per lookup.
    void build_offset_cache() noexcept {
        const rank_t num_ranks = container_->num_ranks();
        const size_t cache_size =
            (num_ranks > 0) ? static_cast<size_t>(num_ranks) + 1 : 1;

        offset_cache_.resize(cache_size);
        offset_cache_[0] = 0;
        if (num_ranks <= 0) {
            return;
        }
        for (rank_t r = 0; r < num_ranks; ++r) {
            const auto idx = static_cast<size_t>(r);
            offset_cache_[idx + 1] =
                offset_cache_[idx] + static_cast<index_t>(container_->local_size_for_rank(r));
        }
    }
};

// =============================================================================
// Type Trait Specialization
// =============================================================================

template <typename Container>
struct is_segmented_view<segmented_view<Container>> : std::true_type {};

// =============================================================================
// Factory Function
// =============================================================================

/// @brief Create a segmented view from a container
/// @tparam Container The container type
/// @param container The container to view
/// @return segmented_view<Container>
template <typename Container>
[[nodiscard]] auto make_segmented_view(Container& container) {
    return segmented_view<Container>{container};
}

}  // namespace dtl
