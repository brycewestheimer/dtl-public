// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file global_iterator.hpp
/// @brief Iterator with global indexing (may communicate)
/// @details Provides global iteration but dereference returns remote_ref.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/views/remote_ref.hpp>

#include <iterator>

namespace dtl {

/// @brief Forward iterator for global iteration over distributed container
/// @tparam Container The distributed container type
/// @details Provides iteration using global indices. Dereferencing returns
///          remote_ref<T> to enforce explicit handling of remote access.
///
/// @par Iterator Category:
/// Forward iterator only (random access would be misleadingly expensive)
///
/// @warning Global iteration can be expensive. Each dereference of a remote
///          element may involve communication. Prefer local_view() or
///          segmented_view() for efficient iteration patterns.
///
/// @par Design Rationale:
/// - Only forward iterator to discourage random access patterns
/// - Returns remote_ref<T> to make remote access explicit
/// - Provided for convenience, not performance
///
/// @par Usage:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto global = vec.global_view();
/// for (auto it = global.begin(); it != global.end(); ++it) {
///     auto ref = *it;  // Returns remote_ref<int>
///     if (ref.is_local()) {
///         int val = ref.get().value();
///     }
/// }
/// @endcode
template <typename Container>
class global_iterator {
public:
    // ========================================================================
    // Iterator Type Aliases
    // ========================================================================

    /// @brief Forward iterator only (random access would be misleading)
    using iterator_category = std::forward_iterator_tag;

    using value_type = typename Container::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = void;  // No meaningful pointer for remote_ref

    /// @brief Element type (const-qualified if Container is const)
    using element_type = std::conditional_t<
        std::is_const_v<Container>,
        const value_type,
        value_type>;
    using reference = remote_ref<element_type>;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// @brief Default constructor (singular iterator)
    global_iterator() noexcept = default;

    /// @brief Construct from container and global index
    /// @param container The distributed container
    /// @param global_idx Starting global index
    global_iterator(Container* container, index_t global_idx) noexcept
        : container_{container}
        , global_idx_{global_idx} {}

    // ========================================================================
    // Dereference
    // ========================================================================

    /// @brief Dereference returns remote_ref
    [[nodiscard]] reference operator*() const {
        rank_t owner_rank = container_ ? container_->owner(global_idx_) : 0;
        if (container_ && container_->is_local(global_idx_)) {
            index_t local_idx = container_->to_local(global_idx_);
            element_type* ptr = container_->local_data() + local_idx;
            return remote_ref<element_type>{owner_rank, global_idx_, ptr};
        }
        return remote_ref<element_type>{owner_rank, global_idx_, nullptr};
    }

    // No operator-> because remote_ref doesn't support it

    // ========================================================================
    // Increment (Forward Iterator)
    // ========================================================================

    /// @brief Pre-increment
    global_iterator& operator++() noexcept {
        ++global_idx_;
        return *this;
    }

    /// @brief Post-increment
    global_iterator operator++(int) noexcept {
        global_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    [[nodiscard]] bool operator==(const global_iterator& other) const noexcept {
        return container_ == other.container_ && global_idx_ == other.global_idx_;
    }

    [[nodiscard]] bool operator!=(const global_iterator& other) const noexcept {
        return !(*this == other);
    }

    // ========================================================================
    // Query Operations
    // ========================================================================

    /// @brief Get the current global index
    [[nodiscard]] index_t global_index() const noexcept {
        return global_idx_;
    }

    /// @brief Check if current position is local
    [[nodiscard]] bool is_local() const noexcept {
        return container_ ? container_->is_local(global_idx_) : false;
    }

    /// @brief Get owner rank for current position
    [[nodiscard]] rank_t owner() const noexcept {
        return container_ ? container_->owner(global_idx_) : no_rank;
    }

private:
    Container* container_ = nullptr;
    index_t global_idx_ = 0;
};

/// @brief Const version of global iterator
template <typename Container>
class const_global_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename Container::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = remote_ref<const value_type>;

    const_global_iterator() noexcept = default;

    const_global_iterator(const Container* container, index_t global_idx) noexcept
        : container_{container}
        , global_idx_{global_idx} {}

    // Allow conversion from non-const
    const_global_iterator(const global_iterator<Container>& other) noexcept
        : container_{other.container_}
        , global_idx_{other.global_index()} {}

    [[nodiscard]] reference operator*() const {
        rank_t owner_rank = container_ ? container_->owner(global_idx_) : 0;
        if (container_ && container_->is_local(global_idx_)) {
            index_t local_idx = container_->to_local(global_idx_);
            const value_type* ptr = container_->local_data() + local_idx;
            return remote_ref<const value_type>{owner_rank, global_idx_, ptr};
        }
        return remote_ref<const value_type>{owner_rank, global_idx_, nullptr};
    }

    const_global_iterator& operator++() noexcept { ++global_idx_; return *this; }
    const_global_iterator operator++(int) noexcept { auto t = *this; ++(*this); return t; }

    [[nodiscard]] bool operator==(const const_global_iterator& o) const noexcept {
        return container_ == o.container_ && global_idx_ == o.global_idx_;
    }
    [[nodiscard]] bool operator!=(const const_global_iterator& o) const noexcept {
        return !(*this == o);
    }

    [[nodiscard]] index_t global_index() const noexcept { return global_idx_; }
    [[nodiscard]] bool is_local() const noexcept {
        return container_ ? container_->is_local(global_idx_) : false;
    }

private:
    const Container* container_ = nullptr;
    index_t global_idx_ = 0;
};

}  // namespace dtl
