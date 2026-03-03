// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file global_view.hpp
/// @brief Global view providing access to entire distributed container
/// @details May communicate - returns remote_ref<T> for remote elements.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/fwd.hpp>
#include <dtl/views/remote_ref.hpp>
#include <dtl/index/partition_map.hpp>
#include <dtl/iterators/global_iterator.hpp>
#include <dtl/runtime/backend_discovery.hpp>

#include <type_traits>

namespace dtl {

/// @brief View providing global access to a distributed container
/// @tparam Container The distributed container type
/// @details This view provides access to all elements in the distributed
///          container using global indices. ALL accesses return remote_ref<T>
///          to maintain syntactically loud remote access.
///
/// @warning Global indexing can be expensive for remote elements.
///          Prefer local_view() for local operations and segmented_view()
///          for distributed iteration.
///
/// @par Design Notes:
/// Even local elements return remote_ref to maintain API consistency
/// and make distribution explicit. Use is_local() to check if communication
/// will occur.
///
/// @par Usage:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto global = vec.global_view();
/// // Access by global index - always returns remote_ref
/// auto ref = global[500];  // Returns remote_ref<int>
/// if (ref.is_local()) {
///     int val = ref.get().value();  // No communication, but explicit
/// } else {
///     int val = ref.get().value();  // Communication occurs
/// }
/// @endcode
template <typename Container>
class global_view {
public:
    /// @brief Value type of the container
    using value_type = typename Container::value_type;

    /// @brief Reference type (always remote_ref)
    /// When Container is const-qualified, reference is also const to prevent mutation
    using reference = std::conditional_t<std::is_const_v<Container>,
                                         remote_ref<const value_type>,
                                         remote_ref<value_type>>;

    /// @brief Const reference type
    using const_reference = remote_ref<const value_type>;

    /// @brief Size type
    using size_type = dtl::size_type;

    /// @brief Index type
    using index_type = dtl::index_t;

    /// @brief Iterator type (global_iterator for distributed traversal)
    using iterator = global_iterator<Container>;

    /// @brief Const iterator type
    using const_iterator = const_global_iterator<Container>;

    // =========================================================================
    // Constructors
    // =========================================================================

    /// @brief Construct from container reference
    /// @param container The distributed container to view
    explicit global_view(Container& container) noexcept
        : container_{&container}
        , remote_transport_available_{runtime::has_any_comm_backend()} {}

    // =========================================================================
    // Size Queries
    // =========================================================================

    /// @brief Get total global size of the container
    [[nodiscard]] size_type size() const noexcept {
        return container_->size();
    }

    /// @brief Check if container is empty
    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    /// @brief Access element by global index
    /// @param global_idx Global index (0-based across all ranks)
    /// @return remote_ref to the element (may be local or remote)
    /// @note Use is_local() to check if communication will be required
    [[nodiscard]] reference operator[](index_type global_idx) {
        rank_t owner_rank = owner(global_idx);
        if (owner_rank == my_rank()) {
            // Local element - provide direct pointer
            index_type local_idx = to_local(global_idx);
            auto* ptr = container_->local_data() + local_idx;
            if constexpr (std::is_const_v<Container>) {
                return remote_ref<const value_type>{owner_rank, global_idx, ptr,
                    remote_transport_available_};
            } else {
                return remote_ref<value_type>{owner_rank, global_idx, ptr,
                    remote_transport_available_};
            }
        } else {
            // Remote element - no local pointer
            if constexpr (std::is_const_v<Container>) {
                return remote_ref<const value_type>{owner_rank, global_idx, nullptr,
                    remote_transport_available_};
            } else {
                return remote_ref<value_type>{owner_rank, global_idx, nullptr,
                    remote_transport_available_};
            }
        }
    }

    /// @brief Access element by global index (const)
    [[nodiscard]] const_reference operator[](index_type global_idx) const {
        rank_t owner_rank = owner(global_idx);
        if (owner_rank == my_rank()) {
            index_type local_idx = to_local(global_idx);
            const value_type* ptr = container_->local_data() + local_idx;
            return remote_ref<const value_type>{owner_rank, global_idx, ptr,
                remote_transport_available_};
        } else {
            return remote_ref<const value_type>{owner_rank, global_idx, nullptr,
                remote_transport_available_};
        }
    }

    // =========================================================================
    // Iterators (global_iterator integration — T07)
    // =========================================================================

    /// @brief Get iterator to beginning of global index space
    /// @warning Global iteration can be expensive. Prefer segmented_view.
    [[nodiscard]] iterator begin() noexcept {
        return iterator{container_, 0};
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator begin() const noexcept {
        return const_iterator{container_, 0};
    }

    /// @brief Get const iterator to beginning
    [[nodiscard]] const_iterator cbegin() const noexcept {
        return begin();
    }

    /// @brief Get iterator past end of global index space
    [[nodiscard]] iterator end() noexcept {
        return iterator{container_, static_cast<index_type>(size())};
    }

    /// @brief Get const iterator past end
    [[nodiscard]] const_iterator end() const noexcept {
        return const_iterator{container_, static_cast<index_type>(size())};
    }

    /// @brief Get const iterator past end
    [[nodiscard]] const_iterator cend() const noexcept {
        return end();
    }

    // =========================================================================
    // Distribution Queries
    // =========================================================================

    /// @brief Check if a global index is stored locally
    /// @param global_idx The global index to check
    /// @return true if the element is on this rank
    [[nodiscard]] bool is_local(index_type global_idx) const noexcept {
        return container_->is_local(global_idx);
    }

    /// @brief Get the rank that owns a global index
    /// @param global_idx The global index to query
    /// @return The owning rank
    [[nodiscard]] rank_t owner(index_type global_idx) const noexcept {
        return container_->owner(global_idx);
    }

    /// @brief Convert global index to local index
    /// @param global_idx The global index
    /// @return The local index on the owning rank
    /// @pre is_local(global_idx) must be true
    [[nodiscard]] index_type to_local(index_type global_idx) const noexcept {
        return container_->to_local(global_idx);
    }

    /// @brief Get number of ranks in the distribution
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return container_->num_ranks();
    }

    /// @brief Get current rank
    /// @note Uses my_rank() for compatibility with distributed_tensor
    ///       (which has a static rank() returning dimensionality)
    [[nodiscard]] rank_t my_rank() const noexcept {
        if constexpr (requires { container_->my_rank(); }) {
            return container_->my_rank();
        } else {
            return container_->rank();
        }
    }

    /// @brief Check whether remote transport is available for this view
    [[nodiscard]] bool remote_access_available() const noexcept {
        return remote_transport_available_;
    }

    // =========================================================================
    // Container Access
    // =========================================================================

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
    bool remote_transport_available_;
};

// =============================================================================
// Type Trait Specialization
// =============================================================================

template <typename Container>
struct is_global_view<global_view<Container>> : std::true_type {};

// =============================================================================
// Factory Function
// =============================================================================

/// @brief Create a global view from a container
/// @tparam Container The container type
/// @param container The container to view
/// @return global_view<Container>
template <typename Container>
[[nodiscard]] auto make_global_view(Container& container) {
    return global_view<Container>{container};
}

}  // namespace dtl
