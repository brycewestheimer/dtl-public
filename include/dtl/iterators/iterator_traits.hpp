// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file iterator_traits.hpp
/// @brief Traits for distributed iterator categories
/// @details Type traits for querying iterator properties.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/iterators/local_iterator.hpp>
#include <dtl/iterators/global_iterator.hpp>
#include <dtl/iterators/device_iterator.hpp>

#include <iterator>
#include <type_traits>

namespace dtl {

// ============================================================================
// Iterator Category Detection
// ============================================================================

/// @brief Check if an iterator is a local iterator (no communication)
template <typename T>
struct is_local_iterator : std::false_type {};

template <typename Container>
struct is_local_iterator<local_iterator<Container>> : std::true_type {};

template <typename Container>
struct is_local_iterator<const_local_iterator<Container>> : std::true_type {};

template <typename T>
inline constexpr bool is_local_iterator_v = is_local_iterator<T>::value;

/// @brief Check if an iterator is a global iterator (may communicate)
template <typename T>
struct is_global_iterator : std::false_type {};

template <typename Container>
struct is_global_iterator<global_iterator<Container>> : std::true_type {};

template <typename Container>
struct is_global_iterator<const_global_iterator<Container>> : std::true_type {};

template <typename T>
inline constexpr bool is_global_iterator_v = is_global_iterator<T>::value;

/// @brief Check if an iterator is a device iterator
template <typename T>
struct is_device_iterator : std::false_type {};

template <typename U>
struct is_device_iterator<device_iterator<U>> : std::true_type {};

template <typename T>
inline constexpr bool is_device_iterator_v = is_device_iterator<T>::value;

/// @brief Check if an iterator is any DTL iterator
template <typename T>
inline constexpr bool is_dtl_iterator_v =
    is_local_iterator_v<T> ||
    is_global_iterator_v<T> ||
    is_device_iterator_v<T>;

// ============================================================================
// Iterator Property Traits
// ============================================================================

/// @brief Traits for distributed iterators
template <typename Iterator>
struct distributed_iterator_traits {
    /// @brief Whether this iterator may communicate
    static constexpr bool may_communicate = false;

    /// @brief Whether this iterator is usable on GPU
    static constexpr bool is_device_usable = false;

    /// @brief Whether this iterator is STL-compatible
    static constexpr bool is_stl_compatible = true;

    /// @brief Whether dereference is O(1)
    static constexpr bool is_constant_time = true;
};

/// @brief Specialization for global_iterator
template <typename Container>
struct distributed_iterator_traits<global_iterator<Container>> {
    static constexpr bool may_communicate = true;
    static constexpr bool is_device_usable = false;
    static constexpr bool is_stl_compatible = false;  // Returns remote_ref
    static constexpr bool is_constant_time = false;   // Communication may occur
};

/// @brief Specialization for device_iterator
template <typename T>
struct distributed_iterator_traits<device_iterator<T>> {
    static constexpr bool may_communicate = false;
    static constexpr bool is_device_usable = true;
    static constexpr bool is_stl_compatible = true;  // In device code
    static constexpr bool is_constant_time = true;
};

// ============================================================================
// Iterator Concepts
// ============================================================================

/// @brief Concept for iterators that never communicate
template <typename T>
concept NonCommunicatingIterator =
    is_local_iterator_v<T> ||
    is_device_iterator_v<T> ||
    (!is_dtl_iterator_v<T> && std::forward_iterator<T>);

/// @brief Concept for iterators that may communicate
template <typename T>
concept CommunicatingIterator = is_global_iterator_v<T>;

// Note: DeviceIterator is defined in core/concepts.hpp

/// @brief Concept for STL-compatible distributed iterators
template <typename T>
concept StlCompatibleDistributedIterator =
    is_dtl_iterator_v<T> &&
    distributed_iterator_traits<T>::is_stl_compatible;

// ============================================================================
// Iterator Utilities
// ============================================================================

/// @brief Get the value type of an iterator range
template <typename Iterator>
using iter_value_t = typename std::iterator_traits<Iterator>::value_type;

/// @brief Get the reference type of an iterator
template <typename Iterator>
using iter_reference_t = typename std::iterator_traits<Iterator>::reference;

/// @brief Get the difference type of an iterator
template <typename Iterator>
using iter_difference_t = typename std::iterator_traits<Iterator>::difference_type;

/// @brief Compute distance between iterators
/// @note For global iterators, this is the global distance
template <typename Iterator>
[[nodiscard]] auto distributed_distance(Iterator first, Iterator last) {
    if constexpr (is_global_iterator_v<Iterator>) {
        return last.global_index() - first.global_index();
    } else {
        return std::distance(first, last);
    }
}

/// @brief Advance iterator by n positions
template <typename Iterator>
void distributed_advance(Iterator& it, iter_difference_t<Iterator> n) {
    if constexpr (std::random_access_iterator<Iterator>) {
        it += n;
    } else {
        for (iter_difference_t<Iterator> i = 0; i < n; ++i) {
            ++it;
        }
    }
}

}  // namespace dtl
