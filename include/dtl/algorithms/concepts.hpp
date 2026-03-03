// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file concepts.hpp
/// @brief C++20 concepts specific to DTL algorithms
/// @details Defines concepts for algorithm constraints and type requirements.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace dtl {

// =============================================================================
// Execution Policy Concepts
// =============================================================================

/// @brief Concept for execution policy types (alias for ExecutionPolicy)
/// @details This is the primary concept used in algorithm declarations.
///          It validates that a type is a valid DTL execution policy.
template <typename T>
concept ExecutionPolicyType =
    requires {
        typename std::decay_t<T>::policy_category;
        requires std::same_as<typename std::decay_t<T>::policy_category, execution_policy_tag>;
        { std::decay_t<T>::mode() } -> std::same_as<execution_mode>;
        { std::decay_t<T>::is_blocking() } -> std::same_as<bool>;
    };

// =============================================================================
// Range Concepts
// =============================================================================

/// @brief Concept for ranges that can be iterated locally (no communication)
/// @details A locally iterable range provides direct access to elements
///          without any inter-rank communication.
template <typename R>
concept LocallyIterable =
    std::ranges::range<R> &&
    requires(R& r) {
        { std::ranges::begin(r) } -> std::input_iterator;
        { std::ranges::end(r) } -> std::sentinel_for<decltype(std::ranges::begin(r))>;
    };

/// @brief Concept for ranges that provide segmented access
/// @details Segmented ranges can be iterated by segments (partitions).
template <typename R>
concept SegmentedIterable =
    requires(R& r) {
        { r.segmented_view() };
    };

/// @brief Concept for contiguous local ranges
/// @details Contiguous ranges allow pointer arithmetic and bulk operations.
template <typename R>
concept ContiguousLocalRange =
    std::ranges::contiguous_range<R> &&
    std::ranges::sized_range<R>;

// =============================================================================
// Operation Concepts
// =============================================================================

/// @brief Concept for binary operations suitable for reductions
/// @details A binary operation must be invocable with two values of type T
///          and produce a result convertible back to T.
template <typename Op, typename T>
concept BinaryReductionOp =
    std::invocable<Op, T, T> &&
    std::convertible_to<std::invoke_result_t<Op, T, T>, T>;

/// @brief Concept for unary functions applicable to elements
/// @details Unary functions take an element reference and may modify it.
template <typename F, typename T>
concept UnaryElementFunction =
    std::invocable<F, T&>;

/// @brief Concept for unary transform operations
/// @details Transform operations take an element and return a transformed value.
template <typename F, typename T>
concept UnaryTransformOp =
    std::invocable<F, const T&>;

/// @brief Concept for binary transform operations
template <typename F, typename T1, typename T2>
concept BinaryTransformOp =
    std::invocable<F, const T1&, const T2&>;

/// @brief Concept for predicates
/// @details Predicates take an element and return a boolean result.
template <typename P, typename T>
concept ElementPredicate =
    std::predicate<P, const T&>;

/// @brief Concept for comparators suitable for sorting
/// @details Comparators must define a strict weak ordering.
template <typename Cmp, typename T>
concept SortComparator =
    std::strict_weak_order<Cmp, T, T>;

// =============================================================================
// Iterator Concepts for Algorithms
// =============================================================================

/// @brief Concept for iterators suitable for for_each
template <typename It>
concept ForEachIterator =
    std::input_iterator<It>;

/// @brief Concept for iterators suitable for transform output
template <typename It>
concept TransformOutputIterator =
    std::output_iterator<It, typename std::iterator_traits<It>::value_type>;

/// @brief Concept for iterators suitable for sorting
template <typename It>
concept SortableIterator =
    std::random_access_iterator<It> &&
    std::sortable<It>;

/// @brief Concept for iterators suitable for reduction
template <typename It>
concept ReducibleIterator =
    std::input_iterator<It>;

// =============================================================================
// Container Algorithm Concepts
// =============================================================================

/// @brief Concept for containers that support local view access
template <typename C>
concept HasLocalView =
    requires(C& c) {
        { c.local_view() } -> LocallyIterable;
    };

/// @brief Concept for containers that support segmented view access
template <typename C>
concept HasSegmentedView =
    requires(C& c) {
        { c.segmented_view() };
    };

/// @brief Concept for containers suitable for in-place algorithms
template <typename C>
concept InPlaceModifiable =
    HasLocalView<C> &&
    requires(C& c) {
        { *c.local_view().begin() } -> std::same_as<typename C::value_type&>;
    };

// =============================================================================
// Reduction Result Concepts
// =============================================================================

/// @brief Concept for types that can be reduction results
/// @details Reduction results must be default-constructible and copyable.
template <typename T>
concept ReductionResult =
    std::default_initializable<T> &&
    std::copyable<T>;

/// @brief Concept for types with an identity element for addition
template <typename T>
concept HasAdditiveIdentity =
    std::default_initializable<T> &&
    requires(T a) {
        { a + T{} } -> std::convertible_to<T>;
    };

/// @brief Concept for types with an identity element for multiplication
template <typename T>
concept HasMultiplicativeIdentity =
    requires(T a) {
        { a * T{1} } -> std::convertible_to<T>;
    };

// =============================================================================
// Algorithm-Specific Type Constraints
// =============================================================================

/// @brief Concept for types that can be counted
template <typename T, typename Container>
concept Countable =
    std::equality_comparable_with<T, typename Container::value_type>;

/// @brief Concept for types that can be found
template <typename T, typename Container>
concept Findable =
    std::equality_comparable_with<T, typename Container::value_type>;

/// @brief Concept for types that can be sorted
template <typename T>
concept Sortable =
    std::totally_ordered<T> ||
    requires(T a, T b) {
        { a < b } -> std::convertible_to<bool>;
    };

// =============================================================================
// Generator Concepts
// =============================================================================

/// @brief Concept for generator functions (used in generate algorithm)
/// @details Generators produce values when invoked with no arguments.
template <typename G, typename T>
concept Generator =
    std::invocable<G> &&
    std::convertible_to<std::invoke_result_t<G>, T>;

/// @brief Concept for iota-style incrementable values
template <typename T>
concept Incrementable =
    requires(T& t) {
        { ++t } -> std::same_as<T&>;
        { t++ } -> std::convertible_to<T>;
    };

// =============================================================================
// Distributed Algorithm Concepts
// =============================================================================

/// @brief Concept for distributed containers with rank information
template <typename C>
concept RankAware =
    requires(const C& c) {
        { c.rank() } -> std::convertible_to<rank_t>;
        { c.num_ranks() } -> std::convertible_to<rank_t>;
    };

/// @brief Concept for containers that support barriers
template <typename C>
concept SupportsBarrier =
    requires(C& c) {
        { c.barrier() };
    };

/// @brief Concept for containers suitable for distributed algorithms
template <typename C>
concept DistributedAlgorithmTarget =
    DistributedContainer<C> &&
    HasLocalView<C> &&
    HasSegmentedView<C>;

// =============================================================================
// Combined Algorithm Constraints
// =============================================================================

/// @brief Combined concept for standard distributed algorithm arguments
/// @details Validates that an execution policy and container are both valid
///          for use in distributed algorithms. Simplifies requires clauses.
/// @tparam EP Execution policy type
/// @tparam C Container type
/// @since 0.1.0
template <typename EP, typename C>
concept DistributedAlgorithmArgs =
    ExecutionPolicyType<EP> && DistributedContainer<C>;

/// @brief Combined concept for collective algorithm arguments
/// @details Validates that an execution policy, container, and communicator
///          are all valid for collective operations.
/// @tparam EP Execution policy type
/// @tparam C Container type
/// @tparam Comm Communicator type
/// @since 0.1.0
template <typename EP, typename C, typename Comm>
concept CollectiveAlgorithmArgs =
    DistributedAlgorithmArgs<EP, C> && Communicator<Comm>;

}  // namespace dtl
