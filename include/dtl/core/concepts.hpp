// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file concepts.hpp
/// @brief C++20 concepts for DTL type constraints
/// @details Defines core concepts used throughout DTL for compile-time
///          type checking and documentation of requirements.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>

#include <concepts>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace dtl {

// =============================================================================
// Core Type Concepts
// =============================================================================

/// @brief Concept for types that can be transported across ranks
/// @details A Transportable type can be serialized and deserialized
///          for inter-rank communication.
template <typename T>
concept Transportable = is_transportable_v<T>;

/// @brief Concept for trivially serializable types (memcpy-safe)
/// @details These types can be directly copied byte-by-byte without
///          any serialization overhead.
template <typename T>
concept TriviallySerializable = is_trivially_serializable_v<T>;

/// @brief Concept for types with value semantics
/// @details Types that are copyable and have well-defined equality.
template <typename T>
concept ValueType =
    std::copyable<T> &&
    std::equality_comparable<T>;

/// @brief Concept for numeric types usable in reductions
/// @details Types that support arithmetic operations.
template <typename T>
concept Numeric =
    std::integral<T> || std::floating_point<T>;

/// @brief Concept for types that can be used as reduction operands
/// @details Must be default constructible and have an identity element.
template <typename T, typename Op>
concept Reducible =
    std::default_initializable<T> &&
    std::invocable<Op, T, T> &&
    std::convertible_to<std::invoke_result_t<Op, T, T>, T>;

// =============================================================================
// Container Concepts
// =============================================================================

/// @brief Concept for distributed container types
/// @details Distributed containers have local/global views and partition info.
template <typename C>
concept DistributedContainer =
    requires(C& c, const C& cc) {
        typename C::value_type;
        typename C::size_type;
        { cc.size() } -> std::convertible_to<typename C::size_type>;
        { cc.local_size() } -> std::convertible_to<typename C::size_type>;
        { c.local_view() };
        { c.global_view() };
        { c.segmented_view() };
    };

/// @brief Concept for distributed tensor containers
/// @details Tensors have multi-dimensional extents and indexing.
template <typename C>
concept DistributedTensor =
    DistributedContainer<C> &&
    requires(const C& c) {
        typename C::extents_type;
        { c.global_extents() } -> std::same_as<typename C::extents_type>;
        { c.local_extents() } -> std::same_as<typename C::extents_type>;
        { c.extent(size_type{}) } -> std::convertible_to<size_type>;
    };

/// @brief Concept for distributed associative containers
/// @details Associative containers (like distributed_map) have fundamentally
///          different semantics than sequence containers:
///          - Key-based (not index-based) access
///          - Non-contiguous memory layout
///          - global_view/segmented_view don't make semantic sense
///
///          This concept captures the minimal requirements for distributed
///          associative containers without requiring sequence container views.
/// @since 0.1.0
template <typename M>
concept DistributedAssociativeContainer =
    requires(M& m, const M& cm) {
        typename M::key_type;
        typename M::mapped_type;
        typename M::size_type;
        { cm.local_size() } -> std::convertible_to<typename M::size_type>;
        { cm.is_local(std::declval<typename M::key_type>()) } -> std::same_as<bool>;
        { cm.owner(std::declval<typename M::key_type>()) } -> std::convertible_to<rank_t>;
        { m.begin() };
        { m.end() };
    };

/// @brief Unified concept for any distributed collection
/// @details This is the most general concept that both DistributedContainer
///          (sequence-based: vector, array, tensor, span) and
///          DistributedAssociativeContainer (key-based: map) satisfy.
///          Use this when writing generic code that operates on either type
///          of distributed collection.
///
/// @par Requirements:
/// - Has a value_type and size_type
/// - Has a local_size() method
/// - Is iterable (begin/end)
///
/// @since 0.1.0
template <typename C>
concept DistributedCollection =
    requires(C& c, const C& cc) {
        typename C::size_type;
        { cc.local_size() } -> std::convertible_to<typename C::size_type>;
        { c.begin() };
        { c.end() };
    };

/// @brief Concept for distributed map containers
/// @details Maps are associative containers with key-value semantics.
///
/// @note distributed_map does NOT satisfy DistributedContainer because:
///       - Key-based (not index-based) access pattern
///       - Non-contiguous memory layout
///       - global_view()/segmented_view() don't have meaningful semantics
///       Use DistributedAssociativeContainer for maps instead.
/// @since 0.1.0 (refined to not inherit from DistributedContainer)
template <typename M>
concept DistributedMap =
    DistributedAssociativeContainer<M>;

// =============================================================================
// View Concepts
// =============================================================================

/// @brief Concept for local view types (STL-compatible)
/// @details Local views provide iterator access to local partition data.
template <typename V>
concept LocalView =
    std::ranges::range<V> &&
    std::ranges::sized_range<V> &&
    requires(V& v) {
        { v.data() } -> std::contiguous_iterator;
    };

/// @brief Concept for global view types
/// @details Global views may return remote_ref for non-local elements.
template <typename V>
concept GlobalView =
    requires(V& v, index_t idx) {
        typename V::value_type;
        { v[idx] };  // May return remote_ref<T>
        { v.size() } -> std::convertible_to<size_type>;
    };

/// @brief Concept for segmented view types
/// @details Segmented views iterate over segments (one per rank).
template <typename V>
concept SegmentedView =
    std::ranges::range<V> &&
    requires(V& v) {
        { *v.begin() } -> LocalView;  // Each segment is a LocalView
    };

// =============================================================================
// Policy Concepts
// =============================================================================

/// @brief Concept for partition policy types
/// @details Partition policies define how data is distributed across ranks.
template <typename P>
concept PartitionPolicy =
    is_partition_policy_v<P> &&
    requires(const P& p, index_t idx, size_type size, rank_t nranks, rank_t rank) {
        { p.owner(idx, size, nranks) } -> std::convertible_to<rank_t>;
        { p.local_size(size, nranks, rank) } -> std::convertible_to<size_type>;
    };

/// @brief Concept for placement policy types
/// @details Placement policies define where data resides (host/device).
template <typename P>
concept PlacementPolicy =
    is_placement_policy_v<P>;

/// @brief Concept for consistency policy types
/// @details Consistency policies define synchronization guarantees.
template <typename P>
concept ConsistencyPolicy =
    is_consistency_policy_v<P>;

/// @brief Concept for execution policy types
/// @details Execution policies define how algorithms execute.
template <typename P>
concept ExecutionPolicy =
    is_execution_policy_v<P>;

/// @brief Concept for error policy types
/// @details Error policies define how errors are reported.
template <typename P>
concept ErrorPolicy =
    is_error_policy_v<P>;

// =============================================================================
// Iterator Concepts
// =============================================================================

/// @brief Concept for local iterators
/// @details Local iterators are random-access and contiguous.
template <typename I>
concept LocalIterator =
    std::random_access_iterator<I> &&
    std::contiguous_iterator<I>;

/// @brief Concept for global iterators
/// @details Global iterators may access remote elements.
template <typename I>
concept GlobalIterator =
    std::forward_iterator<I>;

/// @brief Concept for device iterators
/// @details Device iterators access GPU memory.
template <typename I>
concept DeviceIterator =
    requires(I i) {
        { *i };  // Dereferenceable (on device)
        { ++i } -> std::same_as<I&>;
    };

// =============================================================================
// Algorithm Callable Concepts
// =============================================================================

/// @brief Concept for unary functions usable in for_each
/// @tparam F The function type
/// @tparam T The element type
template <typename F, typename T>
concept UnaryFunction =
    std::invocable<F, T&>;

/// @brief Concept for unary predicates usable in find_if, count_if
/// @tparam P The predicate type
/// @tparam T The element type
template <typename P, typename T>
concept UnaryPredicate =
    std::predicate<P, const T&>;

/// @brief Concept for binary operations usable in reduce
/// @tparam Op The operation type
/// @tparam T The element type
template <typename Op, typename T>
concept BinaryOperation =
    std::invocable<Op, T, T> &&
    std::convertible_to<std::invoke_result_t<Op, T, T>, T>;

/// @brief Concept for transform operations
/// @tparam F The function type
/// @tparam T The input type
template <typename F, typename T>
concept TransformFunction =
    std::invocable<F, const T&> &&
    requires(F f, const T& t) {
        { f(t) };  // Returns transformed value
    };

/// @brief Concept for comparison functions usable in sort
/// @tparam Cmp The comparator type
/// @tparam T The element type
template <typename Cmp, typename T>
concept Comparator =
    std::strict_weak_order<Cmp, T, T>;

// =============================================================================
// Extent Concepts
// =============================================================================

/// @brief Concept for extent types (like std::extents)
/// @details Extents define multi-dimensional shape.
template <typename E>
concept ExtentsType =
    requires(const E& e, size_type n) {
        { E::rank() } -> std::convertible_to<size_type>;
        { E::rank_dynamic() } -> std::convertible_to<size_type>;
        { E::static_extent(n) } -> std::convertible_to<size_type>;
        { e.extent(n) } -> std::convertible_to<size_type>;
    };

// =============================================================================
// Memory Concepts
// =============================================================================

/// @brief Concept for memory space types
/// @details Memory spaces handle allocation in specific memory domains.
template <typename M>
concept MemorySpaceConcept =
    requires(M& m, size_type bytes, size_type alignment) {
        { m.allocate(bytes, alignment) } -> std::same_as<void*>;
        { m.deallocate(std::declval<void*>(), bytes) };
    };

/// @brief Concept for allocator types
/// @tparam A The allocator type
/// @tparam T The element type
template <typename A, typename T>
concept AllocatorFor =
    requires(A& a, size_type n) {
        { a.allocate(n) } -> std::same_as<T*>;
        { a.deallocate(std::declval<T*>(), n) };
    };

}  // namespace dtl
