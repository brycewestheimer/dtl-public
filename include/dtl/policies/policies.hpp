// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file policies.hpp
/// @brief Master include for all DTL policies
/// @details Provides single-header access to all policy types and concepts.
/// @since 0.1.0

#pragma once

// Forward declarations with defaults (must come first)
#include <dtl/core/fwd.hpp>

// ============================================================================
// Partition Policies - How data is divided across ranks
// ============================================================================
#include <dtl/policies/partition/partition_policy.hpp>
#include <dtl/policies/partition/block_partition.hpp>
#include <dtl/policies/partition/cyclic_partition.hpp>
#include <dtl/policies/partition/hash_partition.hpp>
#include <dtl/policies/partition/replicated.hpp>
#include <dtl/policies/partition/custom_partition.hpp>
#include <dtl/policies/partition/dynamic_block.hpp>
#include <dtl/policies/partition/block_nd_partition.hpp>

// ============================================================================
// Placement Policies - Where data resides (host/device)
// ============================================================================
#include <dtl/policies/placement/placement_policy.hpp>
#include <dtl/policies/placement/host_only.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/policies/placement/device_preferred.hpp>
#include <dtl/policies/placement/unified_memory.hpp>
#include <dtl/policies/placement/explicit_placement.hpp>

// ============================================================================
// Consistency Policies - Synchronization and memory ordering
// ============================================================================
#include <dtl/policies/consistency/consistency_policy.hpp>
#include <dtl/policies/consistency/bulk_synchronous.hpp>
#include <dtl/policies/consistency/sequential_consistent.hpp>
#include <dtl/policies/consistency/release_acquire.hpp>
#include <dtl/policies/consistency/relaxed.hpp>

// ============================================================================
// Execution Policies - How operations are executed
// ============================================================================
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/execution/async.hpp>
#include <dtl/policies/execution/on_stream.hpp>

// ============================================================================
// Error Policies - How errors are reported and handled
// ============================================================================
#include <dtl/policies/error/error_policy.hpp>
#include <dtl/policies/error/expected.hpp>
#include <dtl/policies/error/throwing.hpp>
#include <dtl/policies/error/terminating.hpp>
#include <dtl/policies/error/callback.hpp>

namespace dtl {

// ============================================================================
// Default Policy Types
// ============================================================================

/// @brief Default partition policy (block distribution)
using default_partition = block_partition<>;

/// @brief Default placement policy (host memory)
using default_placement = host_only;

/// @brief Default consistency policy (BSP model)
using default_consistency = bulk_synchronous;

/// @brief Default execution policy (sequential)
using default_execution = seq;

/// @brief Default error policy (expected/result-based)
using default_error = expected_policy;

// ============================================================================
// Policy Composition Helpers
// ============================================================================

/// @brief Check if a type is any kind of policy
template <typename T>
inline constexpr bool is_policy_v =
    is_partition_policy_v<T> ||
    is_placement_policy_v<T> ||
    is_consistency_policy_v<T> ||
    is_execution_policy_v<T> ||
    is_error_policy_v<T>;

/// @brief Concept for any DTL policy
template <typename T>
concept Policy = is_policy_v<T>;

// ============================================================================
// Policy Extraction from Variadic Parameter Packs
// ============================================================================

namespace detail {

/// @brief Helper to find first matching policy in pack, or use default
template <template <typename> class Predicate, typename Default, typename... Ts>
struct find_policy_or_default;

/// @brief Base case: no matching policy found, use default
template <template <typename> class Predicate, typename Default>
struct find_policy_or_default<Predicate, Default> {
    using type = Default;
};

/// @brief Recursive case: check head, continue if not matching
template <template <typename> class Predicate, typename Default, typename Head, typename... Tail>
struct find_policy_or_default<Predicate, Default, Head, Tail...> {
    using type = std::conditional_t<
        Predicate<Head>::value,
        Head,
        typename find_policy_or_default<Predicate, Default, Tail...>::type
    >;
};

/// @brief Predicate wrapper for partition policies
template <typename T>
struct is_partition_pred : is_partition_policy<T> {};

/// @brief Predicate wrapper for placement policies
template <typename T>
struct is_placement_pred : is_placement_policy<T> {};

/// @brief Predicate wrapper for consistency policies
template <typename T>
struct is_consistency_pred : is_consistency_policy<T> {};

/// @brief Predicate wrapper for execution policies
template <typename T>
struct is_execution_pred : is_execution_policy<T> {};

/// @brief Predicate wrapper for error policies
template <typename T>
struct is_error_pred : is_error_policy<T> {};

}  // namespace detail

/// @brief Extract partition policy from parameter pack, or use default
/// @tparam Default The default policy if none found
/// @tparam Policies The policies to search through
template <typename Default, typename... Policies>
using extract_partition_policy_t =
    typename detail::find_policy_or_default<detail::is_partition_pred, Default, Policies...>::type;

/// @brief Extract placement policy from parameter pack, or use default
/// @tparam Default The default policy if none found
/// @tparam Policies The policies to search through
template <typename Default, typename... Policies>
using extract_placement_policy_t =
    typename detail::find_policy_or_default<detail::is_placement_pred, Default, Policies...>::type;

/// @brief Extract consistency policy from parameter pack, or use default
/// @tparam Default The default policy if none found
/// @tparam Policies The policies to search through
template <typename Default, typename... Policies>
using extract_consistency_policy_t =
    typename detail::find_policy_or_default<detail::is_consistency_pred, Default, Policies...>::type;

/// @brief Extract execution policy from parameter pack, or use default
/// @tparam Default The default policy if none found
/// @tparam Policies The policies to search through
template <typename Default, typename... Policies>
using extract_execution_policy_t =
    typename detail::find_policy_or_default<detail::is_execution_pred, Default, Policies...>::type;

/// @brief Extract error policy from parameter pack, or use default
/// @tparam Default The default policy if none found
/// @tparam Policies The policies to search through
template <typename Default, typename... Policies>
using extract_error_policy_t =
    typename detail::find_policy_or_default<detail::is_error_pred, Default, Policies...>::type;

// ============================================================================
// Policy Set with Variadic Extraction
// ============================================================================

/// @brief Policy set with explicit types for all policy dimensions
/// @tparam Partition The partition policy
/// @tparam Placement The placement policy
/// @tparam Consistency The consistency policy
/// @tparam Execution The execution policy
/// @tparam Error The error policy
template <
    typename Partition = default_partition,
    typename Placement = default_placement,
    typename Consistency = default_consistency,
    typename Execution = default_execution,
    typename Error = default_error
>
struct policy_set {
    using partition_policy = Partition;
    using placement_policy = Placement;
    using consistency_policy = Consistency;
    using execution_policy = Execution;
    using error_policy = Error;
};

/// @brief Default policy set with all defaults
using default_policies = policy_set<>;

/// @brief Policy set that extracts policies from a variadic parameter pack
/// @tparam Policies Variadic list of policies in any order
/// @details Policies can be specified in any order. Missing policies use defaults.
///          Duplicate policy types will use the first one found.
///
/// @par Example:
/// @code
/// // These are equivalent:
/// using ps1 = make_policy_set<cyclic_partition<>, device_only<0>>;
/// using ps2 = make_policy_set<device_only<0>, cyclic_partition<>>;
///
/// // Both have:
/// // - partition_policy = cyclic_partition<>
/// // - placement_policy = device_only<0>
/// // - consistency_policy = bulk_synchronous (default)
/// // - execution_policy = seq (default)
/// // - error_policy = expected_policy (default)
/// @endcode
template <typename... Policies>
using make_policy_set = policy_set<
    extract_partition_policy_t<default_partition, Policies...>,
    extract_placement_policy_t<default_placement, Policies...>,
    extract_consistency_policy_t<default_consistency, Policies...>,
    extract_execution_policy_t<default_execution, Policies...>,
    extract_error_policy_t<default_error, Policies...>
>;

// ============================================================================
// Policy Merging Utilities
// ============================================================================

/// @brief Merge two policy sets, preferring policies from Override
/// @tparam Base The base policy set
/// @tparam Override The override policy set (takes precedence)
template <typename Base, typename Override>
struct merge_policies {
    // Use Override's policy if it differs from default, otherwise use Base's
    using partition_policy = std::conditional_t<
        std::is_same_v<typename Override::partition_policy, default_partition>,
        typename Base::partition_policy,
        typename Override::partition_policy
    >;
    using placement_policy = std::conditional_t<
        std::is_same_v<typename Override::placement_policy, default_placement>,
        typename Base::placement_policy,
        typename Override::placement_policy
    >;
    using consistency_policy = std::conditional_t<
        std::is_same_v<typename Override::consistency_policy, default_consistency>,
        typename Base::consistency_policy,
        typename Override::consistency_policy
    >;
    using execution_policy = std::conditional_t<
        std::is_same_v<typename Override::execution_policy, default_execution>,
        typename Base::execution_policy,
        typename Override::execution_policy
    >;
    using error_policy = std::conditional_t<
        std::is_same_v<typename Override::error_policy, default_error>,
        typename Base::error_policy,
        typename Override::error_policy
    >;

    using type = policy_set<
        partition_policy,
        placement_policy,
        consistency_policy,
        execution_policy,
        error_policy
    >;
};

/// @brief Helper alias for merge_policies
template <typename Base, typename Override>
using merge_policies_t = typename merge_policies<Base, Override>::type;

// ============================================================================
// Policy Validation
// ============================================================================

/// @brief Check that all types in a pack are valid policies
template <typename... Ts>
inline constexpr bool all_policies_v = (is_policy_v<Ts> && ...);

/// @brief Concept requiring all types to be policies
template <typename... Ts>
concept AllPolicies = all_policies_v<Ts...>;

/// @brief Count how many of each policy type appears in a pack
template <typename... Policies>
struct policy_count {
    static constexpr size_type partition = (is_partition_policy_v<Policies> + ... + 0);
    static constexpr size_type placement = (is_placement_policy_v<Policies> + ... + 0);
    static constexpr size_type consistency = (is_consistency_policy_v<Policies> + ... + 0);
    static constexpr size_type execution = (is_execution_policy_v<Policies> + ... + 0);
    static constexpr size_type error = (is_error_policy_v<Policies> + ... + 0);

    /// @brief Check that at most one of each policy type is specified
    static constexpr bool no_duplicates =
        partition <= 1 && placement <= 1 && consistency <= 1 &&
        execution <= 1 && error <= 1;
};

}  // namespace dtl
