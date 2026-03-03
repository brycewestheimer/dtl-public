// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file policy_mixin.hpp
/// @brief Reusable policy type extraction mixin for distributed containers
/// @details Extracts the 5 policy type aliases from a variadic Policies... pack
///          using make_policy_set<>. All distributed containers share this pattern;
///          this mixin eliminates the repeated boilerplate.
/// @since 0.1.0
/// @see policies.hpp for make_policy_set and policy extraction

#pragma once

#include <dtl/policies/policies.hpp>

namespace dtl {
namespace detail {

/// @brief Mixin providing policy type aliases extracted from a variadic pack
/// @tparam Policies... Policy pack (partition, placement, consistency, execution, error)
/// @details Containers can inherit or alias from this to get all 5 policy type aliases
///          without repeating the extraction boilerplate.
///
/// @par Example Usage:
/// @code
/// template <typename T, typename... Policies>
/// class my_container {
///     using policy_types = detail::container_policy_types<Policies...>;
///     using partition_policy = typename policy_types::partition_policy;
///     using placement_policy = typename policy_types::placement_policy;
///     // ... etc
/// };
/// @endcode
template <typename... Policies>
struct container_policy_types {
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
};

}  // namespace detail
}  // namespace dtl
