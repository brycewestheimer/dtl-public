// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file algorithm_traits.hpp
/// @brief Algorithm traits and domain classification for DTL algorithms
/// @details Provides compile-time classification of algorithm domains and properties.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <type_traits>

namespace dtl {

// =============================================================================
// Domain Classification Tags
// =============================================================================

/// @brief Tag for local algorithms (no communication)
/// @details Local algorithms operate only on the local partition and
///          MUST NOT communicate with other ranks.
struct local_algorithm_tag {};

/// @brief Tag for collective algorithms (all ranks must participate)
/// @details Collective algorithms require all ranks to call the operation
///          and typically involve synchronization or global communication.
struct collective_algorithm_tag {};

/// @brief Tag for distributed algorithms (flexible communication)
/// @details Distributed algorithms may use point-to-point or collective
///          communication with flexible participation requirements.
struct distributed_algorithm_tag {};

// =============================================================================
// Algorithm Traits Primary Template
// =============================================================================

/// @brief Primary algorithm traits template
/// @tparam Algorithm The algorithm type or tag
/// @details Specialize this template for each algorithm to define its properties.
template <typename Algorithm>
struct algorithm_traits {
    /// @brief The domain classification tag (default: local)
    using domain_tag = local_algorithm_tag;

    /// @brief Whether the algorithm requires communication
    static constexpr bool requires_communication = false;

    /// @brief Whether all ranks must participate
    static constexpr bool requires_all_ranks = false;

    /// @brief Whether the algorithm modifies its input
    static constexpr bool is_modifying = false;

    /// @brief Whether the algorithm produces a scalar result
    static constexpr bool produces_scalar = false;
};

// =============================================================================
// Algorithm Domain Concepts
// =============================================================================

/// @brief Concept for local algorithms
/// @details Local algorithms operate only on local data and never communicate.
template <typename Alg>
concept LocalAlgorithm =
    std::same_as<typename algorithm_traits<Alg>::domain_tag, local_algorithm_tag>;

/// @brief Concept for collective algorithms
/// @details Collective algorithms require all ranks to participate.
template <typename Alg>
concept CollectiveAlgorithm =
    std::same_as<typename algorithm_traits<Alg>::domain_tag, collective_algorithm_tag>;

/// @brief Concept for distributed algorithms
/// @details Distributed algorithms may have flexible communication patterns.
template <typename Alg>
concept DistributedAlgorithm =
    std::same_as<typename algorithm_traits<Alg>::domain_tag, distributed_algorithm_tag>;

// =============================================================================
// Algorithm Property Query Helpers
// =============================================================================

/// @brief Check if algorithm requires communication
template <typename Alg>
inline constexpr bool requires_communication_v = algorithm_traits<Alg>::requires_communication;

/// @brief Check if algorithm requires all ranks
template <typename Alg>
inline constexpr bool requires_all_ranks_v = algorithm_traits<Alg>::requires_all_ranks;

/// @brief Check if algorithm is modifying
template <typename Alg>
inline constexpr bool is_modifying_algorithm_v = algorithm_traits<Alg>::is_modifying;

/// @brief Check if algorithm produces scalar result
template <typename Alg>
inline constexpr bool produces_scalar_v = algorithm_traits<Alg>::produces_scalar;

// =============================================================================
// Algorithm Tags for Trait Specialization
// =============================================================================

// Non-modifying algorithms
struct for_each_tag {};
struct find_tag {};
struct count_tag {};
struct all_of_tag {};
struct any_of_tag {};
struct none_of_tag {};

// Modifying algorithms
struct transform_tag {};
struct fill_tag {};
struct copy_tag {};
struct replace_tag {};
struct generate_tag {};

// Reduction algorithms
struct reduce_tag {};
struct transform_reduce_tag {};
struct accumulate_tag {};
struct sum_tag {};
struct product_tag {};
struct min_element_tag {};
struct max_element_tag {};

// Sorting algorithms
struct sort_tag {};
struct stable_sort_tag {};
struct partial_sort_tag {};
struct nth_element_tag {};
struct unique_tag {};

// =============================================================================
// Algorithm Traits Specializations
// =============================================================================

// for_each: local by default, modifying
template <>
struct algorithm_traits<for_each_tag> {
    using domain_tag = local_algorithm_tag;
    static constexpr bool requires_communication = false;
    static constexpr bool requires_all_ranks = false;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// find: collective (global result), non-modifying
template <>
struct algorithm_traits<find_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = false;
};

// count: collective (global count), non-modifying
template <>
struct algorithm_traits<count_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// transform: local by default, modifying
template <>
struct algorithm_traits<transform_tag> {
    using domain_tag = local_algorithm_tag;
    static constexpr bool requires_communication = false;
    static constexpr bool requires_all_ranks = false;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// fill: local by default, modifying
template <>
struct algorithm_traits<fill_tag> {
    using domain_tag = local_algorithm_tag;
    static constexpr bool requires_communication = false;
    static constexpr bool requires_all_ranks = false;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// copy: local or distributed depending on variant, modifying
template <>
struct algorithm_traits<copy_tag> {
    using domain_tag = local_algorithm_tag;
    static constexpr bool requires_communication = false;
    static constexpr bool requires_all_ranks = false;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// reduce: collective, produces scalar
template <>
struct algorithm_traits<reduce_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// transform_reduce: collective, produces scalar
template <>
struct algorithm_traits<transform_reduce_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// sort: collective for global sort, modifying
template <>
struct algorithm_traits<sort_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// min_element/max_element: collective, produces scalar-like result
template <>
struct algorithm_traits<min_element_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

template <>
struct algorithm_traits<max_element_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// unique: collective for global unique, modifying
template <>
struct algorithm_traits<unique_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// stable_sort: collective, modifying
template <>
struct algorithm_traits<stable_sort_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// all_of: collective, non-modifying, produces scalar
template <>
struct algorithm_traits<all_of_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// any_of: collective, non-modifying, produces scalar
template <>
struct algorithm_traits<any_of_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// none_of: collective, non-modifying, produces scalar
template <>
struct algorithm_traits<none_of_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// generate: local, modifying
template <>
struct algorithm_traits<generate_tag> {
    using domain_tag = local_algorithm_tag;
    static constexpr bool requires_communication = false;
    static constexpr bool requires_all_ranks = false;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// accumulate: collective, produces scalar
template <>
struct algorithm_traits<accumulate_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// sum: collective, produces scalar (alias for reduce with plus)
template <>
struct algorithm_traits<sum_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// product: collective, produces scalar (alias for reduce with multiplies)
template <>
struct algorithm_traits<product_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = false;
    static constexpr bool produces_scalar = true;
};

// replace: local by default, modifying
template <>
struct algorithm_traits<replace_tag> {
    using domain_tag = local_algorithm_tag;
    static constexpr bool requires_communication = false;
    static constexpr bool requires_all_ranks = false;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// partial_sort: collective, modifying
template <>
struct algorithm_traits<partial_sort_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

// nth_element: collective, modifying
template <>
struct algorithm_traits<nth_element_tag> {
    using domain_tag = collective_algorithm_tag;
    static constexpr bool requires_communication = true;
    static constexpr bool requires_all_ranks = true;
    static constexpr bool is_modifying = true;
    static constexpr bool produces_scalar = false;
};

}  // namespace dtl
