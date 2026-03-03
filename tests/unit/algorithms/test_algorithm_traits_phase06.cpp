// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_algorithm_traits_phase06.cpp
/// @brief Tests for Phase 06 Task 07: algorithm_traits specializations
/// @details Verify all algorithm tags have correct traits specializations.

#include <dtl/algorithms/algorithm_traits.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// T07: Verify all 8 previously-missing algorithm_traits specializations
// =============================================================================

TEST(AlgorithmTraitsPhase06, StableSortTagHasTraits) {
    using traits = algorithm_traits<stable_sort_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::requires_all_ranks);
    static_assert(traits::is_modifying);
    static_assert(!traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, AllOfTagHasTraits) {
    using traits = algorithm_traits<all_of_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::requires_all_ranks);
    static_assert(!traits::is_modifying);
    static_assert(traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, AnyOfTagHasTraits) {
    using traits = algorithm_traits<any_of_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(!traits::is_modifying);
    static_assert(traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, NoneOfTagHasTraits) {
    using traits = algorithm_traits<none_of_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(!traits::is_modifying);
    static_assert(traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, GenerateTagHasTraits) {
    using traits = algorithm_traits<generate_tag>;
    static_assert(std::same_as<traits::domain_tag, local_algorithm_tag>);
    static_assert(!traits::requires_communication);
    static_assert(!traits::requires_all_ranks);
    static_assert(traits::is_modifying);
    static_assert(!traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, AccumulateTagHasTraits) {
    using traits = algorithm_traits<accumulate_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, SumTagHasTraits) {
    using traits = algorithm_traits<sum_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, ProductTagHasTraits) {
    using traits = algorithm_traits<product_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::produces_scalar);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, ReplaceTagHasTraits) {
    using traits = algorithm_traits<replace_tag>;
    static_assert(std::same_as<traits::domain_tag, local_algorithm_tag>);
    static_assert(!traits::requires_communication);
    static_assert(traits::is_modifying);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, PartialSortTagHasTraits) {
    using traits = algorithm_traits<partial_sort_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::is_modifying);
    SUCCEED();
}

TEST(AlgorithmTraitsPhase06, NthElementTagHasTraits) {
    using traits = algorithm_traits<nth_element_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::is_modifying);
    SUCCEED();
}

// =============================================================================
// Concept checks for new specializations
// =============================================================================

TEST(AlgorithmTraitsPhase06, ConceptChecks) {
    static_assert(CollectiveAlgorithm<stable_sort_tag>);
    static_assert(CollectiveAlgorithm<all_of_tag>);
    static_assert(CollectiveAlgorithm<any_of_tag>);
    static_assert(CollectiveAlgorithm<none_of_tag>);
    static_assert(LocalAlgorithm<generate_tag>);
    static_assert(CollectiveAlgorithm<accumulate_tag>);
    static_assert(CollectiveAlgorithm<sum_tag>);
    static_assert(CollectiveAlgorithm<product_tag>);
    static_assert(LocalAlgorithm<replace_tag>);
    static_assert(CollectiveAlgorithm<partial_sort_tag>);
    static_assert(CollectiveAlgorithm<nth_element_tag>);
    SUCCEED();
}

// =============================================================================
// Property query helpers for new specializations
// =============================================================================

TEST(AlgorithmTraitsPhase06, PropertyQueryHelpers) {
    static_assert(requires_communication_v<stable_sort_tag>);
    static_assert(requires_communication_v<all_of_tag>);
    static_assert(requires_communication_v<any_of_tag>);
    static_assert(requires_communication_v<none_of_tag>);
    static_assert(!requires_communication_v<generate_tag>);
    static_assert(requires_communication_v<accumulate_tag>);
    static_assert(requires_communication_v<sum_tag>);
    static_assert(requires_communication_v<product_tag>);
    static_assert(!requires_communication_v<replace_tag>);

    static_assert(produces_scalar_v<all_of_tag>);
    static_assert(produces_scalar_v<any_of_tag>);
    static_assert(produces_scalar_v<none_of_tag>);
    static_assert(produces_scalar_v<accumulate_tag>);
    static_assert(produces_scalar_v<sum_tag>);
    static_assert(produces_scalar_v<product_tag>);

    static_assert(is_modifying_algorithm_v<stable_sort_tag>);
    static_assert(is_modifying_algorithm_v<generate_tag>);
    static_assert(is_modifying_algorithm_v<replace_tag>);
    static_assert(is_modifying_algorithm_v<partial_sort_tag>);
    static_assert(is_modifying_algorithm_v<nth_element_tag>);
    SUCCEED();
}

}  // namespace dtl::test
