// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_algorithm_infrastructure.cpp
/// @brief Unit tests for Phase 3 algorithm infrastructure
/// @details Tests for algorithm traits, concepts, and dispatch mechanism

#include <dtl/algorithms/algorithm_traits.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/execution/async.hpp>
#include <dtl/views/local_view.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace dtl::test {

// =============================================================================
// Algorithm Traits Tests (Task 3.1.1)
// =============================================================================

TEST(AlgorithmTraitsTest, DomainTagsAreDifferent) {
    static_assert(!std::same_as<local_algorithm_tag, collective_algorithm_tag>);
    static_assert(!std::same_as<local_algorithm_tag, distributed_algorithm_tag>);
    static_assert(!std::same_as<collective_algorithm_tag, distributed_algorithm_tag>);
}

TEST(AlgorithmTraitsTest, ForEachIsLocal) {
    using traits = algorithm_traits<for_each_tag>;
    static_assert(std::same_as<traits::domain_tag, local_algorithm_tag>);
    static_assert(!traits::requires_communication);
    static_assert(!traits::requires_all_ranks);
    static_assert(traits::is_modifying);
}

TEST(AlgorithmTraitsTest, ReduceIsCollective) {
    using traits = algorithm_traits<reduce_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::requires_all_ranks);
    static_assert(traits::produces_scalar);
}

TEST(AlgorithmTraitsTest, SortIsCollective) {
    using traits = algorithm_traits<sort_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::requires_all_ranks);
    static_assert(traits::is_modifying);
}

TEST(AlgorithmTraitsTest, TransformIsLocal) {
    using traits = algorithm_traits<transform_tag>;
    static_assert(std::same_as<traits::domain_tag, local_algorithm_tag>);
    static_assert(!traits::requires_communication);
}

TEST(AlgorithmTraitsTest, FillIsLocal) {
    using traits = algorithm_traits<fill_tag>;
    static_assert(std::same_as<traits::domain_tag, local_algorithm_tag>);
    static_assert(!traits::requires_communication);
    static_assert(traits::is_modifying);
}

TEST(AlgorithmTraitsTest, CountIsCollective) {
    using traits = algorithm_traits<count_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
    static_assert(traits::produces_scalar);
}

TEST(AlgorithmTraitsTest, FindIsCollective) {
    using traits = algorithm_traits<find_tag>;
    static_assert(std::same_as<traits::domain_tag, collective_algorithm_tag>);
    static_assert(traits::requires_communication);
}

TEST(AlgorithmTraitsTest, LocalAlgorithmConcept) {
    static_assert(LocalAlgorithm<for_each_tag>);
    static_assert(LocalAlgorithm<transform_tag>);
    static_assert(LocalAlgorithm<fill_tag>);
    static_assert(!LocalAlgorithm<reduce_tag>);
    static_assert(!LocalAlgorithm<sort_tag>);
}

TEST(AlgorithmTraitsTest, CollectiveAlgorithmConcept) {
    static_assert(CollectiveAlgorithm<reduce_tag>);
    static_assert(CollectiveAlgorithm<sort_tag>);
    static_assert(CollectiveAlgorithm<count_tag>);
    static_assert(!CollectiveAlgorithm<for_each_tag>);
    static_assert(!CollectiveAlgorithm<transform_tag>);
}

TEST(AlgorithmTraitsTest, PropertyQueryHelpers) {
    static_assert(requires_communication_v<reduce_tag>);
    static_assert(!requires_communication_v<for_each_tag>);
    static_assert(requires_all_ranks_v<sort_tag>);
    static_assert(!requires_all_ranks_v<transform_tag>);
    static_assert(is_modifying_algorithm_v<for_each_tag>);
    static_assert(!is_modifying_algorithm_v<count_tag>);
    static_assert(produces_scalar_v<reduce_tag>);
    static_assert(!produces_scalar_v<for_each_tag>);
}

// =============================================================================
// Execution Policy Detection Tests (Task 3.1.2)
// =============================================================================

TEST(ExecutionDispatchTest, SeqPolicyDetection) {
    static_assert(is_seq_policy_v<seq>);
    static_assert(!is_seq_policy_v<par>);
    static_assert(!is_seq_policy_v<async>);
    static_assert(!is_seq_policy_v<int>);
}

TEST(ExecutionDispatchTest, ParPolicyDetection) {
    static_assert(is_par_policy_v<par>);
    static_assert(is_par_policy_v<par_n<4>>);
    static_assert(!is_par_policy_v<seq>);
    static_assert(!is_par_policy_v<async>);
}

TEST(ExecutionDispatchTest, AsyncPolicyDetection) {
    static_assert(is_async_policy_v<async>);
    static_assert(!is_async_policy_v<seq>);
    static_assert(!is_async_policy_v<par>);
}

TEST(ExecutionDispatchTest, SeqDispatcherForEach) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int sum = 0;
    execution_dispatcher<seq>::for_each(data.begin(), data.end(),
                                         [&sum](int x) { sum += x; });
    EXPECT_EQ(sum, 15);
}

TEST(ExecutionDispatchTest, SeqDispatcherReduce) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = execution_dispatcher<seq>::reduce(data.begin(), data.end(),
                                                    0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(ExecutionDispatchTest, SeqDispatcherSort) {
    std::vector<int> data = {5, 2, 4, 1, 3};
    execution_dispatcher<seq>::sort(data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(ExecutionDispatchTest, SeqDispatcherSortWithComparator) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    execution_dispatcher<seq>::sort(data.begin(), data.end(), std::greater<>{});
    EXPECT_EQ(data, (std::vector<int>{5, 4, 3, 2, 1}));
}

TEST(ExecutionDispatchTest, SeqDispatcherFill) {
    std::vector<int> data(5);
    execution_dispatcher<seq>::fill(data.begin(), data.end(), 42);
    EXPECT_EQ(data, (std::vector<int>{42, 42, 42, 42, 42}));
}

TEST(ExecutionDispatchTest, SeqDispatcherCopy) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);
    execution_dispatcher<seq>::copy(src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, src);
}

TEST(ExecutionDispatchTest, SeqDispatcherCount) {
    std::vector<int> data = {1, 2, 2, 3, 2};
    auto count = execution_dispatcher<seq>::count(data.begin(), data.end(), 2);
    EXPECT_EQ(count, 3);
}

TEST(ExecutionDispatchTest, SeqDispatcherCountIf) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto count = execution_dispatcher<seq>::count_if(data.begin(), data.end(),
                                                      [](int x) { return x > 2; });
    EXPECT_EQ(count, 3);
}

TEST(ExecutionDispatchTest, SeqDispatcherFind) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto it = execution_dispatcher<seq>::find(data.begin(), data.end(), 3);
    EXPECT_NE(it, data.end());
    EXPECT_EQ(*it, 3);
}

TEST(ExecutionDispatchTest, SeqDispatcherFindIf) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto it = execution_dispatcher<seq>::find_if(data.begin(), data.end(),
                                                  [](int x) { return x > 3; });
    EXPECT_NE(it, data.end());
    EXPECT_EQ(*it, 4);
}

TEST(ExecutionDispatchTest, SeqDispatcherAllOf) {
    std::vector<int> data = {2, 4, 6, 8};
    bool result = execution_dispatcher<seq>::all_of(data.begin(), data.end(),
                                                     [](int x) { return x % 2 == 0; });
    EXPECT_TRUE(result);
}

TEST(ExecutionDispatchTest, SeqDispatcherAnyOf) {
    std::vector<int> data = {1, 3, 5, 6};
    bool result = execution_dispatcher<seq>::any_of(data.begin(), data.end(),
                                                     [](int x) { return x % 2 == 0; });
    EXPECT_TRUE(result);
}

TEST(ExecutionDispatchTest, SeqDispatcherNoneOf) {
    std::vector<int> data = {1, 3, 5, 7};
    bool result = execution_dispatcher<seq>::none_of(data.begin(), data.end(),
                                                      [](int x) { return x % 2 == 0; });
    EXPECT_TRUE(result);
}

TEST(ExecutionDispatchTest, ParDispatcherForEach) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::atomic<int> sum{0};
    execution_dispatcher<par>::for_each(data.begin(), data.end(),
                                         [&sum](int x) { sum += x; });
    EXPECT_EQ(sum.load(), 15);
}

TEST(ExecutionDispatchTest, ParDispatcherReduce) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = execution_dispatcher<par>::reduce(data.begin(), data.end(),
                                                    0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(ExecutionDispatchTest, ParDispatcherSort) {
    std::vector<int> data = {5, 2, 4, 1, 3};
    execution_dispatcher<par>::sort(data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

// =============================================================================
// Dispatch Helper Function Tests
// =============================================================================

TEST(DispatchHelpersTest, DispatchForEachSeq) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int sum = 0;
    dispatch_for_each(seq{}, data.begin(), data.end(), [&sum](int x) { sum += x; });
    EXPECT_EQ(sum, 15);
}

TEST(DispatchHelpersTest, DispatchForEachPar) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::atomic<int> sum{0};
    dispatch_for_each(par{}, data.begin(), data.end(), [&sum](int x) { sum += x; });
    EXPECT_EQ(sum.load(), 15);
}

TEST(DispatchHelpersTest, DispatchTransformSeq) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);
    dispatch_transform(seq{}, src.begin(), src.end(), dst.begin(),
                       [](int x) { return x * 2; });
    EXPECT_EQ(dst, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(DispatchHelpersTest, DispatchReduceSeq) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(DispatchHelpersTest, DispatchReducePar) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = dispatch_reduce(par{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(DispatchHelpersTest, DispatchSortSeq) {
    std::vector<int> data = {5, 2, 4, 1, 3};
    dispatch_sort(seq{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(DispatchHelpersTest, DispatchSortPar) {
    std::vector<int> data = {5, 2, 4, 1, 3};
    dispatch_sort(par{}, data.begin(), data.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(DispatchHelpersTest, DispatchFillSeq) {
    std::vector<int> data(5);
    dispatch_fill(seq{}, data.begin(), data.end(), 42);
    EXPECT_EQ(data, (std::vector<int>{42, 42, 42, 42, 42}));
}

TEST(DispatchHelpersTest, DispatchCopySeq) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);
    dispatch_copy(seq{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, src);
}

TEST(DispatchHelpersTest, DispatchCountSeq) {
    std::vector<int> data = {1, 2, 2, 3, 2};
    auto count = dispatch_count(seq{}, data.begin(), data.end(), 2);
    EXPECT_EQ(count, 3);
}

TEST(DispatchHelpersTest, DispatchFindSeq) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto it = dispatch_find(seq{}, data.begin(), data.end(), 3);
    EXPECT_NE(it, data.end());
    EXPECT_EQ(*it, 3);
}

// =============================================================================
// Algorithm Concepts Tests (Task 3.1.3)
// =============================================================================

TEST(AlgorithmConceptsTest, ExecutionPolicyTypeConcept) {
    static_assert(ExecutionPolicyType<seq>);
    static_assert(ExecutionPolicyType<par>);
    static_assert(ExecutionPolicyType<async>);
    static_assert(ExecutionPolicyType<par_n<4>>);
    static_assert(!ExecutionPolicyType<int>);
    static_assert(!ExecutionPolicyType<std::vector<int>>);
}

TEST(AlgorithmConceptsTest, LocallyIterableConcept) {
    static_assert(LocallyIterable<std::vector<int>>);
    static_assert(LocallyIterable<local_view<int>>);
    static_assert(!LocallyIterable<int>);
}

TEST(AlgorithmConceptsTest, ContiguousLocalRangeConcept) {
    static_assert(ContiguousLocalRange<std::vector<int>>);
    static_assert(ContiguousLocalRange<local_view<int>>);
    static_assert(ContiguousLocalRange<std::span<int>>);
}

TEST(AlgorithmConceptsTest, BinaryReductionOpConcept) {
    static_assert(BinaryReductionOp<std::plus<>, int>);
    static_assert(BinaryReductionOp<std::multiplies<>, int>);
    static_assert(BinaryReductionOp<std::plus<>, double>);

    auto max_op = [](int a, int b) { return std::max(a, b); };
    static_assert(BinaryReductionOp<decltype(max_op), int>);
}

TEST(AlgorithmConceptsTest, ElementPredicateConcept) {
    static_assert(ElementPredicate<std::function<bool(const int&)>, int>);

    auto is_even = [](const int& x) { return x % 2 == 0; };
    static_assert(ElementPredicate<decltype(is_even), int>);
}

TEST(AlgorithmConceptsTest, SortComparatorConcept) {
    static_assert(SortComparator<std::less<>, int>);
    static_assert(SortComparator<std::greater<>, int>);
}

TEST(AlgorithmConceptsTest, UnaryElementFunctionConcept) {
    auto double_it = [](int& x) { x *= 2; };
    static_assert(UnaryElementFunction<decltype(double_it), int>);
}

TEST(AlgorithmConceptsTest, UnaryTransformOpConcept) {
    auto square = [](const int& x) { return x * x; };
    static_assert(UnaryTransformOp<decltype(square), int>);
}

TEST(AlgorithmConceptsTest, ReductionResultConcept) {
    static_assert(ReductionResult<int>);
    static_assert(ReductionResult<double>);
    static_assert(ReductionResult<std::string>);
}

TEST(AlgorithmConceptsTest, IncrementableConcept) {
    static_assert(Incrementable<int>);
    static_assert(Incrementable<double>);
    static_assert(Incrementable<size_type>);
}

// =============================================================================
// Local View Integration Tests
// =============================================================================

TEST(LocalViewIntegrationTest, ForEachWithLocalView) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    int sum = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&sum](int x) { sum += x; });
    EXPECT_EQ(sum, 15);
}

TEST(LocalViewIntegrationTest, TransformWithLocalView) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    dispatch_transform(seq{}, view.begin(), view.end(), view.begin(),
                       [](int x) { return x * 2; });
    EXPECT_EQ(data, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(LocalViewIntegrationTest, ReduceWithLocalView) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    int result = dispatch_reduce(seq{}, view.begin(), view.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(LocalViewIntegrationTest, SortWithLocalView) {
    std::vector<int> data = {5, 2, 4, 1, 3};
    local_view<int> view(data.data(), data.size());

    dispatch_sort(seq{}, view.begin(), view.end());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(LocalViewIntegrationTest, FillWithLocalView) {
    std::vector<int> data(5);
    local_view<int> view(data.data(), data.size());

    dispatch_fill(seq{}, view.begin(), view.end(), 42);
    EXPECT_EQ(data, (std::vector<int>{42, 42, 42, 42, 42}));
}

// =============================================================================
// Parallel Execution Correctness Tests
// =============================================================================

TEST(ParallelExecutionTest, ReduceIsCorrect) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 1);

    int seq_result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    int par_result = dispatch_reduce(par{}, data.begin(), data.end(), 0, std::plus<>{});

    EXPECT_EQ(seq_result, par_result);
    EXPECT_EQ(seq_result, 500500);  // Sum of 1 to 1000
}

TEST(ParallelExecutionTest, SortIsCorrect) {
    std::vector<int> data1 = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
    std::vector<int> data2 = data1;

    dispatch_sort(seq{}, data1.begin(), data1.end());
    dispatch_sort(par{}, data2.begin(), data2.end());

    EXPECT_EQ(data1, data2);
    EXPECT_TRUE(std::is_sorted(data1.begin(), data1.end()));
}

TEST(ParallelExecutionTest, TransformIsCorrect) {
    std::vector<int> data1(100);
    std::vector<int> data2(100);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);

    dispatch_transform(seq{}, data1.begin(), data1.end(), data1.begin(),
                       [](int x) { return x * x; });
    dispatch_transform(par{}, data2.begin(), data2.end(), data2.begin(),
                       [](int x) { return x * x; });

    EXPECT_EQ(data1, data2);
}

}  // namespace dtl::test
