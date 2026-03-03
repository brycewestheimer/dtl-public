// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: dtl::sort — local sort performance
// Tests sort with various sizes and presort states (random, sorted, reversed).

#include <benchmark/benchmark.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/sorting/sort.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace {

struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

// ============================================================================
// Sort random data (sequential)
// ============================================================================

template <typename T>
void BM_SortRandomSeq(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    std::mt19937 gen(42);

    for (auto _ : state) {
        state.PauseTiming();
        dtl::distributed_vector<T> vec(n, ctx);
        auto local = vec.local_view();
        for (dtl::size_type i = 0; i < local.size(); ++i) {
            local[i] = static_cast<T>(gen() % (n * 10));
        }
        state.ResumeTiming();

        auto r = dtl::sort(dtl::seq{}, vec);
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_SortRandomSeq<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SortRandomSeq<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Sort already-sorted data (best case)
// ============================================================================

void BM_SortPresorted(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;

    for (auto _ : state) {
        state.PauseTiming();
        dtl::distributed_vector<int> vec(n, ctx);
        auto local = vec.local_view();
        std::iota(local.begin(), local.end(), 0);
        state.ResumeTiming();

        auto r = dtl::sort(dtl::seq{}, vec);
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_SortPresorted)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Sort reverse-sorted data (worst case for some algorithms)
// ============================================================================

void BM_SortReversed(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;

    for (auto _ : state) {
        state.PauseTiming();
        dtl::distributed_vector<int> vec(n, ctx);
        auto local = vec.local_view();
        std::iota(local.begin(), local.end(), 0);
        std::reverse(local.begin(), local.end());
        state.ResumeTiming();

        auto r = dtl::sort(dtl::seq{}, vec);
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_SortReversed)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Baseline: std::sort for comparison
// ============================================================================

void BM_StdSort(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::mt19937 gen(42);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int> vec(n);
        for (auto& v : vec) v = static_cast<int>(gen() % (n * 10));
        state.ResumeTiming();

        std::sort(vec.begin(), vec.end());
        benchmark::DoNotOptimize(vec.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_StdSort)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
