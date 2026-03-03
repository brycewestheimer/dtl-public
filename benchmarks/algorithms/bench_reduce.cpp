// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: dtl::reduce — local reduction performance
// Tests reduction at various sizes (1K–1M) and types (int, double).

#include <benchmark/benchmark.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>

#include <functional>
#include <numeric>
#include <vector>

namespace {

// Minimal single-rank context for constructing distributed containers.
struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

// ============================================================================
// Reduce with sequential policy
// ============================================================================

template <typename T>
void BM_ReduceSeq(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> vec(n, T{1}, ctx);

    for (auto _ : state) {
        T result = dtl::reduce(dtl::seq{}, vec, T{0}, std::plus<>{});
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_ReduceSeq<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_ReduceSeq<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Reduce with parallel policy
// ============================================================================

template <typename T>
void BM_ReducePar(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> vec(n, T{1}, ctx);

    for (auto _ : state) {
        T result = dtl::reduce(dtl::par{}, vec, T{0}, std::plus<>{});
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_ReducePar<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_ReducePar<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Baseline: std::accumulate for comparison
// ============================================================================

template <typename T>
void BM_StdAccumulate(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<T> vec(n, T{1});

    for (auto _ : state) {
        T result = std::accumulate(vec.begin(), vec.end(), T{0});
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_StdAccumulate<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_StdAccumulate<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
