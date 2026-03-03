// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: dtl::inclusive_scan / dtl::exclusive_scan — prefix scan performance

#include <benchmark/benchmark.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/reductions/scan.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>

#include <functional>
#include <numeric>
#include <vector>

namespace {

struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

// ============================================================================
// Inclusive scan (sequential)
// ============================================================================

template <typename T>
void BM_InclusiveScanSeq(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> input(n, T{1}, ctx);
    dtl::distributed_vector<T> output(n, ctx);

    for (auto _ : state) {
        auto r = dtl::inclusive_scan(dtl::seq{}, input, output,
                                     T{0}, std::plus<>{});
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)) * 2);
}

BENCHMARK(BM_InclusiveScanSeq<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_InclusiveScanSeq<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Exclusive scan (sequential)
// ============================================================================

template <typename T>
void BM_ExclusiveScanSeq(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> input(n, T{1}, ctx);
    dtl::distributed_vector<T> output(n, ctx);

    for (auto _ : state) {
        auto r = dtl::exclusive_scan(dtl::seq{}, input, output,
                                      T{0}, std::plus<>{});
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)) * 2);
}

BENCHMARK(BM_ExclusiveScanSeq<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_ExclusiveScanSeq<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Baseline: std::inclusive_scan
// ============================================================================

template <typename T>
void BM_StdInclusiveScan(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<T> input(n, T{1});
    std::vector<T> output(n);

    for (auto _ : state) {
        std::inclusive_scan(input.begin(), input.end(), output.begin(),
                           std::plus<>{}, T{0});
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)) * 2);
}

BENCHMARK(BM_StdInclusiveScan<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_StdInclusiveScan<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
