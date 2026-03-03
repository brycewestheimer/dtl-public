// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: dtl::transform — unary and binary transform throughput

#include <benchmark/benchmark.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/modifying/transform.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>

#include <functional>
#include <vector>

namespace {

struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

// ============================================================================
// Unary transform (in-place, sequential)
// ============================================================================

template <typename T>
void BM_TransformUnaryInPlaceSeq(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> vec(n, T{1}, ctx);

    for (auto _ : state) {
        auto r = dtl::transform(dtl::seq{}, vec, [](T x) { return x * 2 + 1; });
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)) * 2);  // read + write
}

BENCHMARK(BM_TransformUnaryInPlaceSeq<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_TransformUnaryInPlaceSeq<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Unary transform (out-of-place, sequential)
// ============================================================================

template <typename T>
void BM_TransformUnaryOutOfPlaceSeq(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> src(n, T{1}, ctx);
    dtl::distributed_vector<T> dst(n, ctx);

    for (auto _ : state) {
        auto r = dtl::transform(dtl::seq{}, src, dst,
                                [](T x) { return x * 2 + 1; });
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)) * 2);
}

BENCHMARK(BM_TransformUnaryOutOfPlaceSeq<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_TransformUnaryOutOfPlaceSeq<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Unary transform (in-place, parallel policy)
// ============================================================================

template <typename T>
void BM_TransformUnaryInPlacePar(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> vec(n, T{1}, ctx);

    for (auto _ : state) {
        auto r = dtl::transform(dtl::par{}, vec, [](T x) { return x * 2 + 1; });
        benchmark::DoNotOptimize(r);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_TransformUnaryInPlacePar<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_TransformUnaryInPlacePar<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Baseline: std::transform for comparison
// ============================================================================

template <typename T>
void BM_StdTransform(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<T> src(n, T{1});
    std::vector<T> dst(n);

    for (auto _ : state) {
        std::transform(src.begin(), src.end(), dst.begin(),
                       [](T x) { return x * 2 + 1; });
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)) * 2);
}

BENCHMARK(BM_StdTransform<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_StdTransform<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
