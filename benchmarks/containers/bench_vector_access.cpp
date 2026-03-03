// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: distributed_vector element access and iteration patterns

#include <benchmark/benchmark.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/views/local_view.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace {

struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

// ============================================================================
// Construction overhead
// ============================================================================

template <typename T>
void BM_VectorConstruction(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;

    for (auto _ : state) {
        dtl::distributed_vector<T> vec(n, ctx);
        benchmark::DoNotOptimize(vec.local_view().data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_VectorConstruction<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_VectorConstruction<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Construction with initial value
// ============================================================================

template <typename T>
void BM_VectorConstructionWithValue(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;

    for (auto _ : state) {
        dtl::distributed_vector<T> vec(n, T{42}, ctx);
        benchmark::DoNotOptimize(vec.local_view().data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_VectorConstructionWithValue<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Sequential local_view iteration (sum via pointer)
// ============================================================================

template <typename T>
void BM_LocalViewIteration(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> vec(n, T{1}, ctx);

    for (auto _ : state) {
        auto local = vec.local_view();
        T sum = T{0};
        for (auto it = local.begin(); it != local.end(); ++it) {
            sum += *it;
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_LocalViewIteration<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_LocalViewIteration<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Index-based access via local_view::operator[]
// ============================================================================

template <typename T>
void BM_LocalViewIndexAccess(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> vec(n, T{1}, ctx);

    for (auto _ : state) {
        auto local = vec.local_view();
        T sum = T{0};
        for (dtl::size_type i = 0; i < local.size(); ++i) {
            sum += local[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_LocalViewIndexAccess<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_LocalViewIndexAccess<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Range-for iteration
// ============================================================================

template <typename T>
void BM_LocalViewRangeFor(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<T> vec(n, T{1}, ctx);

    for (auto _ : state) {
        auto local = vec.local_view();
        T sum = T{0};
        for (const auto& val : local) {
            sum += val;
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_LocalViewRangeFor<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Baseline: std::vector iteration
// ============================================================================

template <typename T>
void BM_StdVectorIteration(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<T> vec(n, T{1});

    for (auto _ : state) {
        T sum = T{0};
        for (const auto& val : vec) {
            sum += val;
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_StdVectorIteration<int>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_StdVectorIteration<double>)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
