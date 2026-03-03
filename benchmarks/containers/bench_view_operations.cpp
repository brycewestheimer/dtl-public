// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: View creation and operations — subview, strided_view, local_view

#include <benchmark/benchmark.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/subview.hpp>
#include <dtl/views/strided_view.hpp>

#include <numeric>
#include <vector>

namespace {

struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

// ============================================================================
// Local view creation overhead
// ============================================================================

void BM_LocalViewCreation(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<int> vec(n, 1, ctx);

    for (auto _ : state) {
        auto local = vec.local_view();
        benchmark::DoNotOptimize(local.data());
        benchmark::DoNotOptimize(local.size());
    }
}

BENCHMARK(BM_LocalViewCreation)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Subview creation overhead
// ============================================================================

void BM_SubviewCreation(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    auto local = vec.local_view();

    for (auto _ : state) {
        auto sub = dtl::make_subview(local, n / 4, n / 2);
        benchmark::DoNotOptimize(sub.data());
        benchmark::DoNotOptimize(sub.size());
    }
}

BENCHMARK(BM_SubviewCreation)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Subview iteration
// ============================================================================

void BM_SubviewIteration(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    auto local = vec.local_view();
    auto sub = dtl::make_subview(local, 0, n / 2);

    for (auto _ : state) {
        double sum = 0.0;
        for (auto it = sub.begin(); it != sub.end(); ++it) {
            sum += *it;
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n / 2));
}

BENCHMARK(BM_SubviewIteration)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Strided view creation
// ============================================================================

void BM_StridedViewCreation(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    auto local = vec.local_view();

    for (auto _ : state) {
        auto strided = dtl::make_strided_view(local, 2);
        benchmark::DoNotOptimize(&strided);
    }
}

BENCHMARK(BM_StridedViewCreation)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Strided view iteration (stride=2, every other element)
// ============================================================================

void BM_StridedViewIteration(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    auto local = vec.local_view();
    auto strided = dtl::make_strided_view(local, 2);

    for (auto _ : state) {
        double sum = 0.0;
        for (dtl::size_type i = 0; i < strided.size(); ++i) {
            sum += strided[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(strided.size()));
}

BENCHMARK(BM_StridedViewIteration)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Strided view with large stride
// ============================================================================

void BM_StridedViewLargeStride(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    const auto stride = static_cast<std::ptrdiff_t>(state.range(1));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    auto local = vec.local_view();
    auto strided = dtl::make_strided_view(local, stride);

    for (auto _ : state) {
        double sum = 0.0;
        for (dtl::size_type i = 0; i < strided.size(); ++i) {
            sum += strided[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(strided.size()));
}

BENCHMARK(BM_StridedViewLargeStride)
    ->Args({1000000, 2})
    ->Args({1000000, 4})
    ->Args({1000000, 8})
    ->Args({1000000, 16})
    ->Args({1000000, 64})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Nested subview (subview of subview)
// ============================================================================

void BM_NestedSubview(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    auto local = vec.local_view();

    for (auto _ : state) {
        auto sub1 = dtl::make_subview(local, 0, n * 3 / 4);
        auto sub2 = sub1.subrange(n / 4, n / 4);
        double sum = 0.0;
        for (auto it = sub2.begin(); it != sub2.end(); ++it) {
            sum += *it;
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n / 4));
}

BENCHMARK(BM_NestedSubview)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
