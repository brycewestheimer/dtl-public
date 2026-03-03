// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: Halo exchange pattern setup/teardown overhead
// Tests the cost of creating subviews for halo regions, data packing, etc.
// No actual MPI communication — measures local bookkeeping.

#include <benchmark/benchmark.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/views/subview.hpp>
#include <dtl/views/local_view.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

// ============================================================================
// Halo subview creation
// ============================================================================

void BM_HaloSubviewCreation(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    const dtl::size_type halo_width = 10;
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);

    for (auto _ : state) {
        auto local = vec.local_view();
        // Create left and right halo subviews
        auto left_halo = dtl::make_subview(local, 0, halo_width);
        auto right_halo = dtl::make_subview(local, local.size() - halo_width, halo_width);
        // Create interior subview
        auto interior = dtl::make_subview(local, halo_width, local.size() - 2 * halo_width);
        benchmark::DoNotOptimize(left_halo.data());
        benchmark::DoNotOptimize(right_halo.data());
        benchmark::DoNotOptimize(interior.data());
    }
}

BENCHMARK(BM_HaloSubviewCreation)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Halo packing: pack halo region into contiguous send buffer
// ============================================================================

void BM_HaloPacking(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    const dtl::size_type halo_width = static_cast<dtl::size_type>(state.range(1));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    std::vector<double> send_buffer(halo_width);

    for (auto _ : state) {
        auto local = vec.local_view();
        auto halo = dtl::make_subview(local, local.size() - halo_width, halo_width);
        // Pack halo to contiguous buffer (simulates send preparation)
        std::copy(halo.begin(), halo.end(), send_buffer.begin());
        benchmark::DoNotOptimize(send_buffer.data());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(halo_width) *
                            static_cast<int64_t>(sizeof(double)));
}

BENCHMARK(BM_HaloPacking)
    ->Args({100000, 10})
    ->Args({100000, 100})
    ->Args({100000, 1000})
    ->Args({1000000, 10})
    ->Args({1000000, 100})
    ->Args({1000000, 1000})
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Halo unpacking: unpack received data into halo region
// ============================================================================

void BM_HaloUnpacking(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    const dtl::size_type halo_width = static_cast<dtl::size_type>(state.range(1));
    single_rank_ctx ctx;
    dtl::distributed_vector<double> vec(n, 1.0, ctx);
    std::vector<double> recv_buffer(halo_width, 2.0);

    for (auto _ : state) {
        auto local = vec.local_view();
        auto halo = dtl::make_subview(local, 0, halo_width);
        // Unpack received data into halo region
        std::copy(recv_buffer.begin(), recv_buffer.end(), halo.begin());
        benchmark::DoNotOptimize(local.data());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(halo_width) *
                            static_cast<int64_t>(sizeof(double)));
}

BENCHMARK(BM_HaloUnpacking)
    ->Args({100000, 10})
    ->Args({100000, 100})
    ->Args({100000, 1000})
    ->Args({1000000, 10})
    ->Args({1000000, 100})
    ->Args({1000000, 1000})
    ->Unit(benchmark::kNanosecond);

}  // namespace
