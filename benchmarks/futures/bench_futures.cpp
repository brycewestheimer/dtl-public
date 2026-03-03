// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: Futures — promise/future creation, resolution, distributed_future overhead

#include <benchmark/benchmark.h>

#include <dtl/futures/distributed_future.hpp>

#include <memory>

namespace {

// ============================================================================
// shared_state creation overhead
// ============================================================================

void BM_SharedStateCreation(benchmark::State& state) {
    for (auto _ : state) {
        auto ss = std::make_shared<dtl::futures::shared_state<int>>();
        benchmark::DoNotOptimize(ss.get());
    }
}

BENCHMARK(BM_SharedStateCreation)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// shared_state set_value + get cycle
// ============================================================================

void BM_SharedStateSetGet(benchmark::State& state) {
    for (auto _ : state) {
        auto ss = std::make_shared<dtl::futures::shared_state<int>>();
        ss->set_value(42);
        int val = ss->get();
        benchmark::DoNotOptimize(val);
    }
}

BENCHMARK(BM_SharedStateSetGet)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// distributed_future<int> construction + get
// ============================================================================

void BM_DistributedFutureCreateGet(benchmark::State& state) {
    for (auto _ : state) {
        auto ss = std::make_shared<dtl::futures::shared_state<int>>();
        ss->set_value(42);
        dtl::futures::distributed_future<int> fut(ss);
        int val = fut.get();
        benchmark::DoNotOptimize(val);
    }
}

BENCHMARK(BM_DistributedFutureCreateGet)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// distributed_future<void> set + wait
// ============================================================================

void BM_DistributedFutureVoidWait(benchmark::State& state) {
    for (auto _ : state) {
        auto ss = std::make_shared<dtl::futures::shared_state<void>>();
        ss->set_value();
        dtl::futures::distributed_future<void> fut(ss);
        fut.wait();
    }
}

BENCHMARK(BM_DistributedFutureVoidWait)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// is_ready check (already resolved)
// ============================================================================

void BM_FutureIsReadyResolved(benchmark::State& state) {
    auto ss = std::make_shared<dtl::futures::shared_state<int>>();
    ss->set_value(42);

    for (auto _ : state) {
        bool ready = ss->is_ready();
        benchmark::DoNotOptimize(ready);
    }
}

BENCHMARK(BM_FutureIsReadyResolved)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// wait_for on already-resolved future
// ============================================================================

void BM_FutureWaitForResolved(benchmark::State& state) {
    auto ss = std::make_shared<dtl::futures::shared_state<int>>();
    ss->set_value(42);
    dtl::futures::distributed_future<int> fut(ss);

    for (auto _ : state) {
        auto status = ss->wait_for(std::chrono::milliseconds(0));
        benchmark::DoNotOptimize(status);
    }
}

BENCHMARK(BM_FutureWaitForResolved)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Batch future creation (simulates many async ops)
// ============================================================================

void BM_BatchFutureCreation(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        std::vector<dtl::futures::distributed_future<int>> futures;
        futures.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto ss = std::make_shared<dtl::futures::shared_state<int>>();
            ss->set_value(static_cast<int>(i));
            futures.emplace_back(ss);
        }
        benchmark::DoNotOptimize(futures.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_BatchFutureCreation)
    ->RangeMultiplier(10)
    ->Range(10, 10000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
