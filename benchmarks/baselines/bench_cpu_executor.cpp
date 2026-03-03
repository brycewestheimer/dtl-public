// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: cpu_executor — thread pool task submission and parallel_for overhead

#include <benchmark/benchmark.h>

#include <cpu/cpu_executor.hpp>

#include <atomic>
#include <numeric>
#include <vector>

namespace {

// ============================================================================
// Single task submit + wait
// ============================================================================

void BM_CpuExecutorSingleTask(benchmark::State& state) {
    dtl::cpu::cpu_executor executor(4);

    for (auto _ : state) {
        auto fut = executor.async_execute([]() { return 42; });
        int val = fut.get();
        benchmark::DoNotOptimize(val);
    }
}

BENCHMARK(BM_CpuExecutorSingleTask)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Batch task submission
// ============================================================================

void BM_CpuExecutorBatchSubmit(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    dtl::cpu::cpu_executor executor(4);

    for (auto _ : state) {
        std::vector<std::future<int>> futures;
        futures.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            futures.push_back(executor.async_execute([i]() {
                return static_cast<int>(i * i);
            }));
        }
        int total = 0;
        for (auto& f : futures) {
            total += f.get();
        }
        benchmark::DoNotOptimize(total);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_CpuExecutorBatchSubmit)
    ->RangeMultiplier(10)
    ->Range(10, 10000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// parallel_for with trivial work
// ============================================================================

void BM_CpuExecutorParallelForTrivial(benchmark::State& state) {
    const auto n = static_cast<dtl::index_t>(state.range(0));
    dtl::cpu::cpu_executor executor(4);
    std::vector<int> results(static_cast<size_t>(n), 0);

    for (auto _ : state) {
        executor.parallel_for(0, n, [&results](dtl::index_t i) {
            results[static_cast<size_t>(i)] = static_cast<int>(i);
        });
        benchmark::DoNotOptimize(results.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_CpuExecutorParallelForTrivial)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// parallel_for with moderate work
// ============================================================================

void BM_CpuExecutorParallelForModerate(benchmark::State& state) {
    const auto n = static_cast<dtl::index_t>(state.range(0));
    dtl::cpu::cpu_executor executor(4);
    std::vector<double> results(static_cast<size_t>(n), 0.0);

    for (auto _ : state) {
        executor.parallel_for(0, n, [&results](dtl::index_t i) {
            double x = static_cast<double>(i);
            results[static_cast<size_t>(i)] = x * x + x * 0.5 + 1.0;
        });
        benchmark::DoNotOptimize(results.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_CpuExecutorParallelForModerate)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// parallel_reduce
// ============================================================================

void BM_CpuExecutorParallelReduce(benchmark::State& state) {
    const auto n = static_cast<dtl::index_t>(state.range(0));
    dtl::cpu::cpu_executor executor(4);
    std::vector<int> data(static_cast<size_t>(n));
    std::iota(data.begin(), data.end(), 0);

    for (auto _ : state) {
        int result = executor.parallel_reduce<int>(
            0, n, 0,
            [&data](dtl::index_t begin, dtl::index_t end) -> int {
                int sum = 0;
                for (auto i = begin; i < end; ++i) {
                    sum += data[static_cast<size_t>(i)];
                }
                return sum;
            },
            std::plus<int>{}
        );
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_CpuExecutorParallelReduce)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Thread pool construction/destruction overhead
// ============================================================================

void BM_ThreadPoolConstruction(benchmark::State& state) {
    const auto n_threads = static_cast<dtl::size_type>(state.range(0));

    for (auto _ : state) {
        dtl::cpu::thread_pool pool(n_threads);
        benchmark::DoNotOptimize(&pool);
    }
}

BENCHMARK(BM_ThreadPoolConstruction)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Synchronize (wait for all pending tasks) overhead
// ============================================================================

void BM_CpuExecutorSynchronize(benchmark::State& state) {
    dtl::cpu::cpu_executor executor(4);

    for (auto _ : state) {
        // Submit a few tasks then synchronize
        for (int i = 0; i < 10; ++i) {
            executor.execute([i]() {
                benchmark::DoNotOptimize(i * i);
            });
        }
        executor.synchronize();
    }
}

BENCHMARK(BM_CpuExecutorSynchronize)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
