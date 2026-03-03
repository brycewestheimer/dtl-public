// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: RMA — memory_window creation and local put/get operations

#include <benchmark/benchmark.h>

#include <dtl/communication/memory_window.hpp>

#include <cstring>
#include <vector>

namespace {

// ============================================================================
// memory_window creation from existing memory
// ============================================================================

void BM_MemoryWindowCreate(benchmark::State& state) {
    const auto size = static_cast<dtl::size_type>(state.range(0));
    std::vector<std::byte> data(size);

    for (auto _ : state) {
        auto result = dtl::memory_window::create(data.data(), size);
        benchmark::DoNotOptimize(&result);
    }
}

BENCHMARK(BM_MemoryWindowCreate)
    ->RangeMultiplier(4)
    ->Range(256, 1 << 20)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// memory_window allocate (owns memory)
// ============================================================================

void BM_MemoryWindowAllocate(benchmark::State& state) {
    const auto size = static_cast<dtl::size_type>(state.range(0));

    for (auto _ : state) {
        auto result = dtl::memory_window::allocate(size);
        benchmark::DoNotOptimize(&result);
    }
}

BENCHMARK(BM_MemoryWindowAllocate)
    ->RangeMultiplier(4)
    ->Range(256, 1 << 20)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Local put via null_window_impl (single-rank self-copy)
// ============================================================================

void BM_MemoryWindowLocalPut(benchmark::State& state) {
    const auto size = static_cast<dtl::size_type>(state.range(0));
    std::vector<std::byte> data(size, std::byte{0});
    auto win_result = dtl::memory_window::create(data.data(), size);
    auto& window = win_result.value();

    std::vector<std::byte> source(size, std::byte{0xAA});

    for (auto _ : state) {
        auto r = window.put(source.data(), size, 0, 0);
        benchmark::DoNotOptimize(r);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
}

BENCHMARK(BM_MemoryWindowLocalPut)
    ->RangeMultiplier(4)
    ->Range(256, 1 << 20)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Local get via null_window_impl (single-rank self-copy)
// ============================================================================

void BM_MemoryWindowLocalGet(benchmark::State& state) {
    const auto size = static_cast<dtl::size_type>(state.range(0));
    std::vector<std::byte> data(size, std::byte{0xBB});
    auto win_result = dtl::memory_window::create(data.data(), size);
    auto& window = win_result.value();

    std::vector<std::byte> dest(size);

    for (auto _ : state) {
        auto r = window.get(dest.data(), size, 0, 0);
        benchmark::DoNotOptimize(r);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
}

BENCHMARK(BM_MemoryWindowLocalGet)
    ->RangeMultiplier(4)
    ->Range(256, 1 << 20)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Fence overhead (null_window_impl — near zero)
// ============================================================================

void BM_MemoryWindowFence(benchmark::State& state) {
    std::vector<std::byte> data(1024, std::byte{0});
    auto win_result = dtl::memory_window::create(data.data(), 1024);
    auto& window = win_result.value();

    for (auto _ : state) {
        auto r = window.fence();
        benchmark::DoNotOptimize(r);
    }
}

BENCHMARK(BM_MemoryWindowFence)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// from_span factory
// ============================================================================

void BM_MemoryWindowFromSpan(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<int> data(n, 42);

    for (auto _ : state) {
        auto result = dtl::memory_window::from_span(std::span{data});
        benchmark::DoNotOptimize(&result);
    }
}

BENCHMARK(BM_MemoryWindowFromSpan)
    ->RangeMultiplier(10)
    ->Range(100, 1000000)
    ->Unit(benchmark::kNanosecond);

}  // namespace
