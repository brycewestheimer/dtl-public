// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: Communication overhead — serialization and data packing
// Tests serialization cost for various types without actual MPI calls.

#include <benchmark/benchmark.h>

#include <dtl/serialization/serialization.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace {

// ============================================================================
// Trivial type serialization (int)
// ============================================================================

void BM_SerializeTrivialInt(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<int> data(n, 42);
    std::vector<std::byte> buffer(n * sizeof(int));

    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) {
            dtl::serializer<int>::serialize(data[i], buffer.data() + i * sizeof(int));
        }
        benchmark::DoNotOptimize(buffer.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(int)));
}

BENCHMARK(BM_SerializeTrivialInt)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Trivial type serialization (double)
// ============================================================================

void BM_SerializeTrivialDouble(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<double> data(n, 3.14);
    std::vector<std::byte> buffer(n * sizeof(double));

    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) {
            dtl::serializer<double>::serialize(data[i], buffer.data() + i * sizeof(double));
        }
        benchmark::DoNotOptimize(buffer.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(double)));
}

BENCHMARK(BM_SerializeTrivialDouble)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Deserialization (int)
// ============================================================================

void BM_DeserializeTrivialInt(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<std::byte> buffer(n * sizeof(int));
    // Fill buffer with serialized ints
    for (size_t i = 0; i < n; ++i) {
        int val = static_cast<int>(i);
        std::memcpy(buffer.data() + i * sizeof(int), &val, sizeof(int));
    }

    for (auto _ : state) {
        int total = 0;
        for (size_t i = 0; i < n; ++i) {
            int val = dtl::serializer<int>::deserialize(
                buffer.data() + i * sizeof(int), sizeof(int));
            total += val;
        }
        benchmark::DoNotOptimize(total);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(int)));
}

BENCHMARK(BM_DeserializeTrivialInt)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Data packing: pack scattered data into contiguous buffer
// ============================================================================

void BM_DataPacking(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    // Simulate scattered data in a larger array
    std::vector<double> source(n * 2, 1.0);
    std::vector<double> packed(n);

    for (auto _ : state) {
        // Pack every other element (simulating gather for collectives)
        for (size_t i = 0; i < n; ++i) {
            packed[i] = source[i * 2];
        }
        benchmark::DoNotOptimize(packed.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(double)));
}

BENCHMARK(BM_DataPacking)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Serialized size computation
// ============================================================================

void BM_SerializedSizeComputation(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<int> data(n, 42);

    for (auto _ : state) {
        dtl::size_type total = 0;
        for (size_t i = 0; i < n; ++i) {
            total += dtl::serializer<int>::serialized_size(data[i]);
        }
        benchmark::DoNotOptimize(total);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}

BENCHMARK(BM_SerializedSizeComputation)
    ->RangeMultiplier(10)
    ->Range(1000, 1000000)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
