// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: Point-to-point message preparation overhead
// Tests buffer preparation and serialization for send/recv operations.
// No actual MPI — measures local overhead only.

#include <benchmark/benchmark.h>

#include <dtl/serialization/serialization.hpp>

#include <cstring>
#include <vector>

namespace {

// ============================================================================
// Message buffer preparation (contiguous copy)
// ============================================================================

void BM_MessageBufferPrep(benchmark::State& state) {
    const auto msg_size = static_cast<size_t>(state.range(0));
    std::vector<std::byte> source(msg_size);
    std::vector<std::byte> send_buffer(msg_size);

    // Fill source with test data
    for (size_t i = 0; i < msg_size; ++i) {
        source[i] = static_cast<std::byte>(i & 0xFF);
    }

    for (auto _ : state) {
        std::memcpy(send_buffer.data(), source.data(), msg_size);
        benchmark::DoNotOptimize(send_buffer.data());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(msg_size));
}

BENCHMARK(BM_MessageBufferPrep)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 20)   // 64 bytes to 1 MiB
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Typed message serialization (array of doubles)
// ============================================================================

void BM_MessageSerializeDoubles(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<double> data(n, 3.14159);
    std::vector<std::byte> buffer(n * sizeof(double));

    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) {
            dtl::serializer<double>::serialize(
                data[i], buffer.data() + i * sizeof(double));
        }
        benchmark::DoNotOptimize(buffer.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(double)));
}

BENCHMARK(BM_MessageSerializeDoubles)
    ->RangeMultiplier(10)
    ->Range(100, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Typed message deserialization (array of doubles)
// ============================================================================

void BM_MessageDeserializeDoubles(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<std::byte> buffer(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        double val = static_cast<double>(i);
        std::memcpy(buffer.data() + i * sizeof(double), &val, sizeof(double));
    }
    std::vector<double> output(n);

    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) {
            output[i] = dtl::serializer<double>::deserialize(
                buffer.data() + i * sizeof(double), sizeof(double));
        }
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(double)));
}

BENCHMARK(BM_MessageDeserializeDoubles)
    ->RangeMultiplier(10)
    ->Range(100, 1000000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Bulk memcpy (baseline for send)
// ============================================================================

void BM_BulkMemcpy(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));
    std::vector<std::byte> src(size);
    std::vector<std::byte> dst(size);

    for (auto _ : state) {
        std::memcpy(dst.data(), src.data(), size);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
}

BENCHMARK(BM_BulkMemcpy)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 20)
    ->Unit(benchmark::kNanosecond);

}  // namespace
