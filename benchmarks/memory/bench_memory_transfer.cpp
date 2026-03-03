// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: Host-to-host copy operations (dtl::copy_to_host, memcpy)
// Without CUDA, dtl's copy operations fall back to memcpy, so these
// benchmarks measure the host-side copy throughput.

#include <benchmark/benchmark.h>

#include <dtl/memory/copy.hpp>
#include <dtl/memory/host_memory_space.hpp>

#include <cstring>
#include <vector>

namespace {

// ============================================================================
// dtl::copy_to_host (host-to-host fallback path)
// ============================================================================

template <typename T>
void BM_CopyToHost(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<T> source(n, T{42});

    for (auto _ : state) {
        auto result = dtl::copy_to_host(source.data(), n);
        benchmark::DoNotOptimize(result.data());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_CopyToHost<int>)
    ->RangeMultiplier(4)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_CopyToHost<double>)
    ->RangeMultiplier(4)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Contiguous memcpy benchmark
// ============================================================================

void BM_Memcpy(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));
    std::vector<std::byte> src(size, std::byte{0xAA});
    std::vector<std::byte> dst(size);

    for (auto _ : state) {
        std::memcpy(dst.data(), src.data(), size);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
}

BENCHMARK(BM_Memcpy)
    ->RangeMultiplier(4)
    ->Range(256, 1 << 24)  // 256 B to 16 MiB
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Typed element-by-element copy
// ============================================================================

template <typename T>
void BM_ElementwiseCopy(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<T> src(n, T{1});
    std::vector<T> dst(n);

    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = src[i];
        }
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_ElementwiseCopy<int>)
    ->RangeMultiplier(4)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_ElementwiseCopy<double>)
    ->RangeMultiplier(4)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// std::copy (compiler-optimized)
// ============================================================================

template <typename T>
void BM_StdCopy(benchmark::State& state) {
    const auto n = static_cast<size_t>(state.range(0));
    std::vector<T> src(n, T{1});
    std::vector<T> dst(n);

    for (auto _ : state) {
        std::copy(src.begin(), src.end(), dst.begin());
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_StdCopy<int>)
    ->RangeMultiplier(4)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_StdCopy<double>)
    ->RangeMultiplier(4)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
