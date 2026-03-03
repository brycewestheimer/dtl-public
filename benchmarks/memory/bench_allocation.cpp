// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark: host_memory_space allocate/deallocate cycles

#include <benchmark/benchmark.h>

#include <dtl/memory/host_memory_space.hpp>

#include <cstdlib>
#include <vector>

namespace {

// ============================================================================
// Basic allocate/deallocate cycle
// ============================================================================

void BM_HostAllocDealloc(benchmark::State& state) {
    const auto size = static_cast<dtl::size_type>(state.range(0));

    for (auto _ : state) {
        void* ptr = dtl::host_memory_space::allocate(size);
        benchmark::DoNotOptimize(ptr);
        dtl::host_memory_space::deallocate(ptr, size);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
}

BENCHMARK(BM_HostAllocDealloc)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 24)  // 64 B to 16 MiB
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Aligned allocate/deallocate cycle
// ============================================================================

void BM_HostAlignedAllocDealloc(benchmark::State& state) {
    const auto size = static_cast<dtl::size_type>(state.range(0));
    const dtl::size_type alignment = 64;  // cache-line aligned

    for (auto _ : state) {
        void* ptr = dtl::host_memory_space::allocate(size, alignment);
        benchmark::DoNotOptimize(ptr);
        dtl::host_memory_space::deallocate(ptr, size, alignment);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
}

BENCHMARK(BM_HostAlignedAllocDealloc)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 24)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Typed allocate/deallocate cycle
// ============================================================================

template <typename T>
void BM_HostTypedAllocDealloc(benchmark::State& state) {
    const auto count = static_cast<dtl::size_type>(state.range(0));

    for (auto _ : state) {
        T* ptr = dtl::host_memory_space::allocate_typed<T>(count);
        benchmark::DoNotOptimize(ptr);
        dtl::host_memory_space::deallocate_typed(ptr, count);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count) *
                            static_cast<int64_t>(sizeof(T)));
}

BENCHMARK(BM_HostTypedAllocDealloc<int>)
    ->RangeMultiplier(10)
    ->Range(100, 1000000)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_HostTypedAllocDealloc<double>)
    ->RangeMultiplier(10)
    ->Range(100, 1000000)
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Rapid allocate/deallocate churn (small allocations)
// ============================================================================

void BM_HostAllocChurn(benchmark::State& state) {
    const auto count = static_cast<size_t>(state.range(0));
    const dtl::size_type alloc_size = 256;  // small allocation
    std::vector<void*> ptrs(count);

    for (auto _ : state) {
        // Allocate a batch
        for (size_t i = 0; i < count; ++i) {
            ptrs[i] = dtl::host_memory_space::allocate(alloc_size);
        }
        // Deallocate in reverse
        for (size_t i = count; i > 0; --i) {
            dtl::host_memory_space::deallocate(ptrs[i - 1], alloc_size);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count) * 2);  // alloc + dealloc
}

BENCHMARK(BM_HostAllocChurn)
    ->RangeMultiplier(10)
    ->Range(10, 10000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Baseline: malloc/free
// ============================================================================

void BM_MallocFree(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        void* ptr = std::malloc(size);
        benchmark::DoNotOptimize(ptr);
        std::free(ptr);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(size));
}

BENCHMARK(BM_MallocFree)
    ->RangeMultiplier(4)
    ->Range(64, 1 << 24)
    ->Unit(benchmark::kNanosecond);

}  // namespace
