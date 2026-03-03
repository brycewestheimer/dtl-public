// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file bench_cuda_operations.cpp
/// @brief Benchmarks for CUDA memory operations and executor overhead
/// @details Measures GPU memory allocation, host<->device transfers,
///          and DTL executor launch overhead.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA

#include <cuda_runtime.h>

#include <backends/cuda/cuda_executor.hpp>
#include <backends/cuda/cuda_memory_space.hpp>

#include <benchmark/benchmark.h>

#include <cstring>
#include <vector>

namespace {

// ============================================================================
// Helper: ensure CUDA is available
// ============================================================================

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ============================================================================
// CUDA Memory Allocation — cudaMalloc
// ============================================================================

static void BM_CudaMalloc(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithError("No CUDA devices");
        return;
    }
    const size_t bytes = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        benchmark::DoNotOptimize(ptr);
        if (err == cudaSuccess) {
            cudaFree(ptr);
        }
    }

    state.counters["bytes"] = static_cast<double>(bytes);
}

BENCHMARK(BM_CudaMalloc)
    ->Arg(256)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1 << 20)   // 1 MB
    ->Arg(1 << 24)   // 16 MB
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// ============================================================================
// CUDA Memory Allocation — cudaMallocManaged (Unified)
// ============================================================================

static void BM_CudaMallocManaged(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithError("No CUDA devices");
        return;
    }
    const size_t bytes = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, bytes);
        benchmark::DoNotOptimize(ptr);
        if (err == cudaSuccess) {
            cudaFree(ptr);
        }
    }

    state.counters["bytes"] = static_cast<double>(bytes);
}

BENCHMARK(BM_CudaMallocManaged)
    ->Arg(256)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1 << 20)
    ->Arg(1 << 24)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// ============================================================================
// Host -> Device Transfer Rate
// ============================================================================

static void BM_HostToDevice(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithError("No CUDA devices");
        return;
    }
    const size_t bytes = static_cast<size_t>(state.range(0));

    // Pre-allocate both sides
    std::vector<char> h_buf(bytes, 0x42);
    void* d_buf = nullptr;
    cudaMalloc(&d_buf, bytes);

    for (auto _ : state) {
        cudaMemcpy(d_buf, h_buf.data(), bytes, cudaMemcpyHostToDevice);
    }

    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes));
    cudaFree(d_buf);
}

BENCHMARK(BM_HostToDevice)
    ->Arg(256)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1 << 20)
    ->Arg(1 << 24)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// ============================================================================
// Device -> Host Transfer Rate
// ============================================================================

static void BM_DeviceToHost(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithError("No CUDA devices");
        return;
    }
    const size_t bytes = static_cast<size_t>(state.range(0));

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, bytes);
    cudaMemset(d_buf, 0x55, bytes);

    std::vector<char> h_buf(bytes, 0);

    for (auto _ : state) {
        cudaMemcpy(h_buf.data(), d_buf, bytes, cudaMemcpyDeviceToHost);
    }

    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes));
    cudaFree(d_buf);
}

BENCHMARK(BM_DeviceToHost)
    ->Arg(256)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1 << 20)
    ->Arg(1 << 24)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// ============================================================================
// Async Transfer via cudaMemcpyAsync + Stream Sync
// ============================================================================

static void BM_HostToDeviceAsync(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithError("No CUDA devices");
        return;
    }
    const size_t bytes = static_cast<size_t>(state.range(0));

    // Pin the host buffer for async transfers
    void* h_buf = nullptr;
    cudaMallocHost(&h_buf, bytes);
    std::memset(h_buf, 0x42, bytes);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes));

    cudaStreamDestroy(stream);
    cudaFree(d_buf);
    cudaFreeHost(h_buf);
}

BENCHMARK(BM_HostToDeviceAsync)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1 << 20)
    ->Arg(1 << 24)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// ============================================================================
// GPU Kernel Launch Overhead via DTL Executor
// ============================================================================

static void BM_CudaDispatchOverhead(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithError("No CUDA devices");
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        // Dispatch a trivial no-op kernel (just records an event)
        auto future = dtl::cuda::dispatch_gpu_async(stream, [](cudaStream_t) {
            // Empty kernel — measuring dispatch + event overhead
        });
        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
}

BENCHMARK(BM_CudaDispatchOverhead)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Iterations(10000);

// ============================================================================
// DTL CUDA Memory Space Allocate/Deallocate
// ============================================================================

static void BM_DtlCudaMemorySpaceAlloc(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithError("No CUDA devices");
        return;
    }
    const size_t bytes = static_cast<size_t>(state.range(0));

    dtl::cuda::cuda_memory_space mem_space;

    for (auto _ : state) {
        void* ptr = mem_space.allocate(bytes);
        benchmark::DoNotOptimize(ptr);
        if (ptr) {
            mem_space.deallocate(ptr, bytes);
        }
    }

    state.counters["bytes"] = static_cast<double>(bytes);
}

BENCHMARK(BM_DtlCudaMemorySpaceAlloc)
    ->Arg(4096)
    ->Arg(65536)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

}  // anonymous namespace

BENCHMARK_MAIN();

#else  // !DTL_ENABLE_CUDA

#include <benchmark/benchmark.h>

static void BM_CudaNotEnabled(benchmark::State& state) {
    for (auto _ : state) {
        // no-op
    }
    state.SkipWithError("CUDA not enabled — skipping CUDA benchmarks");
}

BENCHMARK(BM_CudaNotEnabled);

BENCHMARK_MAIN();

#endif  // DTL_ENABLE_CUDA
