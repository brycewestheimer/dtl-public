// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file bench_nccl_collectives.cpp
/// @brief NCCL collective operation benchmarks using raw NCCL API
/// @details Benchmarks allreduce, broadcast, reduce, and allgather
///          latencies for GPU-resident buffers.
/// @note Uses raw NCCL API directly since DTL's nccl_communicator wrapper
///       has pre-existing API incompatibilities.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <vector>

namespace {

struct NcclState {
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;
    int rank = 0;
    int size = 1;
    bool valid = false;

    void init() {
        if (valid) return;

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
            return;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        cudaSetDevice(rank % device_count);
        cudaStreamCreate(&stream);

        ncclUniqueId id;
        if (rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (ncclCommInitRank(&comm, size, id, rank) == ncclSuccess)
            valid = true;
    }

    ~NcclState() {
        if (valid) ncclCommDestroy(comm);
        if (stream) cudaStreamDestroy(stream);
    }
};

static NcclState g_nccl;

void BM_NcclAllReduce(benchmark::State& state) {
    g_nccl.init();
    if (!g_nccl.valid) { state.SkipWithError("NCCL not available"); return; }

    const size_t bytes = static_cast<size_t>(state.range(0));
    const size_t count = bytes / sizeof(float);

    float *d_send = nullptr, *d_recv = nullptr;
    cudaMalloc(&d_send, bytes);
    cudaMalloc(&d_recv, bytes);
    cudaMemset(d_send, 1, bytes);

    for (auto _ : state) {
        ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum,
                       g_nccl.comm, g_nccl.stream);
        cudaStreamSynchronize(g_nccl.stream);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(bytes));
    cudaFree(d_send);
    cudaFree(d_recv);
}

void BM_NcclBroadcast(benchmark::State& state) {
    g_nccl.init();
    if (!g_nccl.valid) { state.SkipWithError("NCCL not available"); return; }

    const size_t bytes = static_cast<size_t>(state.range(0));
    const size_t count = bytes / sizeof(float);

    float* d_buf = nullptr;
    cudaMalloc(&d_buf, bytes);
    cudaMemset(d_buf, 1, bytes);

    for (auto _ : state) {
        ncclBroadcast(d_buf, d_buf, count, ncclFloat, 0,
                       g_nccl.comm, g_nccl.stream);
        cudaStreamSynchronize(g_nccl.stream);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(bytes));
    cudaFree(d_buf);
}

void BM_NcclReduce(benchmark::State& state) {
    g_nccl.init();
    if (!g_nccl.valid) { state.SkipWithError("NCCL not available"); return; }

    const size_t bytes = static_cast<size_t>(state.range(0));
    const size_t count = bytes / sizeof(float);

    float *d_send = nullptr, *d_recv = nullptr;
    cudaMalloc(&d_send, bytes);
    cudaMalloc(&d_recv, bytes);
    cudaMemset(d_send, 1, bytes);

    for (auto _ : state) {
        ncclReduce(d_send, d_recv, count, ncclFloat, ncclSum, 0,
                    g_nccl.comm, g_nccl.stream);
        cudaStreamSynchronize(g_nccl.stream);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(bytes));
    cudaFree(d_send);
    cudaFree(d_recv);
}

void BM_NcclAllGather(benchmark::State& state) {
    g_nccl.init();
    if (!g_nccl.valid) { state.SkipWithError("NCCL not available"); return; }

    const size_t send_bytes = static_cast<size_t>(state.range(0));
    const size_t send_count = send_bytes / sizeof(float);
    const size_t recv_bytes = send_bytes * static_cast<size_t>(g_nccl.size);

    float *d_send = nullptr, *d_recv = nullptr;
    cudaMalloc(&d_send, send_bytes);
    cudaMalloc(&d_recv, recv_bytes);
    cudaMemset(d_send, 1, send_bytes);

    for (auto _ : state) {
        ncclAllGather(d_send, d_recv, send_count, ncclFloat,
                       g_nccl.comm, g_nccl.stream);
        cudaStreamSynchronize(g_nccl.stream);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                             static_cast<int64_t>(recv_bytes));
    cudaFree(d_send);
    cudaFree(d_recv);
}

// Register with power-of-2 sizes from 4B to 16MB
BENCHMARK(BM_NcclAllReduce)->RangeMultiplier(4)->Range(4, 16 << 20);
BENCHMARK(BM_NcclBroadcast)->RangeMultiplier(4)->Range(4, 16 << 20);
BENCHMARK(BM_NcclReduce)->RangeMultiplier(4)->Range(4, 16 << 20);
BENCHMARK(BM_NcclAllGather)->RangeMultiplier(4)->Range(4, 16 << 20);

}  // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    MPI_Finalize();
    return 0;
}

#else

#include <benchmark/benchmark.h>

static void BM_NcclSkipped(benchmark::State& state) {
    state.SkipWithError("NCCL or CUDA not enabled");
}
BENCHMARK(BM_NcclSkipped);

BENCHMARK_MAIN();

#endif
