// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file bench_mpi_collectives.cpp
/// @brief Benchmarks for MPI collective operations via DTL communicator
/// @details Measures latency of allreduce, broadcast, barrier, and send/recv
///          across a range of message sizes.
///          Run with: mpirun -n <N> ./bench_mpi_collectives

#include <dtl/core/config.hpp>

#if DTL_ENABLE_MPI

#include <dtl/core/environment.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/communication/collective_ops.hpp>
#include <dtl/communication/point_to_point.hpp>
#include <dtl/communication/reduction_ops.hpp>

#include <benchmark/benchmark.h>
#include <mpi.h>

#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace {

// Global DTL environment — initialized in main
static dtl::environment* g_env = nullptr;

// ============================================================================
// MPI Allreduce Latency — varying message sizes
// ============================================================================

static void BM_MpiAllreduce(benchmark::State& state) {
    dtl::mpi_domain mpi;
    auto& comm = mpi.communicator();
    const int64_t bytes = state.range(0);
    const size_t count = static_cast<size_t>(bytes) / sizeof(int);
    if (count == 0) {
        state.SkipWithError("Message size too small for int allreduce");
        return;
    }

    std::vector<int> send(count, 1);
    std::vector<int> recv(count, 0);

    for (auto _ : state) {
        dtl::allreduce(comm,
                       std::span<const int>(send),
                       std::span<int>(recv),
                       dtl::reduce_sum<>{});
    }

    state.SetBytesProcessed(state.iterations() * bytes);
    state.counters["ranks"] = static_cast<double>(comm.size());
    state.counters["msg_bytes"] = static_cast<double>(bytes);
}

BENCHMARK(BM_MpiAllreduce)
    ->Arg(4)           // 1 int = 4 B
    ->Arg(64)          // 16 ints
    ->Arg(256)
    ->Arg(1024)        // 1 KB
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)       // 64 KB
    ->Arg(262144)
    ->Arg(1048576)     // 1 MB
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Iterations(1000);

// ============================================================================
// MPI Broadcast Latency
// ============================================================================

static void BM_MpiBroadcast(benchmark::State& state) {
    dtl::mpi_domain mpi;
    auto& comm = mpi.communicator();
    const int64_t bytes = state.range(0);
    const size_t count = static_cast<size_t>(bytes);

    std::vector<char> data(count, 0);
    if (comm.rank() == 0) {
        std::fill(data.begin(), data.end(), static_cast<char>(42));
    }

    for (auto _ : state) {
        dtl::broadcast(comm, std::span<char>(data), /*root=*/0);
    }

    state.SetBytesProcessed(state.iterations() * bytes);
    state.counters["ranks"] = static_cast<double>(comm.size());
}

BENCHMARK(BM_MpiBroadcast)
    ->Arg(1)
    ->Arg(64)
    ->Arg(1024)
    ->Arg(16384)
    ->Arg(262144)
    ->Arg(1048576)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Iterations(1000);

// ============================================================================
// MPI Barrier Latency
// ============================================================================

static void BM_MpiBarrier(benchmark::State& state) {
    dtl::mpi_domain mpi;
    auto& comm = mpi.communicator();

    for (auto _ : state) {
        comm.barrier();
    }

    state.counters["ranks"] = static_cast<double>(comm.size());
}

BENCHMARK(BM_MpiBarrier)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Iterations(10000);

// ============================================================================
// MPI Send/Recv Bandwidth (ping-pong between rank 0 and rank 1)
// ============================================================================

static void BM_MpiSendRecvBandwidth(benchmark::State& state) {
    dtl::mpi_domain mpi;
    auto& comm = mpi.communicator();
    if (comm.size() < 2) {
        state.SkipWithError("Send/Recv benchmark requires at least 2 ranks");
        return;
    }

    const int64_t bytes = state.range(0);
    const size_t count = static_cast<size_t>(bytes);
    std::vector<char> buf(count, 0);
    constexpr int tag = 99;

    for (auto _ : state) {
        if (comm.rank() == 0) {
            comm.send(buf.data(), count, 1, tag);
            comm.recv(buf.data(), count, 1, tag);
        } else if (comm.rank() == 1) {
            comm.recv(buf.data(), count, 0, tag);
            comm.send(buf.data(), count, 0, tag);
        }
        // Other ranks idle — consistent with ping-pong pattern
    }

    // 2x bytes per iteration (send + recv round-trip)
    state.SetBytesProcessed(state.iterations() * bytes * 2);
    state.counters["ranks"] = static_cast<double>(comm.size());
}

BENCHMARK(BM_MpiSendRecvBandwidth)
    ->Arg(1)
    ->Arg(64)
    ->Arg(1024)
    ->Arg(16384)
    ->Arg(262144)
    ->Arg(1048576)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Iterations(1000);

}  // anonymous namespace

// ============================================================================
// Custom main — MPI must be initialized before benchmarks run
// ============================================================================

int main(int argc, char** argv) {
    g_env = new dtl::environment(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        delete g_env;
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    delete g_env;
    g_env = nullptr;
    return 0;
}

#else  // !DTL_ENABLE_MPI

#include <benchmark/benchmark.h>

static void BM_MpiNotEnabled(benchmark::State& state) {
    for (auto _ : state) {
        // no-op
    }
    state.SkipWithError("MPI not enabled — skipping MPI benchmarks");
}

BENCHMARK(BM_MpiNotEnabled);

BENCHMARK_MAIN();

#endif  // DTL_ENABLE_MPI
