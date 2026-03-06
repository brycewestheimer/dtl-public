// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file bench_scaling.cpp
/// @brief Multi-rank strong/weak scaling benchmarks for DTL
/// @details Run via mpirun: mpirun -np N ./bench_scaling --benchmark_out=result.json
///          Reports ranks, msg_bytes, and efficiency as benchmark counters.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_MPI

#include <dtl/dtl.hpp>
#include <dtl/core/environment.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/algorithms.hpp>
#include <dtl/communication/communication.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>

#include <benchmark/benchmark.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace {

// Global MPI state
dtl::environment* g_env = nullptr;
dtl::mpi::mpi_comm_adapter g_comm{};
dtl::rank_t g_rank = 0;
dtl::rank_t g_size = 1;

// ============================================================================
// Strong Scaling: Allreduce — Fixed global size (1M doubles)
// ============================================================================

void BM_StrongScaling_Allreduce(benchmark::State& state) {
    constexpr dtl::size_type global_elements = 1'000'000;
    const dtl::size_type local_elements = global_elements / static_cast<dtl::size_type>(g_size);
    const dtl::size_type msg_bytes = local_elements * sizeof(double);

    std::vector<double> send(local_elements, static_cast<double>(g_rank + 1));
    std::vector<double> recv(local_elements, 0.0);

    // Warmup
    dtl::allreduce(g_comm,
                   std::span<const double>(send),
                   std::span<double>(recv),
                   dtl::reduce_sum<>{});
    g_comm.barrier();

    for (auto _ : state) {
        dtl::allreduce(g_comm,
                       std::span<const double>(send),
                       std::span<double>(recv),
                       dtl::reduce_sum<>{});
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(msg_bytes));
    state.counters["ranks"] = static_cast<double>(g_size);
    state.counters["msg_bytes"] = static_cast<double>(msg_bytes);
    state.counters["global_elements"] = static_cast<double>(global_elements);
}

BENCHMARK(BM_StrongScaling_Allreduce)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Iterations(100);

// ============================================================================
// Weak Scaling: Allreduce — Fixed per-rank size (64K doubles)
// ============================================================================

void BM_WeakScaling_Allreduce(benchmark::State& state) {
    constexpr dtl::size_type per_rank_elements = 65'536;
    const dtl::size_type msg_bytes = per_rank_elements * sizeof(double);

    std::vector<double> send(per_rank_elements, static_cast<double>(g_rank + 1));
    std::vector<double> recv(per_rank_elements, 0.0);

    // Warmup
    dtl::allreduce(g_comm,
                   std::span<const double>(send),
                   std::span<double>(recv),
                   dtl::reduce_sum<>{});
    g_comm.barrier();

    for (auto _ : state) {
        dtl::allreduce(g_comm,
                       std::span<const double>(send),
                       std::span<double>(recv),
                       dtl::reduce_sum<>{});
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(msg_bytes));
    state.counters["ranks"] = static_cast<double>(g_size);
    state.counters["msg_bytes"] = static_cast<double>(msg_bytes);
    state.counters["per_rank_elements"] = static_cast<double>(per_rank_elements);
}

BENCHMARK(BM_WeakScaling_Allreduce)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Iterations(100);

// ============================================================================
// Strong Scaling: Distributed Sort — Fixed 1M ints
// ============================================================================

void BM_StrongScaling_DistributedSort(benchmark::State& state) {
    constexpr dtl::size_type global_elements = 1'000'000;

    for (auto _ : state) {
        state.PauseTiming();
        dtl::mpi_domain mpi;
        dtl::distributed_vector<int> vec(global_elements, mpi);

        // Fill with reverse-order values
        auto local = vec.local_view();
        for (dtl::size_type i = 0; i < local.size(); ++i) {
            local[i] = static_cast<int>(global_elements) -
                       static_cast<int>(g_rank * local.size() + i);
        }
        g_comm.barrier();
        state.ResumeTiming();

        dtl::sort(dtl::seq{}, vec, std::less<>{}, g_comm);
    }

    state.counters["ranks"] = static_cast<double>(g_size);
    state.counters["global_elements"] = static_cast<double>(global_elements);
    state.counters["msg_bytes"] = static_cast<double>(
        (global_elements / static_cast<dtl::size_type>(g_size)) * sizeof(int));
}

BENCHMARK(BM_StrongScaling_DistributedSort)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(10);

}  // namespace

// ============================================================================
// MPI-aware main
// ============================================================================

int main(int argc, char** argv) {
    g_env = new dtl::environment(argc, argv);

    auto ctx = g_env->make_world_context();
    g_rank = ctx.rank();
    g_size = ctx.size();
    g_comm = ctx.get<dtl::mpi_domain>().communicator();

    ::benchmark::Initialize(&argc, argv);

    // Only rank 0 reports to avoid interleaved output
    if (g_rank == 0) {
        ::benchmark::RunSpecifiedBenchmarks();
    } else {
        // Non-root ranks still participate in collectives
        ::benchmark::RunSpecifiedBenchmarks();
    }

    ::benchmark::Shutdown();

    delete g_env;
    g_env = nullptr;

    return 0;
}

#else  // !DTL_ENABLE_MPI

#include <benchmark/benchmark.h>

void BM_ScalingSkipped(benchmark::State& state) {
    for (auto _ : state) {
        // No-op: MPI not available
    }
}

BENCHMARK(BM_ScalingSkipped);
BENCHMARK_MAIN();

#endif  // DTL_ENABLE_MPI
