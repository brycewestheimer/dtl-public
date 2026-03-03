// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file bench_c_api.cpp
/// @brief Benchmarks for DTL C API (C ABI) overhead
/// @details Compares C API operations against native C++ API to quantify
///          the C binding overhead. Uses Google Benchmark.

#include <benchmark/benchmark.h>

#include <dtl/bindings/c/dtl.h>

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/views/local_view.hpp>

#include <cstdint>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

namespace {

/// Lightweight single-rank context for native C++ vector construction.
struct single_rank_ctx {
    [[nodiscard]] dtl::rank_t rank() const noexcept { return 0; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return 1; }
};

/// RAII wrapper so every benchmark function doesn't repeat create/destroy.
struct c_context_guard {
    dtl_context_t ctx = nullptr;
    c_context_guard() { dtl_context_create_default(&ctx); }
    ~c_context_guard() {
        if (ctx) dtl_context_destroy(ctx);
    }
    c_context_guard(const c_context_guard&) = delete;
    c_context_guard& operator=(const c_context_guard&) = delete;
};

} // namespace

// ============================================================================
// Vector create / destroy cycle — C API
// ============================================================================

static void BM_CApi_VectorCreateDestroy(benchmark::State& state) {
    c_context_guard g;
    const auto n = static_cast<dtl_size_t>(state.range(0));

    for (auto _ : state) {
        dtl_vector_t vec = nullptr;
        dtl_vector_create(g.ctx, DTL_DTYPE_FLOAT64, n, &vec);
        benchmark::DoNotOptimize(vec);
        dtl_vector_destroy(vec);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_CApi_VectorCreateDestroy)
    ->RangeMultiplier(10)
    ->Range(100, 1'000'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Vector create / destroy cycle — native C++ API
// ============================================================================

static void BM_Cpp_VectorCreateDestroy(benchmark::State& state) {
    const auto n = static_cast<dtl::size_type>(state.range(0));
    single_rank_ctx ctx;

    for (auto _ : state) {
        dtl::distributed_vector<double> vec(n, ctx);
        benchmark::DoNotOptimize(vec.local_view().data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_Cpp_VectorCreateDestroy)
    ->RangeMultiplier(10)
    ->Range(100, 1'000'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Vector create-fill / destroy — C API
// ============================================================================

static void BM_CApi_VectorCreateFillDestroy(benchmark::State& state) {
    c_context_guard g;
    const auto n = static_cast<dtl_size_t>(state.range(0));
    double fill = 42.0;

    for (auto _ : state) {
        dtl_vector_t vec = nullptr;
        dtl_vector_create_fill(g.ctx, DTL_DTYPE_FLOAT64, n, &fill, &vec);
        benchmark::DoNotOptimize(vec);
        dtl_vector_destroy(vec);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_CApi_VectorCreateFillDestroy)
    ->RangeMultiplier(10)
    ->Range(100, 1'000'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Vector element access — C API (get_local / set_local)
// ============================================================================

static void BM_CApi_VectorElementAccess(benchmark::State& state) {
    c_context_guard g;
    const auto n = static_cast<dtl_size_t>(state.range(0));

    dtl_vector_t vec = nullptr;
    dtl_vector_create(g.ctx, DTL_DTYPE_INT32, n, &vec);

    int32_t val = 0;
    for (auto _ : state) {
        for (dtl_size_t i = 0; i < dtl_vector_local_size(vec); ++i) {
            dtl_vector_get_local(vec, i, &val);
            benchmark::DoNotOptimize(val);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(dtl_vector_local_size(vec)));
    dtl_vector_destroy(vec);
}

BENCHMARK(BM_CApi_VectorElementAccess)
    ->RangeMultiplier(10)
    ->Range(100, 100'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Vector element access — raw pointer (baseline)
// ============================================================================

static void BM_RawPointer_VectorElementAccess(benchmark::State& state) {
    c_context_guard g;
    const auto n = static_cast<dtl_size_t>(state.range(0));

    dtl_vector_t vec = nullptr;
    dtl_vector_create(g.ctx, DTL_DTYPE_INT32, n, &vec);

    const int32_t* data = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);

    int32_t val = 0;
    for (auto _ : state) {
        for (dtl_size_t i = 0; i < local_size; ++i) {
            val = data[i];
            benchmark::DoNotOptimize(val);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(local_size));
    dtl_vector_destroy(vec);
}

BENCHMARK(BM_RawPointer_VectorElementAccess)
    ->RangeMultiplier(10)
    ->Range(100, 100'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Vector fill via C API  (dtl_fill_vector)
// ============================================================================

static void BM_CApi_VectorFill(benchmark::State& state) {
    c_context_guard g;
    const auto n = static_cast<dtl_size_t>(state.range(0));

    dtl_vector_t vec = nullptr;
    dtl_vector_create(g.ctx, DTL_DTYPE_FLOAT64, n, &vec);
    double fill = 1.0;

    for (auto _ : state) {
        dtl_fill_vector(vec, &fill);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(dtl_vector_local_size(vec)));
    dtl_vector_destroy(vec);
}

BENCHMARK(BM_CApi_VectorFill)
    ->RangeMultiplier(10)
    ->Range(100, 1'000'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Vector local reduce (sum) — C API
// ============================================================================

static void BM_CApi_VectorReduceLocalSum(benchmark::State& state) {
    c_context_guard g;
    const auto n = static_cast<dtl_size_t>(state.range(0));

    double fill = 1.0;
    dtl_vector_t vec = nullptr;
    dtl_vector_create_fill(g.ctx, DTL_DTYPE_FLOAT64, n, &fill, &vec);

    double result = 0.0;
    for (auto _ : state) {
        dtl_reduce_local_vector(vec, DTL_OP_SUM, &result);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(dtl_vector_local_size(vec)));
    dtl_vector_destroy(vec);
}

BENCHMARK(BM_CApi_VectorReduceLocalSum)
    ->RangeMultiplier(10)
    ->Range(100, 1'000'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Vector sort — C API
// ============================================================================

static void BM_CApi_VectorSort(benchmark::State& state) {
    c_context_guard g;
    const auto n = static_cast<dtl_size_t>(state.range(0));

    dtl_vector_t vec = nullptr;
    dtl_vector_create(g.ctx, DTL_DTYPE_INT32, n, &vec);

    for (auto _ : state) {
        state.PauseTiming();
        // Refill in reverse order each iteration
        int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
        dtl_size_t local_size = dtl_vector_local_size(vec);
        for (dtl_size_t i = 0; i < local_size; ++i) {
            data[i] = static_cast<int32_t>(local_size - i);
        }
        state.ResumeTiming();

        dtl_sort_vector(vec);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(dtl_vector_local_size(vec)));
    dtl_vector_destroy(vec);
}

BENCHMARK(BM_CApi_VectorSort)
    ->RangeMultiplier(10)
    ->Range(100, 100'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Allreduce overhead — C API (single-rank, measures call overhead)
// ============================================================================

static void BM_CApi_Allreduce(benchmark::State& state) {
    c_context_guard g;
    const auto count = static_cast<dtl_size_t>(state.range(0));

    std::vector<double> sendbuf(count, 1.0);
    std::vector<double> recvbuf(count, 0.0);

    for (auto _ : state) {
        dtl_allreduce(g.ctx, sendbuf.data(), recvbuf.data(),
                      count, DTL_DTYPE_FLOAT64, DTL_OP_SUM);
        benchmark::DoNotOptimize(recvbuf.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count));
}

BENCHMARK(BM_CApi_Allreduce)
    ->RangeMultiplier(10)
    ->Range(1, 10'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Broadcast overhead — C API
// ============================================================================

static void BM_CApi_Broadcast(benchmark::State& state) {
    c_context_guard g;
    const auto count = static_cast<dtl_size_t>(state.range(0));

    std::vector<int32_t> buf(count, 42);

    for (auto _ : state) {
        dtl_broadcast(g.ctx, buf.data(), count, DTL_DTYPE_INT32, 0);
        benchmark::DoNotOptimize(buf.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(count));
}

BENCHMARK(BM_CApi_Broadcast)
    ->RangeMultiplier(10)
    ->Range(1, 10'000)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Context create/destroy overhead
// ============================================================================

static void BM_CApi_ContextCreateDestroy(benchmark::State& state) {
    for (auto _ : state) {
        dtl_context_t ctx = nullptr;
        dtl_context_create_default(&ctx);
        benchmark::DoNotOptimize(ctx);
        dtl_context_destroy(ctx);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_CApi_ContextCreateDestroy)->Unit(benchmark::kMicrosecond);
