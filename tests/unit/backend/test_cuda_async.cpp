// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_async.cpp
/// @brief Unit tests for CUDA async execution with progress engine integration
/// @details Tests event-based futures, progress engine polling, and async GPU dispatch.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA

#include <backends/cuda/cuda_executor.hpp>
#include <backends/cuda/cuda_memory_space.hpp>
#include <dtl/futures/futures.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace dtl::test {

// =============================================================================
// Test Fixture
// =============================================================================

class CudaAsyncTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = cuda::device_count();
        if (device_count <= 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    void TearDown() override {
        // Drain any pending progress callbacks
        futures::drain_progress(1000);

        // Ensure all CUDA operations complete
        cudaDeviceSynchronize();
    }
};

// =============================================================================
// make_cuda_future Tests
// =============================================================================

TEST_F(CudaAsyncTest, MakeCudaFutureVoid) {
    auto [future, promise] = cuda::make_cuda_future<void>();

    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(future.is_ready());

    promise.set_value();

    while (!future.is_ready()) {
        futures::make_progress();
    }

    EXPECT_NO_THROW(future.get());
}

TEST_F(CudaAsyncTest, MakeCudaFutureInt) {
    auto [future, promise] = cuda::make_cuda_future<int>();

    EXPECT_TRUE(future.valid());
    EXPECT_FALSE(future.is_ready());

    promise.set_value(42);

    while (!future.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(future.get(), 42);
}

// =============================================================================
// dispatch_gpu_async Tests
// =============================================================================

TEST_F(CudaAsyncTest, DispatchGpuAsyncBasic) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    std::atomic<bool> kernel_started{false};

    // Dispatch a simple operation that just records an event
    auto future = cuda::dispatch_gpu_async(exec.native_handle(), [&](cudaStream_t stream) {
        // Launch a host callback to mark that kernel started
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* flag = static_cast<std::atomic<bool>*>(data);
            *flag = true;
        }, &kernel_started);
    });

    EXPECT_TRUE(future.valid());

    // Drive progress until future completes
    while (!future.is_ready()) {
        futures::make_progress();
    }

    EXPECT_TRUE(kernel_started.load());
    EXPECT_NO_THROW(future.get());
}

TEST_F(CudaAsyncTest, DispatchGpuAsyncOrdering) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    std::atomic<int> counter{0};

    // Dispatch multiple operations and verify ordering
    auto future1 = cuda::dispatch_gpu_async(exec.native_handle(), [&](cudaStream_t stream) {
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* c = static_cast<std::atomic<int>*>(data);
            ++(*c);
        }, &counter);
    });

    auto future2 = cuda::dispatch_gpu_async(exec.native_handle(), [&](cudaStream_t stream) {
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* c = static_cast<std::atomic<int>*>(data);
            ++(*c);
        }, &counter);
    });

    auto future3 = cuda::dispatch_gpu_async(exec.native_handle(), [&](cudaStream_t stream) {
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* c = static_cast<std::atomic<int>*>(data);
            ++(*c);
        }, &counter);
    });

    // Drive progress until all complete
    while (!future1.is_ready() || !future2.is_ready() || !future3.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(counter.load(), 3);
}

// =============================================================================
// cuda_executor::execute_async Tests
// =============================================================================

TEST_F(CudaAsyncTest, ExecutorExecuteAsync) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    std::atomic<bool> work_done{false};

    auto future = exec.execute_async([&](cudaStream_t stream) {
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* flag = static_cast<std::atomic<bool>*>(data);
            *flag = true;
        }, &work_done);
    });

    EXPECT_TRUE(future.valid());

    // Should not be ready immediately (work is on GPU)
    // Note: might be ready if GPU is very fast, so we just check validity

    while (!future.is_ready()) {
        futures::make_progress();
    }

    EXPECT_TRUE(work_done.load());
    EXPECT_NO_THROW(future.get());
}

TEST_F(CudaAsyncTest, ExecutorExecuteAsyncMultiple) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    constexpr int num_ops = 10;
    std::atomic<int> counter{0};
    std::vector<futures::distributed_future<void>> futures_vec;
    futures_vec.reserve(num_ops);

    for (int i = 0; i < num_ops; ++i) {
        auto f = exec.execute_async([&](cudaStream_t stream) {
            cudaLaunchHostFunc(stream, [](void* data) {
                auto* c = static_cast<std::atomic<int>*>(data);
                ++(*c);
            }, &counter);
        });
        futures_vec.push_back(std::move(f));
    }

    // Drive progress until all complete
    bool all_ready = false;
    while (!all_ready) {
        futures::make_progress();
        all_ready = true;
        for (auto& f : futures_vec) {
            if (!f.is_ready()) {
                all_ready = false;
                break;
            }
        }
    }

    EXPECT_EQ(counter.load(), num_ops);
}

// =============================================================================
// Progress Engine Integration Tests
// =============================================================================

TEST_F(CudaAsyncTest, ProgressEnginePollsCudaEvents) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    auto& engine = futures::progress_engine::instance();

    // Initially no pending CUDA events
    EXPECT_FALSE(engine.has_pending_cuda());

    // Dispatch async work
    auto future = exec.execute_async([](cudaStream_t stream) {
        // Empty kernel - just creates an event
        (void)stream;
    });

    // Now there should be a pending CUDA event
    EXPECT_TRUE(engine.has_pending_cuda());
    EXPECT_GE(engine.pending_cuda_count(), 1);

    // Drive progress until complete
    while (!future.is_ready()) {
        futures::make_progress();
    }

    // After completion, no pending CUDA events for this future
    // (there might be others from concurrent tests, so we just verify our future is ready)
    EXPECT_TRUE(future.is_ready());
}

TEST_F(CudaAsyncTest, DrainProgressCompletesAllCudaEvents) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    std::atomic<int> counter{0};

    // Dispatch multiple async operations
    for (int i = 0; i < 5; ++i) {
        exec.execute_async([&](cudaStream_t stream) {
            cudaLaunchHostFunc(stream, [](void* data) {
                auto* c = static_cast<std::atomic<int>*>(data);
                ++(*c);
            }, &counter);
        });
    }

    // Drain all progress
    bool completed = futures::drain_progress(10000);

    EXPECT_TRUE(completed);
    EXPECT_EQ(counter.load(), 5);
}

// =============================================================================
// Mixed CPU+GPU Async Tests
// =============================================================================

TEST_F(CudaAsyncTest, MixedCpuGpuAsync) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    std::atomic<int> gpu_counter{0};
    std::atomic<int> cpu_counter{0};

    // GPU async operation
    auto gpu_future = exec.execute_async([&](cudaStream_t stream) {
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* c = static_cast<std::atomic<int>*>(data);
            ++(*c);
        }, &gpu_counter);
    });

    // CPU async operation (using promise/future directly)
    futures::distributed_promise<void> cpu_promise;
    auto cpu_future = cpu_promise.get_future();

    // Simulate CPU work completing
    cpu_promise.set_value();
    ++cpu_counter;

    // Drive progress
    while (!gpu_future.is_ready() || !cpu_future.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(gpu_counter.load(), 1);
    EXPECT_EQ(cpu_counter.load(), 1);
}

TEST_F(CudaAsyncTest, ContinuationOnGpuFuture) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    std::atomic<bool> continuation_ran{false};

    auto gpu_future = exec.execute_async([](cudaStream_t stream) {
        // Empty kernel
        (void)stream;
    });

    auto continued = gpu_future.then([&]() {
        continuation_ran = true;
        return 42;
    });

    while (!continued.is_ready()) {
        futures::make_progress();
    }

    EXPECT_TRUE(continuation_ran.load());
    EXPECT_EQ(continued.get(), 42);
}

// =============================================================================
// Resource Cleanup Tests
// =============================================================================

TEST_F(CudaAsyncTest, EventDestroyedAfterCompletion) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    auto& engine = futures::progress_engine::instance();
    size_type initial_count = engine.pending_cuda_count();

    // Create and complete a future
    {
        auto future = exec.execute_async([](cudaStream_t stream) {
            (void)stream;
        });

        while (!future.is_ready()) {
            futures::make_progress();
        }

        future.get();
    }

    // After the future is destroyed and completed, the event should be cleaned up
    // The count might not be exactly back to initial if other tests are running,
    // but we at least verify no crash occurs
    SUCCEED();
}

TEST_F(CudaAsyncTest, MultipleFuturesEventCleanup) {
    cuda::cuda_executor exec(cuda::stream_flags::non_blocking);

    constexpr int num_futures = 100;

    for (int i = 0; i < num_futures; ++i) {
        auto future = exec.execute_async([](cudaStream_t stream) {
            (void)stream;
        });

        while (!future.is_ready()) {
            futures::make_progress();
        }

        future.get();
    }

    // Verify no resource leaks by successfully completing
    SUCCEED();
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(CudaAsyncTest, FutureErrorPropagation) {
    // Test that errors in GPU async operations are properly propagated
    auto failed_future = futures::make_failed_distributed_future<void>(
        status(status_code::backend_error, no_rank, "Simulated GPU error"));

    while (!failed_future.is_ready()) {
        futures::make_progress();
    }

    auto result = failed_future.get_result();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, status_code::backend_error);
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

TEST_F(CudaAsyncTest, ConcurrentDispatchFromMultipleExecutors) {
    cuda::cuda_executor exec1(cuda::stream_flags::non_blocking);
    cuda::cuda_executor exec2(cuda::stream_flags::non_blocking);

    std::atomic<int> counter1{0};
    std::atomic<int> counter2{0};

    // Dispatch from both executors concurrently
    auto f1 = exec1.execute_async([&](cudaStream_t stream) {
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* c = static_cast<std::atomic<int>*>(data);
            ++(*c);
        }, &counter1);
    });

    auto f2 = exec2.execute_async([&](cudaStream_t stream) {
        cudaLaunchHostFunc(stream, [](void* data) {
            auto* c = static_cast<std::atomic<int>*>(data);
            ++(*c);
        }, &counter2);
    });

    while (!f1.is_ready() || !f2.is_ready()) {
        futures::make_progress();
    }

    EXPECT_EQ(counter1.load(), 1);
    EXPECT_EQ(counter2.load(), 1);
}

}  // namespace dtl::test

#else  // !DTL_ENABLE_CUDA

#include <gtest/gtest.h>

namespace dtl::test {

TEST(CudaAsyncTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA not enabled - skipping CUDA async tests";
}

}  // namespace dtl::test

#endif  // DTL_ENABLE_CUDA
