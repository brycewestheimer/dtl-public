// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_multithread_device_scoping.cpp
/// @brief Integration test for multi-threaded device scoping
/// @details Verifies that device guards work correctly when multiple threads
///          allocate containers on different devices simultaneously.
/// @since 0.1.0

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA

#include <dtl/dtl.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/cuda/device_guard.hpp>
#include <dtl/memory/cuda_device_memory_space.hpp>

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

namespace {

class MultiThreadDeviceScopingTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_count_ = dtl::cuda::device_count();
        original_device_ = dtl::cuda::current_device_id();
    }

    void TearDown() override {
        if (device_count_ > 0 && original_device_ >= 0) {
            cudaSetDevice(original_device_);
        }
    }

    int device_count_{0};
    int original_device_{-1};
};

#define SKIP_IF_NO_CUDA() \
    if (device_count_ == 0) { \
        GTEST_SKIP() << "No CUDA devices available"; \
    }

#define SKIP_IF_SINGLE_GPU() \
    if (device_count_ < 2) { \
        GTEST_SKIP() << "Test requires at least 2 GPUs"; \
    }

// ============================================================================
// Multi-Thread Device Guard Tests
// ============================================================================

TEST_F(MultiThreadDeviceScopingTest, TwoThreadsTwoDevices) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();

    std::atomic<bool> thread0_success{false};
    std::atomic<bool> thread1_success{false};
    std::atomic<int> thread0_device_before{-1};
    std::atomic<int> thread1_device_before{-1};
    std::atomic<int> thread0_device_after{-1};
    std::atomic<int> thread1_device_after{-1};

    auto worker = [](int target_device, std::atomic<bool>& success,
                     std::atomic<int>& device_before, std::atomic<int>& device_after) {
        // Record device before (should be undefined or 0)
        device_before = dtl::cuda::current_device_id();
        
        // Use device guard
        {
            dtl::cuda::device_guard guard(target_device);
            
            // Allocate memory (should be on target device)
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, 1024);
            if (err != cudaSuccess || ptr == nullptr) {
                success = false;
                return;
            }
            
            // Verify allocation is on correct device
            cudaPointerAttributes attrs;
            err = cudaPointerGetAttributes(&attrs, ptr);
            if (err != cudaSuccess || attrs.device != target_device) {
                cudaFree(ptr);
                success = false;
                return;
            }
            
            cudaFree(ptr);
        }
        
        // Record device after guard (should be same as before)
        device_after = dtl::cuda::current_device_id();
        success = true;
    };

    std::thread t0(worker, 0, std::ref(thread0_success),
                   std::ref(thread0_device_before), std::ref(thread0_device_after));
    std::thread t1(worker, 1, std::ref(thread1_success),
                   std::ref(thread1_device_before), std::ref(thread1_device_after));

    t0.join();
    t1.join();

    EXPECT_TRUE(thread0_success.load());
    EXPECT_TRUE(thread1_success.load());
    
    // Each thread's device should be restored after guard
    EXPECT_EQ(thread0_device_before.load(), thread0_device_after.load());
    EXPECT_EQ(thread1_device_before.load(), thread1_device_after.load());
}

TEST_F(MultiThreadDeviceScopingTest, ContainerCreationFromMultipleThreads) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();

    std::atomic<bool> thread0_success{false};
    std::atomic<bool> thread1_success{false};
    std::atomic<int> container0_device{-1};
    std::atomic<int> container1_device{-1};

    auto worker = [](int target_device, std::atomic<bool>& success,
                     std::atomic<int>& container_device) {
        try {
            auto ctx = dtl::make_cpu_context().with_cuda(target_device);
            dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);
            
            container_device = vec.device_id();
            
            // Verify pointer is on correct device
            int ptr_device = dtl::cuda::get_pointer_device(vec.local_data());
            
            success = (vec.device_id() == target_device && ptr_device == target_device);
        } catch (...) {
            success = false;
        }
    };

    std::thread t0(worker, 0, std::ref(thread0_success), std::ref(container0_device));
    std::thread t1(worker, 1, std::ref(thread1_success), std::ref(container1_device));

    t0.join();
    t1.join();

    EXPECT_TRUE(thread0_success.load());
    EXPECT_TRUE(thread1_success.load());
    EXPECT_EQ(container0_device.load(), 0);
    EXPECT_EQ(container1_device.load(), 1);
}

TEST_F(MultiThreadDeviceScopingTest, ManyThreadsManyAllocations) {
    SKIP_IF_NO_CUDA();
    
    constexpr int num_threads = 8;
    constexpr int allocations_per_thread = 10;
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    auto worker = [this, &success_count, &failure_count](int thread_id) {
        int target_device = thread_id % device_count_;
        
        for (int i = 0; i < allocations_per_thread; ++i) {
            try {
                auto ctx = dtl::make_cpu_context().with_cuda(target_device);
                dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);
                
                if (vec.device_id() == target_device) {
                    int ptr_device = dtl::cuda::get_pointer_device(vec.local_data());
                    if (ptr_device == target_device) {
                        success_count++;
                        continue;
                    }
                }
                failure_count++;
            } catch (...) {
                failure_count++;
            }
        }
    };

    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads * allocations_per_thread);
    EXPECT_EQ(failure_count.load(), 0);
}

TEST_F(MultiThreadDeviceScopingTest, NoDeviceContamination) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();
    
    // This test ensures that device operations in one thread don't affect another
    
    std::mutex mutex;
    std::vector<std::pair<int, int>> results;  // (expected_device, actual_device)

    auto worker = [&mutex, &results](int target_device) {
        dtl::cuda::device_guard guard(target_device);
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Check current device
        int current = dtl::cuda::current_device_id();
        
        {
            std::lock_guard<std::mutex> lock(mutex);
            results.emplace_back(target_device, current);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(worker, i % device_count_);
    }

    for (auto& t : threads) {
        t.join();
    }

    // All results should match: expected == actual
    for (const auto& [expected, actual] : results) {
        EXPECT_EQ(expected, actual) << "Device contamination detected";
    }
}

TEST_F(MultiThreadDeviceScopingTest, NestedGuardsInThreads) {
    SKIP_IF_NO_CUDA();
    SKIP_IF_SINGLE_GPU();

    std::atomic<bool> success{true};

    auto worker = [this, &success]() {
        int initial = dtl::cuda::current_device_id();
        
        {
            dtl::cuda::device_guard outer(0);
            int after_outer = dtl::cuda::current_device_id();
            
            if (after_outer != 0) {
                success = false;
                return;
            }
            
            {
                dtl::cuda::device_guard inner(1);
                int after_inner = dtl::cuda::current_device_id();
                
                if (after_inner != 1) {
                    success = false;
                    return;
                }
            }
            
            // After inner guard, should be back to 0
            int after_inner_destroyed = dtl::cuda::current_device_id();
            if (after_inner_destroyed != 0) {
                success = false;
                return;
            }
        }
        
        // After outer guard, should be back to initial
        int final_device = dtl::cuda::current_device_id();
        if (final_device != initial) {
            success = false;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_TRUE(success.load());
}

}  // namespace

#else  // !DTL_ENABLE_CUDA

#include <gtest/gtest.h>

TEST(MultiThreadDeviceScopingTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA not enabled in this build";
}

#endif  // DTL_ENABLE_CUDA
