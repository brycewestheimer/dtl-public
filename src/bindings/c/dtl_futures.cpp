// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_futures.cpp
 * @brief DTL C bindings - Futures implementation
 * @since 0.1.0
 */

#include <dtl/bindings/c/dtl_futures.h>
#include <dtl/futures/progress.hpp>

#include "dtl_internal.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <vector>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

// ============================================================================
// Internal Structures
// ============================================================================

/**
 * Future implementation
 *
 * Stores an arbitrary byte buffer as the result value, with
 * thread-safe completion signaling via mutex + condition variable.
 */
struct dtl_future_s {
    // Completion flag (atomic for lock-free test)
    std::atomic<bool> completed{false};

    // Stored result value (arbitrary bytes)
    std::vector<char> value;

    // Synchronization primitives for blocking wait
    std::mutex mtx;
    std::condition_variable cv;

    // Validation magic number
    uint32_t magic;
    static constexpr uint32_t VALID_MAGIC = 0xDEADF070;
};

// ============================================================================
// Validation Helpers
// ============================================================================

static bool is_valid_future(dtl_future_t fut) {
    return fut && fut->magic == dtl_future_s::VALID_MAGIC;
}

// ============================================================================
// Future Lifecycle
// ============================================================================

extern "C" {

dtl_status dtl_future_create(dtl_future_t* fut) {
    if (!fut) {
        return DTL_ERROR_NULL_POINTER;
    }

    dtl_future_s* impl = nullptr;
    try {
        impl = new dtl_future_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    impl->completed.store(false, std::memory_order_relaxed);
    impl->magic = dtl_future_s::VALID_MAGIC;

    *fut = impl;
    return DTL_SUCCESS;
}

void dtl_future_destroy(dtl_future_t fut) {
    if (!is_valid_future(fut)) {
        return;
    }

    fut->magic = 0;
    delete fut;
}

// ============================================================================
// Future Synchronization
// ============================================================================

dtl_status dtl_future_wait(dtl_future_t fut) {
    if (!is_valid_future(fut)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Fast path: already complete
    if (fut->completed.load(std::memory_order_acquire)) {
        return DTL_SUCCESS;
    }

    // Slow path: progress-aware wait loop
    while (!fut->completed.load(std::memory_order_acquire)) {
        dtl::futures::progress_engine::instance().poll();

        std::unique_lock<std::mutex> lock(fut->mtx);
        fut->cv.wait_for(lock, std::chrono::milliseconds(1), [&]() {
            return fut->completed.load(std::memory_order_acquire);
        });
    }

    return DTL_SUCCESS;
}

dtl_status dtl_future_test(dtl_future_t fut, int* completed) {
    if (!is_valid_future(fut)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!completed) {
        return DTL_ERROR_NULL_POINTER;
    }

    dtl::futures::progress_engine::instance().poll();
    *completed = fut->completed.load(std::memory_order_acquire) ? 1 : 0;
    return DTL_SUCCESS;
}

// ============================================================================
// Future Value Access
// ============================================================================

dtl_status dtl_future_get(dtl_future_t fut, void* buffer, dtl_size_t size) {
    if (!is_valid_future(fut)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }
    if (!buffer) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!fut->completed.load(std::memory_order_acquire)) {
        return DTL_ERROR_INVALID_STATE;
    }

    // Check buffer size is sufficient
    std::lock_guard<std::mutex> lock(fut->mtx);
    if (size < fut->value.size()) {
        return DTL_ERROR_BUFFER_TOO_SMALL;
    }

    if (!fut->value.empty()) {
        std::memcpy(buffer, fut->value.data(), fut->value.size());
    }

    return DTL_SUCCESS;
}

dtl_status dtl_future_set(dtl_future_t fut, const void* value,
                           dtl_size_t size) {
    if (!is_valid_future(fut)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Prevent double-completion
    if (fut->completed.load(std::memory_order_acquire)) {
        return DTL_ERROR_INVALID_STATE;
    }

    {
        std::lock_guard<std::mutex> lock(fut->mtx);

        // Store the value
        if (size > 0 && value) {
            try {
                fut->value.assign(
                    static_cast<const char*>(value),
                    static_cast<const char*>(value) + size);
            } catch (...) {
                return DTL_ERROR_ALLOCATION_FAILED;
            }
        }

        // Mark as completed
        fut->completed.store(true, std::memory_order_release);
    }

    // Wake all waiters
    fut->cv.notify_all();

    return DTL_SUCCESS;
}

// ============================================================================
// Future Combinators
// ============================================================================

dtl_status dtl_when_all(const dtl_future_t* futures, dtl_size_t count,
                         dtl_future_t* result) {
    if (!futures) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!result) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (count == 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Create the result future
    dtl_future_t res = nullptr;
    dtl_status status = dtl_future_create(&res);
    if (status != DTL_SUCCESS) {
        return status;
    }

    // Copy the input future handles for the background thread
    std::vector<dtl_future_t> input_futures;
    try {
        input_futures.assign(futures, futures + count);
    } catch (...) {
        dtl_future_destroy(res);
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Drive completion through the progress engine (no detached threads).
    try {
        dtl::futures::progress_engine::instance().register_callback(
            [res, input_futures = std::move(input_futures)]() mutable -> bool {
                for (auto& f : input_futures) {
                    if (!is_valid_future(f)) {
                        continue;
                    }
                    int done = 0;
                    auto test_status = dtl_future_test(f, &done);
                    if (test_status != DTL_SUCCESS || !done) {
                        return true;  // keep polling
                    }
                }
                dtl_future_set(res, nullptr, 0);
                return false;
            });
    } catch (...) {
        dtl_future_destroy(res);
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    *result = res;
    return DTL_SUCCESS;
}

dtl_status dtl_when_any(const dtl_future_t* futures, dtl_size_t count,
                         dtl_future_t* result,
                         dtl_size_t* completed_index) {
    if (!futures) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!result) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (!completed_index) {
        return DTL_ERROR_NULL_POINTER;
    }
    if (count == 0) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Create the result future
    dtl_future_t res = nullptr;
    dtl_status status = dtl_future_create(&res);
    if (status != DTL_SUCCESS) {
        return status;
    }

    // Copy the input future handles for the background thread
    std::vector<dtl_future_t> input_futures;
    try {
        input_futures.assign(futures, futures + count);
    } catch (...) {
        dtl_future_destroy(res);
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Do not retain caller-owned completed_index for async writes.
    // Index will be stored in the result future value payload.
    *completed_index = count;

    try {
        dtl::futures::progress_engine::instance().register_callback(
            [res, input_futures = std::move(input_futures)]() mutable -> bool {
                for (dtl_size_t i = 0; i < input_futures.size(); ++i) {
                    if (!is_valid_future(input_futures[i])) {
                        continue;
                    }
                    int done = 0;
                    auto test_status = dtl_future_test(input_futures[i], &done);
                    if (test_status == DTL_SUCCESS && done) {
                        dtl_size_t winner = i;
                        dtl_future_set(res, &winner, sizeof(winner));
                        return false;
                    }
                }
                return true;
            });
    } catch (...) {
        dtl_future_destroy(res);
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    *result = res;
    return DTL_SUCCESS;
}

}  // extern "C"
