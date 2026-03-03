// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file executor.hpp
/// @brief Executor concept for computation dispatch
/// @details Defines requirements for executing work items.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <functional>

namespace dtl {

// ============================================================================
// Executor Concept
// ============================================================================

/// @brief Core executor concept for synchronous execution
/// @details Defines minimum requirements for an executor.
///
/// @par Required Operations:
/// - execute(): Execute a callable synchronously
/// - name(): Get executor name
template <typename T>
concept Executor = requires(T& exec, const T& cexec, std::function<void()> f) {
    // Execute callable
    { exec.execute(f) } -> std::same_as<void>;

    // Identity
    { cexec.name() } -> std::convertible_to<const char*>;
};

// ============================================================================
// Sync Executor Concept
// ============================================================================

/// @brief Executor that completes work before returning
template <typename T>
concept SyncExecutor = Executor<T> &&
    requires(T& exec) {
    // Guarantee synchronous completion
    { exec.is_synchronous() } -> std::same_as<bool>;
};

// ============================================================================
// Parallel Executor Concept
// ============================================================================

/// @brief Executor supporting parallel execution
/// @details Can execute multiple work items concurrently.
template <typename T>
concept ParallelExecutor = Executor<T> &&
    requires(T& exec, const T& cexec,
             size_type count, std::function<void(size_type)> f) {
    // Parallel for over range
    { exec.parallel_for(count, f) } -> std::same_as<void>;

    // Query parallelism
    { cexec.max_parallelism() } -> std::same_as<size_type>;
    { cexec.suggested_parallelism() } -> std::same_as<size_type>;
};

// ============================================================================
// Bulk Executor Concept
// ============================================================================

/// @brief Executor optimized for bulk operations
template <typename T>
concept BulkExecutor = ParallelExecutor<T> &&
    requires(T& exec, size_type count, std::function<void(size_type, size_type)> f) {
    // Bulk execution with chunk boundaries
    { exec.bulk_execute(count, f) } -> std::same_as<void>;
};

// ============================================================================
// Executor Properties
// ============================================================================

/// @brief Properties describing an executor
struct executor_properties {
    /// @brief Maximum concurrent work items
    size_type max_concurrency = 1;

    /// @brief Whether execution is in-order
    bool in_order = true;

    /// @brief Whether executor owns threads
    bool owns_threads = false;

    /// @brief Whether work stealing is supported
    bool supports_work_stealing = false;
};

// ============================================================================
// Executor Tag Types
// ============================================================================

/// @brief Tag for inline (immediate) executor
struct inline_executor_tag {};

/// @brief Tag for thread pool executor
struct thread_pool_executor_tag {};

/// @brief Tag for single-threaded executor
struct single_thread_executor_tag {};

/// @brief Tag for GPU executor
struct gpu_executor_tag {};

// ============================================================================
// Executor Traits
// ============================================================================

/// @brief Traits for executor types
template <typename Exec>
struct executor_traits {
    /// @brief Whether executor is synchronous
    static constexpr bool is_sync = false;

    /// @brief Whether executor supports parallelism
    static constexpr bool is_parallel = false;

    /// @brief Whether executor runs on GPU
    static constexpr bool is_gpu = false;

    /// @brief Default chunk size for bulk operations
    static constexpr size_type default_chunk_size = 1;
};

// ============================================================================
// Standard Executors
// ============================================================================

/// @brief Inline executor that runs immediately
/// @details Executes work items in the calling thread.
class inline_executor {
public:
    using tag_type = inline_executor_tag;

    /// @brief Execute callable immediately
    template <typename F>
    void execute(F&& f) {
        std::forward<F>(f)();
    }

    /// @brief Get executor name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "inline";
    }

    /// @brief Executor is synchronous
    [[nodiscard]] static constexpr bool is_synchronous() noexcept {
        return true;
    }
};

/// @brief Sequential executor for ordered execution
class sequential_executor {
public:
    using tag_type = single_thread_executor_tag;

    /// @brief Execute callable
    template <typename F>
    void execute(F&& f) {
        std::forward<F>(f)();
    }

    /// @brief Parallel for (sequential implementation)
    template <typename F>
    void parallel_for(size_type count, F&& f) {
        for (size_type i = 0; i < count; ++i) {
            f(i);
        }
    }

    /// @brief Get executor name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "sequential";
    }

    /// @brief Maximum parallelism (always 1)
    [[nodiscard]] static constexpr size_type max_parallelism() noexcept {
        return 1;
    }

    /// @brief Suggested parallelism
    [[nodiscard]] static constexpr size_type suggested_parallelism() noexcept {
        return 1;
    }
};

}  // namespace dtl
