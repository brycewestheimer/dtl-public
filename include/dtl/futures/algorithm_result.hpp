// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file algorithm_result.hpp
/// @brief Return type adaptation based on execution policy
/// @details Provides unified return types for sync and async algorithms.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/algorithms/dispatch.hpp>

namespace dtl::futures {

// ============================================================================
// Execution Policy Tags
// ============================================================================

// Forward declarations / imports for execution policy types
// seq and par are defined in dtl:: (policies/execution/); import them
using dtl::seq;
using dtl::par;

// async_policy is defined here as the algorithm-level async tag
// (distinct from dtl::async which is the execution policy struct)
struct async_policy;

// ============================================================================
// Algorithm Result Traits
// ============================================================================

/// @brief Determine algorithm return type based on execution policy
/// @tparam Policy Execution policy type
/// @tparam T Result value type
template <typename Policy, typename T>
struct algorithm_result {
    /// @brief Return type for the algorithm
    using type = result<T>;
};

/// @brief Specialization for async policy
template <typename T>
struct algorithm_result<async_policy, T> {
    using type = distributed_future<T>;
};

/// @brief Helper alias
template <typename Policy, typename T>
using algorithm_result_t = typename algorithm_result<Policy, T>::type;

// ============================================================================
// Void Specializations
// ============================================================================

/// @brief Result type for void-returning algorithms
template <typename Policy>
struct algorithm_result<Policy, void> {
    using type = result<void>;
};

/// @brief Async void result
template <>
struct algorithm_result<async_policy, void> {
    using type = distributed_future<void>;
};

// ============================================================================
// Result Conversion
// ============================================================================

/// @brief Convert a value to algorithm result
/// @tparam Policy Execution policy type
/// @tparam T Value type
/// @param value The value
/// @return Algorithm result
template <typename Policy, typename T>
[[nodiscard]] auto make_algorithm_result(T&& value) {
    if constexpr (std::is_same_v<Policy, async_policy>) {
        return make_ready_distributed_future(std::forward<T>(value));
    } else {
        return result<std::decay_t<T>>(std::forward<T>(value));
    }
}

/// @brief Create void algorithm result
/// @tparam Policy Execution policy type
/// @return Void algorithm result
template <typename Policy>
[[nodiscard]] auto make_algorithm_result_void() {
    if constexpr (std::is_same_v<Policy, async_policy>) {
        return make_ready_distributed_future();
    } else {
        return result<void>();
    }
}

/// @brief Create error algorithm result
/// @tparam Policy Execution policy type
/// @tparam T Expected value type
/// @param error The error
/// @return Error result
template <typename Policy, typename T>
[[nodiscard]] auto make_algorithm_error(status error) {
    if constexpr (std::is_same_v<Policy, async_policy>) {
        return make_failed_distributed_future<T>(std::move(error));
    } else {
        return make_error<T>(error.code(), error.message());
    }
}

// ============================================================================
// Algorithm Result Wrapper
// ============================================================================

/// @brief Wrapper that holds either sync or async result
/// @tparam T Value type
template <typename T>
class unified_result {
public:
    /// @brief Construct with sync result
    explicit unified_result(result<T> sync_result)
        : is_async_(false)
        , sync_result_(std::move(sync_result)) {}

    /// @brief Construct with async result
    explicit unified_result(distributed_future<T> async_result)
        : is_async_(true)
        , async_result_(std::move(async_result)) {}

    /// @brief Check if result is async
    [[nodiscard]] bool is_async() const noexcept {
        return is_async_;
    }

    /// @brief Get sync result (blocks if async)
    [[nodiscard]] result<T> get() {
        if (is_async_) {
            return async_result_.get_result();
        }
        return std::move(sync_result_);
    }

    /// @brief Get future (creates ready future if sync)
    [[nodiscard]] distributed_future<T> get_future() {
        if (is_async_) {
            return std::move(async_result_);
        }
        if (sync_result_.has_value()) {
            return make_ready_distributed_future(std::move(sync_result_.value()));
        }
        return make_failed_distributed_future<T>(sync_result_.error());
    }

    /// @brief Wait for result to be ready
    void wait() {
        if (is_async_) {
            async_result_.wait();
        }
    }

    /// @brief Check if ready
    [[nodiscard]] bool is_ready() const noexcept {
        if (is_async_) {
            return async_result_.is_ready();
        }
        return true;  // Sync results are always ready
    }

private:
    bool is_async_;
    result<T> sync_result_;
    distributed_future<T> async_result_;
};

// ============================================================================
// Policy Detection
// ============================================================================

// Note: is_seq_policy_v, is_par_policy_v, is_async_policy_v are defined in
// algorithms/dispatch.hpp - include that header for these traits

/// @brief Check if policy is synchronous (seq or par)
template <typename Policy>
inline constexpr bool is_sync_policy_v =
    std::is_same_v<std::decay_t<Policy>, seq> || std::is_same_v<std::decay_t<Policy>, par>;

// Note: is_async_policy_v is defined in algorithms/dispatch.hpp

// ============================================================================
// Execution Helper
// ============================================================================

/// @brief Execute algorithm with appropriate return type
/// @tparam Policy Execution policy
/// @tparam F Function type
/// @param func Algorithm function returning T
/// @return Algorithm result
template <typename Policy, typename F>
auto execute_algorithm(F&& func) {
    using T = std::invoke_result_t<F>;

    if constexpr (is_async_policy_v<Policy>) {
        // Launch asynchronously via progress engine
        auto state = std::make_shared<shared_state<T>>();
        auto started = std::make_shared<std::atomic<bool>>(false);

        futures::progress_engine::instance().register_callback(
            [state, started, f = std::forward<F>(func)]() mutable {
                // Execute on first poll
                bool expected = false;
                if (started->compare_exchange_strong(expected, true)) {
                    try {
                        if constexpr (std::is_void_v<T>) {
                            f();
                            state->set_value();
                        } else {
                            state->set_value(f());
                        }
                    } catch (const std::exception& e) {
                        state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                    }
                }
                return false;  // Done after first execution
            });

        return distributed_future<T>(std::move(state));
    } else {
        // Execute synchronously
        try {
            if constexpr (std::is_void_v<T>) {
                func();
                return result<void>();
            } else {
                return result<T>(func());
            }
        } catch (const std::exception& e) {
            return make_error<T>(status_code::operation_failed, e.what());
        }
    }
}

}  // namespace dtl::futures
