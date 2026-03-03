// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file failure_handler.hpp
/// @brief Failure detection and recovery hooks
/// @details Provides extensible failure handling for distributed operations,
///          including callback registration and recovery strategies.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/status.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dtl {

/// @brief Categories of failures for handling purposes
enum class failure_category {
    recoverable,      ///< Failure can potentially be recovered from
    non_recoverable,  ///< Failure is fatal and cannot be recovered
    transient,        ///< Temporary failure that may resolve itself
    resource,         ///< Resource exhaustion (memory, connections)
    communication,    ///< Network/communication failure
    corruption        ///< Data corruption detected
};

/// @brief Get failure category for a status code
/// @param code The status code to categorize
/// @return The failure category
[[nodiscard]] inline failure_category categorize_failure(status_code code) noexcept {
    const int val = static_cast<int>(code);

    if (val >= 100 && val < 200) {
        // Communication errors - often transient
        if (code == status_code::timeout || code == status_code::connection_lost) {
            return failure_category::transient;
        }
        return failure_category::communication;
    }

    if (val >= 200 && val < 300) {
        // Memory errors - resource issues
        return failure_category::resource;
    }

    if (val >= 300 && val < 400) {
        // Serialization errors - potential corruption
        if (code == status_code::invalid_format) {
            return failure_category::corruption;
        }
        return failure_category::non_recoverable;
    }

    // Default to non-recoverable
    return failure_category::non_recoverable;
}

/// @brief Context information passed to failure handlers
struct failure_context {
    status failure_status;           ///< The status that triggered the failure
    failure_category category;       ///< Category of the failure
    rank_t local_rank;               ///< Local rank where failure detected
    rank_t failed_rank;              ///< Remote rank that failed (if applicable)
    std::string operation;           ///< Name of the operation that failed
    bool is_collective;              ///< Whether this was a collective operation
};

/// @brief Possible recovery actions
enum class recovery_action {
    none,          ///< No recovery attempted
    retry,         ///< Retry the operation
    skip,          ///< Skip this operation and continue
    abort,         ///< Abort the entire computation
    checkpoint,    ///< Rollback to checkpoint and retry
    redistribute   ///< Redistribute work around failed ranks
};

/// @brief Result of a failure handler
struct failure_result {
    recovery_action action;          ///< Action to take
    int retry_count;                 ///< Number of retries attempted/remaining
    std::string message;             ///< Optional message about the recovery
};

/// @brief Type for failure handler callbacks
using failure_handler_fn = std::function<failure_result(const failure_context&)>;

/// @brief Manager for failure handlers
/// @details Allows registration of custom failure handlers that are called
///          when errors occur. Handlers are called in order until one
///          provides a recovery action.
class failure_handler_manager {
public:
    /// @brief Get the singleton instance
    [[nodiscard]] static failure_handler_manager& instance() {
        static failure_handler_manager mgr;
        return mgr;
    }

    /// @brief Register a failure handler
    /// @param handler The handler function to register
    /// @return Handle for unregistering the handler
    size_type register_handler(failure_handler_fn handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        handlers_.push_back(std::move(handler));
        return handlers_.size() - 1;
    }

    /// @brief Unregister a failure handler
    /// @param handle The handle returned by register_handler
    void unregister_handler(size_type handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (handle < handlers_.size()) {
            handlers_[handle] = nullptr;
        }
    }

    /// @brief Handle a failure
    /// @param ctx The failure context
    /// @return The result from the first handler that provides an action
    [[nodiscard]] failure_result handle_failure(const failure_context& ctx) {
        // Copy handlers under the lock to avoid holding it during callback
        std::vector<failure_handler_fn> handlers_copy;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            handlers_copy = handlers_;
        }
        for (const auto& handler : handlers_copy) {
            if (handler) {
                auto result = handler(ctx);
                if (result.action != recovery_action::none) {
                    return result;
                }
            }
        }
        // Default: abort on non-recoverable, retry on transient
        if (ctx.category == failure_category::transient) {
            return failure_result{recovery_action::retry, 3, "default retry"};
        }
        return failure_result{recovery_action::abort, 0, "no handler"};
    }

    /// @brief Clear all registered handlers
    void clear_handlers() {
        std::lock_guard<std::mutex> lock(mutex_);
        handlers_.clear();
    }

private:
    failure_handler_manager() = default;
    mutable std::mutex mutex_;
    std::vector<failure_handler_fn> handlers_;
};

/// @brief RAII guard for registering a scoped failure handler
class scoped_failure_handler {
public:
    /// @brief Register a handler for this scope
    /// @param handler The handler function
    explicit scoped_failure_handler(failure_handler_fn handler)
        : handle_{failure_handler_manager::instance().register_handler(
              std::move(handler))} {}

    /// @brief Unregister the handler
    ~scoped_failure_handler() {
        failure_handler_manager::instance().unregister_handler(handle_);
    }

    // Non-copyable, non-movable
    scoped_failure_handler(const scoped_failure_handler&) = delete;
    scoped_failure_handler& operator=(const scoped_failure_handler&) = delete;
    scoped_failure_handler(scoped_failure_handler&&) = delete;
    scoped_failure_handler& operator=(scoped_failure_handler&&) = delete;

private:
    size_type handle_;
};

}  // namespace dtl
