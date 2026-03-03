// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file active_message.hpp
/// @brief Active message interface for low-level messaging
/// @details Provides fire-and-forget and request-reply active messages
///          with user-defined payload handling.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/error/result.hpp>
#include <dtl/remote/action.hpp>
#include <dtl/remote/rpc_request.hpp>
#include <dtl/remote/progress.hpp>

#include <cstring>
#include <functional>
#include <span>
#include <vector>

namespace dtl::remote {

// Import futures types from dtl::futures:: namespace
using dtl::futures::distributed_future;
using dtl::futures::distributed_promise;

// ============================================================================
// Active Message Handler Type
// ============================================================================

/// @brief Handler for incoming active messages
/// @details Called when an active message is received. Handlers should
///          process quickly and not block.
using am_handler = std::function<void(
    rank_t source,                      ///< Source rank
    const std::byte* payload,           ///< Message payload
    size_type payload_size              ///< Payload size
)>;

/// @brief Handler with reply capability
/// @details Can send a reply back to the source.
using am_handler_with_reply = std::function<std::vector<std::byte>(
    rank_t source,                      ///< Source rank
    const std::byte* payload,           ///< Message payload
    size_type payload_size              ///< Payload size
)>;

// ============================================================================
// Active Message Registry
// ============================================================================

/// @brief Registry for active message handlers
/// @details Maps action IDs to handlers for active message dispatch.
class am_registry {
public:
    /// @brief Get the singleton instance
    [[nodiscard]] static am_registry& instance() noexcept {
        static am_registry reg;
        return reg;
    }

    /// @brief Register a fire-and-forget handler
    /// @param id Action ID for this handler
    /// @param handler Handler function
    void register_handler(action_id id, am_handler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        handlers_[id] = std::move(handler);
    }

    /// @brief Register a handler with reply
    /// @param id Action ID for this handler
    /// @param handler Handler function that returns reply data
    void register_handler_with_reply(action_id id, am_handler_with_reply handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        reply_handlers_[id] = std::move(handler);
    }

    /// @brief Unregister a handler
    void unregister_handler(action_id id) {
        std::lock_guard<std::mutex> lock(mutex_);
        handlers_.erase(id);
        reply_handlers_.erase(id);
    }

    /// @brief Dispatch to a handler
    /// @return true if handler was found and invoked
    bool dispatch(action_id id, rank_t source,
                  const std::byte* payload, size_type size) {
        am_handler handler;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = handlers_.find(id);
            if (it != handlers_.end()) {
                handler = it->second;
            }
        }

        if (handler) {
            handler(source, payload, size);
            return true;
        }
        return false;
    }

    /// @brief Dispatch to a handler with reply
    /// @return Reply data if handler found, nullopt otherwise
    std::optional<std::vector<std::byte>> dispatch_with_reply(
        action_id id, rank_t source,
        const std::byte* payload, size_type size) {

        am_handler_with_reply handler;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = reply_handlers_.find(id);
            if (it != reply_handlers_.end()) {
                handler = it->second;
            }
        }

        if (handler) {
            return handler(source, payload, size);
        }
        return std::nullopt;
    }

private:
    am_registry() = default;

    std::mutex mutex_;
    std::unordered_map<action_id, am_handler> handlers_;
    std::unordered_map<action_id, am_handler_with_reply> reply_handlers_;
};

// ============================================================================
// Active Message Context
// ============================================================================

/// @brief Context for sending active messages
class am_context {
public:
    /// @brief Send function type
    using send_fn = std::function<result<void>(
        rank_t target,
        const std::byte* data,
        size_type size,
        int tag)>;

    /// @brief Default constructor
    am_context() : my_rank_(no_rank) {}

    /// @brief Construct with rank and sender
    am_context(rank_t my_rank, send_fn sender)
        : my_rank_(my_rank), sender_(std::move(sender)) {}

    /// @brief Get local rank
    [[nodiscard]] rank_t rank() const noexcept { return my_rank_; }

    /// @brief Set local rank
    void set_rank(rank_t r) noexcept { my_rank_ = r; }

    /// @brief Set sender
    void set_sender(send_fn fn) { sender_ = std::move(fn); }

    /// @brief Get pending request manager
    [[nodiscard]] pending_request_manager& pending_requests() {
        return remote_progress_manager::instance().pending_requests();
    }

    /// @brief Send raw data
    result<void> send_raw(rank_t target, const std::byte* data,
                          size_type size, int tag = 0) {
        if (!sender_) {
            return status(status_code::not_implemented, "No sender configured");
        }
        return sender_(target, data, size, tag);
    }

    /// @brief Get default context
    [[nodiscard]] static am_context& default_context() {
        static am_context ctx;
        return ctx;
    }

private:
    rank_t my_rank_;
    send_fn sender_;
};

// ============================================================================
// Active Message Send Functions
// ============================================================================

/// @brief Send a fire-and-forget active message
/// @param ctx AM context
/// @param target Target rank
/// @param handler_id Handler action ID
/// @param payload Message payload
/// @return Status of send operation
inline result<void> send_am(am_context& ctx, rank_t target, action_id handler_id,
                           std::span<const std::byte> payload) {
    // Build message with header
    message_buffer msg;
    msg.reserve(payload.size());
    msg.set_header(message_header::make_fire_forget(
        handler_id, payload.size(), ctx.rank()));

    if (!payload.empty()) {
        std::memcpy(msg.payload_data(), payload.data(), payload.size());
    }

    return ctx.send_raw(target, msg.data(), msg.size());
}

/// @brief Send AM with default context
inline result<void> send_am(rank_t target, action_id handler_id,
                           std::span<const std::byte> payload) {
    return send_am(am_context::default_context(), target, handler_id, payload);
}

/// @brief Send an active message expecting a reply
/// @param ctx AM context
/// @param target Target rank
/// @param handler_id Handler action ID
/// @param payload Message payload
/// @return Future for reply data
inline distributed_future<std::vector<std::byte>> send_am_with_reply(
    am_context& ctx, rank_t target, action_id handler_id,
    std::span<const std::byte> payload) {

    // Create promise/future
    distributed_promise<std::vector<std::byte>> promise;
    auto future = promise.get_future();

    // Register pending request
    auto request_id = ctx.pending_requests().register_request<std::vector<std::byte>>(
        target, std::move(promise));

    // Build message
    message_buffer msg;
    msg.reserve(payload.size());
    msg.set_header(message_header::make_request(
        handler_id, request_id, payload.size(), ctx.rank()));

    if (!payload.empty()) {
        std::memcpy(msg.payload_data(), payload.data(), payload.size());
    }

    // Send
    auto send_result = ctx.send_raw(target, msg.data(), msg.size());
    if (!send_result) {
        ctx.pending_requests().fail(request_id, send_result.error());
    }

    return future;
}

/// @brief Send AM with reply (default context)
inline distributed_future<std::vector<std::byte>> send_am_with_reply(
    rank_t target, action_id handler_id,
    std::span<const std::byte> payload) {
    return send_am_with_reply(am_context::default_context(), target,
                              handler_id, payload);
}

// ============================================================================
// Handler Registration Helpers
// ============================================================================

/// @brief Register an active message handler for an action
/// @tparam Func Function to handle the message
/// @param handler Handler function
template <auto Func>
void register_am_handler(am_handler handler) {
    am_registry::instance().register_handler(action<Func>::id(), std::move(handler));
}

/// @brief Register an AM handler with reply
template <auto Func>
void register_am_handler_with_reply(am_handler_with_reply handler) {
    am_registry::instance().register_handler_with_reply(
        action<Func>::id(), std::move(handler));
}

/// @brief Unregister an AM handler
template <auto Func>
void unregister_am_handler() {
    am_registry::instance().unregister_handler(action<Func>::id());
}

}  // namespace dtl::remote
