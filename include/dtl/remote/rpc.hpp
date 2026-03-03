// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rpc.hpp
/// @brief RPC interfaces for remote procedure calls
/// @details Provides type-safe RPC with compile-time action binding and
///          automatic argument serialization.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/error/result.hpp>
#include <dtl/remote/action.hpp>
#include <dtl/remote/argument_pack.hpp>
#include <dtl/remote/rpc_request.hpp>
#include <dtl/remote/rpc_serialization.hpp>
#include <dtl/remote/progress.hpp>

#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtl::remote {

// Import futures types from dtl::futures:: namespace
using dtl::futures::distributed_future;
using dtl::futures::distributed_promise;

// Forward declarations
class rpc_context;

// ============================================================================
// RPC Context
// ============================================================================

/// @brief Context for RPC operations
/// @details Holds configuration and state for making RPC calls.
///          In a full implementation, this would be bound to a communicator.
class rpc_context {
public:
    /// @brief Message send function type
    using send_fn = std::function<result<void>(
        rank_t target,
        const std::byte* data,
        size_type size,
        int tag)>;

    /// @brief Default constructor (creates default context)
    rpc_context() : my_rank_(no_rank) {}

    /// @brief Construct with rank and send function
    rpc_context(rank_t my_rank, send_fn sender)
        : my_rank_(my_rank), sender_(std::move(sender)) {}

    /// @brief Get local rank
    [[nodiscard]] rank_t rank() const noexcept { return my_rank_; }

    /// @brief Set local rank
    void set_rank(rank_t r) noexcept { my_rank_ = r; }

    /// @brief Set send function
    void set_sender(send_fn fn) { sender_ = std::move(fn); }

    /// @brief Get pending request manager
    [[nodiscard]] pending_request_manager& pending_requests() {
        return remote_progress_manager::instance().pending_requests();
    }

    /// @brief Send a message to a target rank
    result<void> send(rank_t target, const std::byte* data,
                      size_type size, int tag = 0) {
        if (!sender_) {
            return status(status_code::not_implemented, "No sender configured");
        }
        return sender_(target, data, size, tag);
    }

    /// @brief Get the global default context
    [[nodiscard]] static rpc_context& default_context() {
        static rpc_context ctx;
        return ctx;
    }

private:
    rank_t my_rank_;
    send_fn sender_;
};

// ============================================================================
// Synchronous RPC (Blocking)
// ============================================================================

/// @brief Make a synchronous (blocking) RPC call
/// @tparam Func Function pointer for the action
/// @tparam Args Argument types
/// @param ctx RPC context
/// @param target Target rank
/// @param args Arguments to pass to the remote function
/// @return Result of the remote function call
template <auto Func, typename... Args>
auto call_sync(rpc_context& ctx, rank_t target, Args&&... args)
    -> result<typename action<Func>::response_type> {

    using Action = action<Func>;
    using response_t = typename Action::response_type;

    // Make async call
    auto future = call<Func>(ctx, target, std::forward<Args>(args)...);

    // Block waiting for result
    while (!future.is_ready()) {
        make_all_progress();
        std::this_thread::yield();
    }

    return future.get_result();
}

/// @brief Make a synchronous RPC call (default context)
template <auto Func, typename... Args>
auto call_sync(rank_t target, Args&&... args)
    -> result<typename action<Func>::response_type> {
    return call_sync<Func>(rpc_context::default_context(), target,
                          std::forward<Args>(args)...);
}

// ============================================================================
// Asynchronous RPC (Non-blocking)
// ============================================================================

/// @brief Make an asynchronous (non-blocking) RPC call
/// @tparam Func Function pointer for the action
/// @tparam Args Argument types
/// @param ctx RPC context
/// @param target Target rank
/// @param args Arguments to pass to the remote function
/// @return Future for the result
template <auto Func, typename... Args>
auto call(rpc_context& ctx, rank_t target, Args&&... args)
    -> distributed_future<typename action<Func>::response_type> {

    using Action = action<Func>;
    using response_t = typename Action::response_type;
    using args_tuple = typename Action::request_type;

    // Create promise and future
    distributed_promise<response_t> promise;
    auto future = promise.get_future();

    // Register pending request
    auto request_id = ctx.pending_requests().register_request<response_t>(
        target, std::move(promise));

    // Serialize arguments
    using pack_t = argument_pack_for<args_tuple>;
    auto payload = pack_t::serialize_to_vector(std::forward<Args>(args)...);

    // Build message
    message_buffer msg;
    msg.reserve(payload.size());
    msg.set_header(message_header::make_request(
        Action::id(), request_id, payload.size(), ctx.rank()));
    if (!payload.empty()) {
        std::memcpy(msg.payload_data(), payload.data(), payload.size());
    }

    // Send message
    auto send_result = ctx.send(target, msg.data(), msg.size());
    if (!send_result) {
        ctx.pending_requests().fail(request_id, send_result.error());
    }

    return future;
}

/// @brief Make an asynchronous RPC call (default context)
template <auto Func, typename... Args>
auto call(rank_t target, Args&&... args)
    -> distributed_future<typename action<Func>::response_type> {
    return call<Func>(rpc_context::default_context(), target,
                     std::forward<Args>(args)...);
}

// ============================================================================
// Fire-and-Forget RPC
// ============================================================================

/// @brief Send a fire-and-forget RPC (no response expected)
/// @tparam Func Function pointer for the action (must return void)
/// @tparam Args Argument types
/// @param ctx RPC context
/// @param target Target rank
/// @param args Arguments to pass to the remote function
/// @return Status of the send operation
template <auto Func, typename... Args>
    requires std::is_void_v<typename action<Func>::response_type>
result<void> send(rpc_context& ctx, rank_t target, Args&&... args) {
    using Action = action<Func>;
    using args_tuple = typename Action::request_type;

    // Serialize arguments
    using pack_t = argument_pack_for<args_tuple>;
    auto payload = pack_t::serialize_to_vector(std::forward<Args>(args)...);

    // Build message
    message_buffer msg;
    msg.reserve(payload.size());
    msg.set_header(message_header::make_fire_forget(
        Action::id(), payload.size(), ctx.rank()));
    if (!payload.empty()) {
        std::memcpy(msg.payload_data(), payload.data(), payload.size());
    }

    // Send message
    return ctx.send(target, msg.data(), msg.size());
}

/// @brief Fire-and-forget RPC (default context)
template <auto Func, typename... Args>
    requires std::is_void_v<typename action<Func>::response_type>
result<void> send(rank_t target, Args&&... args) {
    return send<Func>(rpc_context::default_context(), target,
                     std::forward<Args>(args)...);
}

// ============================================================================
// Multi-Target RPC
// ============================================================================

/// @brief Make RPC calls to multiple targets
/// @tparam Func Function pointer for the action
/// @tparam RankRange Range of ranks
/// @tparam Args Argument types
/// @param ctx RPC context
/// @param targets Range of target ranks
/// @param args Arguments (same for all targets)
/// @return Vector of futures for results
template <auto Func, typename RankRange, typename... Args>
auto call_multi(rpc_context& ctx, RankRange&& targets, Args&&... args)
    -> std::vector<distributed_future<typename action<Func>::response_type>> {

    std::vector<distributed_future<typename action<Func>::response_type>> futures;

    for (rank_t target : targets) {
        futures.push_back(call<Func>(ctx, target, args...));
    }

    return futures;
}

/// @brief Multi-target RPC (default context)
template <auto Func, typename RankRange, typename... Args>
auto call_multi(RankRange&& targets, Args&&... args)
    -> std::vector<distributed_future<typename action<Func>::response_type>> {
    return call_multi<Func>(rpc_context::default_context(),
                           std::forward<RankRange>(targets),
                           std::forward<Args>(args)...);
}

// ============================================================================
// Action Dispatcher
// ============================================================================

/// @brief Dispatch table for handling incoming RPC requests
/// @tparam Actions List of action types to handle
template <typename... Actions>
class action_dispatcher {
public:
    /// @brief Set the RPC context used for sending responses
    void set_context(rpc_context& ctx) {
        ctx_ = &ctx;
    }

    /// @brief Register this dispatcher with the progress manager
    void register_handlers() {
        remote_progress_manager::instance().set_request_handler(
            [this](const message_header& header, const std::byte* payload,
                   size_type payload_size) {
                handle_request(header, payload, payload_size);
            });

        remote_progress_manager::instance().set_response_handler(
            [](const message_header& header, const std::byte* payload,
               size_type payload_size) {
                // Route to pending requests manager
                auto& pending = remote_progress_manager::instance().pending_requests();
                if (header.msg_type == message_header::response_type) {
                    pending.complete(header.request, payload, payload_size);
                } else if (header.msg_type == message_header::error_type) {
                    pending.fail(header.request,
                        status(status_code::operation_failed, "Remote error"));
                }
            });
    }

private:
    void handle_request(const message_header& header, const std::byte* payload,
                       size_type payload_size) {
        // Try to dispatch to each registered action
        bool handled = (try_dispatch<Actions>(header, payload, payload_size) || ...);

        if (!handled) {
            // Unknown action - send error response if we have a context
            if (ctx_ && header.msg_type == message_header::request_type) {
                message_buffer msg;
                msg.reserve(0);
                msg.set_header(message_header::make_error(
                    header.request, 0, ctx_->rank()));
                ctx_->send(header.source_rank, msg.data(), msg.size());
            }
        }
    }

    template <typename Action>
    bool try_dispatch(const message_header& header, const std::byte* payload,
                     size_type payload_size) {
        if (header.action != Action::id()) {
            return false;
        }

        // Deserialize arguments
        using args_tuple = typename Action::request_type;
        using response_t = typename Action::response_type;
        using pack_t = argument_pack_for<args_tuple>;

        auto args = pack_t::deserialize(payload, payload_size);

        // Invoke the action and handle result
        if constexpr (Action::is_void) {
            Action::invoke_tuple(args);
            // For request-type messages, send empty response acknowledgment
            if (ctx_ && header.msg_type == message_header::request_type) {
                message_buffer msg;
                msg.reserve(0);
                msg.set_header(message_header::make_response(
                    header.request, 0, ctx_->rank()));
                ctx_->send(header.source_rank, msg.data(), msg.size());
            }
        } else {
            auto result = Action::invoke_tuple(args);
            // Serialize and send the result back to the caller
            if (ctx_ && header.msg_type == message_header::request_type) {
                size_type result_size = serialized_size(result);
                message_buffer msg;
                msg.reserve(result_size);
                if (result_size > 0) {
                    serialize(result, msg.payload_data());
                }
                msg.set_header(message_header::make_response(
                    header.request, result_size, ctx_->rank()));
                ctx_->send(header.source_rank, msg.data(), msg.size());
            }
        }

        return true;
    }

    rpc_context* ctx_ = nullptr;
};

}  // namespace dtl::remote
