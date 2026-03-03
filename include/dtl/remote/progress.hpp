// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file progress.hpp
/// @brief Remote operation progress model
/// @details Provides progress-based message processing for RPC operations.
///          Integrates with dtl::futures::progress_engine for unified progress.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/remote/rpc_request.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace dtl::remote {

// ============================================================================
// Remote Progress Manager
// ============================================================================

/// @brief Manages progress for remote operations
/// @details Processes incoming messages and drives pending operations.
///          Uses manual progress (explicit calls required).
class remote_progress_manager {
public:
    /// @brief Handler for incoming messages
    using message_handler = std::function<void(const message_header&,
                                               const std::byte* payload,
                                               size_type payload_size)>;

    /// @brief Get the singleton instance
    [[nodiscard]] static remote_progress_manager& instance() noexcept {
        static remote_progress_manager mgr;
        return mgr;
    }

    /// @brief Set the message handler for incoming requests
    void set_request_handler(message_handler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        request_handler_ = std::move(handler);
    }

    /// @brief Set the message handler for incoming responses
    void set_response_handler(message_handler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        response_handler_ = std::move(handler);
    }

    /// @brief Queue an incoming message for processing
    void enqueue_message(const std::byte* data, size_type size) {
        if (size < message_header::serialized_size()) return;

        std::vector<std::byte> msg(data, data + size);

        std::lock_guard<std::mutex> lock(mutex_);
        incoming_queue_.push(std::move(msg));
    }

    /// @brief Process pending messages (drive progress)
    /// @return Number of messages processed
    size_type process_messages() {
        std::queue<std::vector<std::byte>> to_process;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::swap(to_process, incoming_queue_);
        }

        size_type count = 0;
        while (!to_process.empty()) {
            auto& msg = to_process.front();
            process_one_message(msg.data(), msg.size());
            to_process.pop();
            ++count;
        }

        return count;
    }

    /// @brief Check if there are pending messages
    [[nodiscard]] bool has_pending() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return !incoming_queue_.empty();
    }

    /// @brief Get the pending request manager
    [[nodiscard]] pending_request_manager& pending_requests() noexcept {
        return pending_requests_;
    }

private:
    remote_progress_manager() {
        // Register with async progress engine
        futures::progress_engine::instance().register_callback([this]() {
            return process_messages() > 0 || has_pending();
        });
    }

    void process_one_message(const std::byte* data, size_type size) {
        auto header = message_header::deserialize(data);
        const std::byte* payload = data + message_header::serialized_size();
        size_type payload_size = size - message_header::serialized_size();

        message_handler handler;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (header.msg_type == message_header::request_type ||
                header.msg_type == message_header::fire_forget_type) {
                handler = request_handler_;
            } else {
                handler = response_handler_;
            }
        }

        if (handler) {
            handler(header, payload, payload_size);
        }
    }

    mutable std::mutex mutex_;
    std::queue<std::vector<std::byte>> incoming_queue_;
    message_handler request_handler_;
    message_handler response_handler_;
    pending_request_manager pending_requests_;
};

// ============================================================================
// Progress Functions
// ============================================================================

/// @brief Drive remote operation progress
/// @return Number of messages processed
inline size_type make_remote_progress() {
    return remote_progress_manager::instance().process_messages();
}

/// @brief Combined progress (async + remote)
inline void make_all_progress() {
    futures::make_progress();
    make_remote_progress();
}

}  // namespace dtl::remote
