// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rpc_request.hpp
/// @brief RPC request and response management
/// @details Provides request tracking, correlation, and pending request management.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/remote/action.hpp>
#include <dtl/futures/distributed_future.hpp>

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace dtl::remote {

// Import futures types from dtl::futures:: namespace
using dtl::futures::distributed_future;
using dtl::futures::distributed_promise;

// ============================================================================
// Request ID
// ============================================================================

/// @brief Unique identifier for an RPC request
using request_id = std::uint64_t;

/// @brief Invalid request ID sentinel
inline constexpr request_id invalid_request_id = 0;

// ============================================================================
// Message Header
// ============================================================================

/// @brief Message header for RPC protocol
/// @details Prefixed to all RPC messages for routing and correlation.
struct message_header {
    action_id action;           ///< Action to invoke
    request_id request;         ///< Request ID for correlation
    size_type payload_size;     ///< Size of payload following header
    rank_t source_rank;         ///< Originating rank
    std::uint8_t msg_type;      ///< Message type (request/response/etc)
    std::uint8_t reserved[3];   ///< Reserved for alignment/future use

    /// @brief Message types
    enum type : std::uint8_t {
        request_type = 1,       ///< RPC request
        response_type = 2,      ///< RPC response
        error_type = 3,         ///< Error response
        fire_forget_type = 4    ///< Fire-and-forget (no response expected)
    };

    /// @brief Default constructor
    message_header() noexcept
        : action(invalid_action_id)
        , request(invalid_request_id)
        , payload_size(0)
        , source_rank(no_rank)
        , msg_type(0)
        , reserved{} {}

    /// @brief Construct for a request
    static message_header make_request(action_id act, request_id req,
                                       size_type size, rank_t src) noexcept {
        message_header h;
        h.action = act;
        h.request = req;
        h.payload_size = size;
        h.source_rank = src;
        h.msg_type = request_type;
        return h;
    }

    /// @brief Construct for a response
    static message_header make_response(request_id req, size_type size,
                                        rank_t src) noexcept {
        message_header h;
        h.request = req;
        h.payload_size = size;
        h.source_rank = src;
        h.msg_type = response_type;
        return h;
    }

    /// @brief Construct for an error
    static message_header make_error(request_id req, size_type size,
                                     rank_t src) noexcept {
        message_header h;
        h.request = req;
        h.payload_size = size;
        h.source_rank = src;
        h.msg_type = error_type;
        return h;
    }

    /// @brief Construct for fire-and-forget
    static message_header make_fire_forget(action_id act, size_type size,
                                           rank_t src) noexcept {
        message_header h;
        h.action = act;
        h.payload_size = size;
        h.source_rank = src;
        h.msg_type = fire_forget_type;
        return h;
    }

    /// @brief Serialize header to buffer
    static size_type serialize(const message_header& h, std::byte* buffer) noexcept {
        std::memcpy(buffer, &h, sizeof(message_header));
        return sizeof(message_header);
    }

    /// @brief Deserialize header from buffer
    static message_header deserialize(const std::byte* buffer) noexcept {
        message_header h;
        std::memcpy(&h, buffer, sizeof(message_header));
        return h;
    }

    /// @brief Size of serialized header
    static constexpr size_type serialized_size() noexcept {
        return sizeof(message_header);
    }
};

// ============================================================================
// Pending Request
// ============================================================================

namespace detail {

/// @brief Type-erased pending request
class pending_request_base {
public:
    virtual ~pending_request_base() = default;

    /// @brief Set the response data
    virtual void set_response(const std::byte* data, size_type size) = 0;

    /// @brief Set an error
    virtual void set_error(status error) = 0;

    /// @brief Get the request ID
    [[nodiscard]] request_id id() const noexcept { return id_; }

    /// @brief Get the target rank
    [[nodiscard]] rank_t target() const noexcept { return target_; }

protected:
    pending_request_base(request_id id, rank_t target)
        : id_(id), target_(target) {}

    request_id id_;
    rank_t target_;
};

/// @brief Typed pending request
template <typename T>
class pending_request : public pending_request_base {
public:
    pending_request(request_id id, rank_t target,
                   distributed_promise<T> promise)
        : pending_request_base(id, target)
        , promise_(std::move(promise)) {}

    void set_response(const std::byte* data, size_type size) override {
        try {
            if constexpr (std::is_void_v<T>) {
                (void)data;
                (void)size;
                promise_.set_value();
            } else {
                T value = dtl::deserialize<T>(data, size);
                promise_.set_value(std::move(value));
            }
        } catch (const std::exception& e) {
            promise_.set_error(status(status_code::serialization_error, e.what()));
        }
    }

    void set_error(status error) override {
        promise_.set_error(std::move(error));
    }

private:
    distributed_promise<T> promise_;
};

}  // namespace detail

// ============================================================================
// Pending Request Manager
// ============================================================================

/// @brief Manages pending RPC requests and correlates responses
class pending_request_manager {
public:
    /// @brief Default constructor
    pending_request_manager() : next_id_(1) {}

    /// @brief Register a new pending request
    /// @tparam T Response type
    /// @param target Target rank
    /// @param promise Promise to fulfill
    /// @return Request ID
    template <typename T>
    request_id register_request(rank_t target, distributed_promise<T> promise) {
        request_id id = next_id_.fetch_add(1, std::memory_order_relaxed);

        auto pending = std::make_unique<detail::pending_request<T>>(
            id, target, std::move(promise));

        std::lock_guard<std::mutex> lock(mutex_);
        pending_[id] = std::move(pending);

        return id;
    }

    /// @brief Complete a pending request with response data
    /// @param id Request ID
    /// @param data Response data
    /// @param size Response size
    /// @return true if request was found and completed
    bool complete(request_id id, const std::byte* data, size_type size) {
        std::unique_ptr<detail::pending_request_base> pending;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pending_.find(id);
            if (it == pending_.end()) {
                return false;
            }
            pending = std::move(it->second);
            pending_.erase(it);
        }

        pending->set_response(data, size);
        return true;
    }

    /// @brief Fail a pending request with an error
    /// @param id Request ID
    /// @param error Error status
    /// @return true if request was found and failed
    bool fail(request_id id, status error) {
        std::unique_ptr<detail::pending_request_base> pending;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pending_.find(id);
            if (it == pending_.end()) {
                return false;
            }
            pending = std::move(it->second);
            pending_.erase(it);
        }

        pending->set_error(std::move(error));
        return true;
    }

    /// @brief Get number of pending requests
    [[nodiscard]] size_type pending_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size();
    }

    /// @brief Cancel all pending requests
    void cancel_all() {
        std::unordered_map<request_id, std::unique_ptr<detail::pending_request_base>> to_cancel;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::swap(pending_, to_cancel);
        }

        for (auto& [id, pending] : to_cancel) {
            pending->set_error(error_status(status_code::canceled, no_rank, "Request canceled"));
        }
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<request_id, std::unique_ptr<detail::pending_request_base>> pending_;
    std::atomic<request_id> next_id_;
};

// ============================================================================
// Message Buffer
// ============================================================================

/// @brief Buffer for constructing RPC messages
class message_buffer {
public:
    /// @brief Default constructor
    message_buffer() = default;

    /// @brief Reserve space for header + payload
    void reserve(size_type payload_size) {
        data_.resize(message_header::serialized_size() + payload_size);
    }

    /// @brief Set the header
    void set_header(const message_header& header) {
        message_header::serialize(header, data_.data());
    }

    /// @brief Get mutable pointer to payload area
    [[nodiscard]] std::byte* payload_data() noexcept {
        return data_.data() + message_header::serialized_size();
    }

    /// @brief Get const pointer to payload area
    [[nodiscard]] const std::byte* payload_data() const noexcept {
        return data_.data() + message_header::serialized_size();
    }

    /// @brief Get total message data
    [[nodiscard]] const std::byte* data() const noexcept {
        return data_.data();
    }

    /// @brief Get total message size
    [[nodiscard]] size_type size() const noexcept {
        return data_.size();
    }

    /// @brief Get mutable access to underlying vector
    [[nodiscard]] std::vector<std::byte>& vector() noexcept {
        return data_;
    }

private:
    std::vector<std::byte> data_;
};

}  // namespace dtl::remote
