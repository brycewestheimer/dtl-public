// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rpc_serialization.hpp
/// @brief RPC-specific serialization utilities
/// @details Provides payload format and helpers for dynamic RPC invocation.
///          Works with dtl::serializer<T> and external adapters (cereal, boost).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/remote/action.hpp>
#include <dtl/remote/argument_pack.hpp>
#include <dtl/remote/rpc_request.hpp>
#include <dtl/serialization/serializer.hpp>

#include <cstring>
#include <vector>

namespace dtl::remote {

// ============================================================================
// RPC Payload Format
// ============================================================================

/// @brief RPC payload structure
/// @details Wire format for RPC messages:
///
///   ┌────────────────────────────────────────┐
///   │           message_header               │  (fixed size)
///   ├────────────────────────────────────────┤
///   │           payload bytes                │  (variable size)
///   └────────────────────────────────────────┘
///
/// For requests:  payload = serialized argument tuple
/// For responses: payload = serialized result (or error)
/// For errors:    payload = error_payload struct

/// @brief Error payload for RPC error responses
struct error_payload {
    int32_t error_code;
    int32_t reserved;
    // Followed by variable-length error message (not included in struct size)
    
    /// @brief Serialized size (without message)
    static constexpr size_type base_size() noexcept {
        return sizeof(int32_t) * 2;
    }
    
    /// @brief Serialize to buffer
    static size_type serialize(int32_t code, const char* message,
                               std::byte* buffer, size_type capacity) noexcept {
        if (capacity < base_size()) return 0;
        
        error_payload hdr{code, 0};
        std::memcpy(buffer, &hdr, base_size());
        
        if (message) {
            size_type msg_len = std::strlen(message);
            size_type total = base_size() + msg_len;
            if (total <= capacity) {
                std::memcpy(buffer + base_size(), message, msg_len);
                return total;
            }
        }
        return base_size();
    }
    
    /// @brief Deserialize from buffer
    static error_payload deserialize(const std::byte* buffer) noexcept {
        error_payload hdr;
        std::memcpy(&hdr, buffer, base_size());
        return hdr;
    }
};

// ============================================================================
// RPC Serialization Helpers
// ============================================================================

/// @brief Serialize an RPC request
/// @tparam A Action type
/// @tparam Args Argument types
/// @param source_rank Originating rank
/// @param request_id Request ID for correlation
/// @param args Arguments to serialize
/// @return Complete message buffer
template <Action A, typename... Args>
[[nodiscard]] std::vector<std::byte> serialize_request(
    rank_t source_rank,
    request_id request_id,
    Args&&... args) {
    
    using pack_t = argument_pack<std::decay_t<Args>...>;
    
    // Serialize arguments
    auto payload = pack_t::serialize_to_vector(std::forward<Args>(args)...);
    
    // Build complete message
    std::vector<std::byte> buffer;
    buffer.resize(message_header::serialized_size() + payload.size());
    
    auto header = message_header::make_request(
        A::id(), request_id, payload.size(), source_rank);
    message_header::serialize(header, buffer.data());
    
    if (!payload.empty()) {
        std::memcpy(buffer.data() + message_header::serialized_size(),
                   payload.data(), payload.size());
    }
    
    return buffer;
}

/// @brief Serialize an RPC response
/// @tparam T Result type
/// @param request_id Request ID for correlation
/// @param source_rank Responding rank
/// @param result The result value
/// @return Complete message buffer
template <typename T>
[[nodiscard]] std::vector<std::byte> serialize_response(
    request_id request_id,
    rank_t source_rank,
    const T& result) {
    
    size_type result_size = serialized_size(result);
    
    std::vector<std::byte> buffer;
    buffer.resize(message_header::serialized_size() + result_size);
    
    auto header = message_header::make_response(
        request_id, result_size, source_rank);
    message_header::serialize(header, buffer.data());
    
    serialize(result, buffer.data() + message_header::serialized_size());
    
    return buffer;
}

/// @brief Serialize a void RPC response
/// @param request_id Request ID for correlation
/// @param source_rank Responding rank
/// @return Complete message buffer (header only, no payload)
[[nodiscard]] inline std::vector<std::byte> serialize_void_response(
    request_id request_id,
    rank_t source_rank) {
    
    std::vector<std::byte> buffer;
    buffer.resize(message_header::serialized_size());
    
    auto header = message_header::make_response(
        request_id, 0, source_rank);
    message_header::serialize(header, buffer.data());
    
    return buffer;
}

/// @brief Serialize an RPC error response
/// @param request_id Request ID for correlation
/// @param source_rank Responding rank
/// @param error_code Error code
/// @param message Optional error message
/// @return Complete message buffer
[[nodiscard]] inline std::vector<std::byte> serialize_error(
    request_id request_id,
    rank_t source_rank,
    int32_t error_code,
    const char* message = nullptr) {
    
    size_type msg_len = message ? std::strlen(message) : 0;
    size_type payload_size = error_payload::base_size() + msg_len;
    
    std::vector<std::byte> buffer;
    buffer.resize(message_header::serialized_size() + payload_size);
    
    auto header = message_header::make_error(
        request_id, payload_size, source_rank);
    message_header::serialize(header, buffer.data());
    
    error_payload::serialize(error_code, message,
        buffer.data() + message_header::serialized_size(),
        payload_size);
    
    return buffer;
}

// ============================================================================
// RPC Deserialization Helpers
// ============================================================================

/// @brief Deserialize RPC request arguments
/// @tparam ArgsTuple Tuple type for arguments
/// @param payload Payload buffer (after header)
/// @param payload_size Payload size
/// @return Tuple of deserialized arguments
template <typename ArgsTuple>
[[nodiscard]] ArgsTuple deserialize_request_args(
    const std::byte* payload,
    size_type payload_size) {
    
    using pack_t = argument_pack_for<ArgsTuple>;
    return pack_t::deserialize(payload, payload_size);
}

/// @brief Deserialize RPC response result
/// @tparam T Result type
/// @param payload Payload buffer (after header)
/// @param payload_size Payload size
/// @return Deserialized result
template <typename T>
[[nodiscard]] T deserialize_response_result(
    const std::byte* payload,
    size_type payload_size) {
    
    return deserialize<T>(payload, payload_size);
}

/// @brief Deserialize RPC error payload
/// @param payload Payload buffer (after header)
/// @param payload_size Payload size
/// @return Error status
[[nodiscard]] inline status deserialize_error(
    const std::byte* payload,
    size_type payload_size) {
    
    if (payload_size < error_payload::base_size()) {
        return status(status_code::serialization_error, "Malformed error payload");
    }
    
    auto err = error_payload::deserialize(payload);
    
    // Extract message if present
    if (payload_size > error_payload::base_size()) {
        size_type msg_len = payload_size - error_payload::base_size();
        std::string message(
            reinterpret_cast<const char*>(payload + error_payload::base_size()),
            msg_len);
        return status(static_cast<status_code>(err.error_code), std::move(message));
    }
    
    return status(static_cast<status_code>(err.error_code));
}

// ============================================================================
// RPC Message Processing
// ============================================================================

/// @brief Process an incoming RPC message using a registry
/// @tparam N Registry capacity
/// @param registry Action registry for handler lookup
/// @param message Complete message buffer
/// @param message_size Message size
/// @param response_handler Callback for sending response
/// @return true if message was processed
template <size_type N>
bool process_rpc_message(
    const action_registry<N>& registry,
    const std::byte* message,
    size_type message_size,
    std::function<void(const std::byte*, size_type)> response_handler) {
    
    if (message_size < message_header::serialized_size()) {
        return false;
    }
    
    auto header = message_header::deserialize(message);
    const std::byte* payload = message + message_header::serialized_size();
    size_type payload_size = message_size - message_header::serialized_size();
    
    // Look up handler
    auto handler_opt = registry.find(header.action);
    if (!handler_opt) {
        if (header.msg_type == message_header::request_type) {
            // Send error response for unknown action
            auto error_msg = serialize_error(
                header.request, no_rank,
                static_cast<int32_t>(status_code::not_found),
                "Unknown action ID");
            response_handler(error_msg.data(), error_msg.size());
        }
        return false;
    }
    
    const auto& handler = *handler_opt;
    
    // Prepare response buffer
    std::vector<std::byte> response_payload(4096);  // Reasonable default
    
    // Invoke handler
    size_type result_size = handler.invoke(
        payload, payload_size,
        response_payload.data(), response_payload.size());
    
    // Send response if this was a request (not fire-and-forget)
    if (header.msg_type == message_header::request_type) {
        std::vector<std::byte> response;
        response.resize(message_header::serialized_size() + result_size);
        
        auto resp_header = message_header::make_response(
            header.request, result_size, no_rank);
        message_header::serialize(resp_header, response.data());
        
        if (result_size > 0) {
            std::memcpy(response.data() + message_header::serialized_size(),
                       response_payload.data(), result_size);
        }
        
        response_handler(response.data(), response.size());
    }
    
    return true;
}

}  // namespace dtl::remote
