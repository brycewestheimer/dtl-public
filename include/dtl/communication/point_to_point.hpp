// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file point_to_point.hpp
/// @brief Point-to-point communication operations
/// @details Provides send, recv, and non-blocking variants.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/communication/message_status.hpp>

#include <span>
#include <utility>

namespace dtl {

// ============================================================================
// Message Tags
// ============================================================================

/// @brief Wildcard tag matching any message tag
inline constexpr int any_tag = -1;

/// @brief Wildcard source matching any sender
inline constexpr rank_t any_source = -1;

// ============================================================================
// Send Operations
// ============================================================================

/// @brief Send data to a specific rank
/// @tparam Comm Communicator type satisfying Communicator concept
/// @tparam T Element type
/// @param comm The communicator
/// @param data Data to send
/// @param dest Destination rank
/// @param tag Message tag
/// @return Result indicating success or error
template <Communicator Comm, typename T>
result<void> send(Comm& comm, std::span<const T> data, rank_t dest, int tag = 0) {
    comm.send(data.data(), data.size() * sizeof(T), dest, tag);
    return {};
}

/// @brief Send a single value to a specific rank
/// @tparam Comm Communicator type
/// @tparam T Value type
/// @param comm The communicator
/// @param value Value to send
/// @param dest Destination rank
/// @param tag Message tag
/// @return Result indicating success or error
template <Communicator Comm, typename T>
result<void> send(Comm& comm, const T& value, rank_t dest, int tag = 0) {
    comm.send(&value, sizeof(T), dest, tag);
    return {};
}

/// @brief Non-blocking send
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param data Data to send
/// @param dest Destination rank
/// @param tag Message tag
/// @return Request handle for completion checking
template <Communicator Comm, typename T>
[[nodiscard]] request_handle isend(Comm& comm, std::span<const T> data, rank_t dest, int tag = 0) {
    return comm.isend(data.data(), data.size() * sizeof(T), dest, tag);
}

/// @brief Synchronous send (blocks until message received)
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param data Data to send
/// @param dest Destination rank
/// @param tag Message tag
/// @return Result indicating success or error
template <Communicator Comm, typename T>
result<void> ssend(Comm& comm, std::span<const T> data, rank_t dest, int tag = 0) {
    comm.ssend(data.data(), data.size() * sizeof(T), dest, tag);
    return {};
}

/// @brief Ready send (assumes matching receive is posted)
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param data Data to send
/// @param dest Destination rank
/// @param tag Message tag
/// @return Result indicating success or error
template <Communicator Comm, typename T>
result<void> rsend(Comm& comm, std::span<const T> data, rank_t dest, int tag = 0) {
    comm.rsend(data.data(), data.size() * sizeof(T), dest, tag);
    return {};
}

// ============================================================================
// Receive Operations
// ============================================================================

/// @brief Receive data from a specific rank
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param data Buffer to receive into
/// @param source Source rank (or any_source)
/// @param tag Message tag (or any_tag)
/// @return Result containing message status
template <Communicator Comm, typename T>
result<message_status> recv(Comm& comm, std::span<T> data, rank_t source = any_source, int tag = any_tag) {
    comm.recv(data.data(), data.size() * sizeof(T), source, tag);
    message_status status;
    // Resolve wildcard values to actual values
    // For single-rank communicators, source is always 0
    status.source = (source == any_source) ? rank_t{0} : source;
    status.tag = (tag == any_tag) ? 0 : tag;
    status.count = data.size();
    status.error = 0;
    return status;
}

/// @brief Receive a single value from a specific rank
/// @tparam Comm Communicator type
/// @tparam T Value type
/// @param comm The communicator
/// @param value Reference to store received value
/// @param source Source rank (or any_source)
/// @param tag Message tag (or any_tag)
/// @return Result containing message status
template <Communicator Comm, typename T>
result<message_status> recv(Comm& comm, T& value, rank_t source = any_source, int tag = any_tag) {
    comm.recv(&value, sizeof(T), source, tag);
    message_status status;
    // Resolve wildcard values to actual values
    status.source = (source == any_source) ? rank_t{0} : source;
    status.tag = (tag == any_tag) ? 0 : tag;
    status.count = 1;
    status.error = 0;
    return status;
}

/// @brief Non-blocking receive
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param data Buffer to receive into
/// @param source Source rank (or any_source)
/// @param tag Message tag (or any_tag)
/// @return Request handle for completion checking
template <Communicator Comm, typename T>
[[nodiscard]] request_handle irecv(Comm& comm, std::span<T> data, rank_t source = any_source, int tag = any_tag) {
    return comm.irecv(data.data(), data.size() * sizeof(T), source, tag);
}

// ============================================================================
// Request Operations
// ============================================================================

/// @brief Wait for a non-blocking operation to complete
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @param req Request handle
/// @return Result containing message status
template <Communicator Comm>
result<message_status> wait(Comm& comm, request_handle& req) {
    comm.wait(req);
    message_status status;
    return status;
}

/// @brief Test if a non-blocking operation has completed
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @param req Request handle
/// @return true if complete, false otherwise
template <Communicator Comm>
[[nodiscard]] bool test(Comm& comm, request_handle& req) {
    return comm.test(req);
}

/// @brief Wait for all requests to complete
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @param requests Array of request handles
/// @return Result indicating success or error
template <Communicator Comm>
result<void> wait_all(Comm& comm, std::span<request_handle> requests) {
    for (auto& req : requests) {
        comm.wait(req);
    }
    return {};
}

/// @brief Wait for any request to complete
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @param requests Array of request handles
/// @return Index of completed request
template <Communicator Comm>
[[nodiscard]] result<size_type> wait_any(Comm& comm, std::span<request_handle> requests) {
    if (requests.empty()) {
        return make_error<size_type>(status_code::invalid_argument, "No requests to wait on");
    }
    return comm.waitany(requests.data(), requests.size());
}

// ============================================================================
// Probe Operations
// ============================================================================

/// @brief Probe for incoming message without receiving
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @param source Source rank (or any_source)
/// @param tag Message tag (or any_tag)
/// @return Result containing message status
template <Communicator Comm>
result<message_status> probe(Comm& comm, rank_t source = any_source, int tag = any_tag) {
    return comm.probe(source, tag);
}

/// @brief Non-blocking probe for incoming message
/// @tparam Comm Communicator type
/// @param comm The communicator
/// @param source Source rank (or any_source)
/// @param tag Message tag (or any_tag)
/// @return Result containing pair: {message_available, status}
template <Communicator Comm>
[[nodiscard]] result<std::pair<bool, message_status>> iprobe(Comm& comm, rank_t source = any_source, int tag = any_tag) {
    return comm.iprobe(source, tag);
}

// ============================================================================
// Send-Receive Operations
// ============================================================================

/// @brief Combined send and receive
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to send
/// @param dest Destination rank
/// @param send_tag Send message tag
/// @param recv_data Buffer for received data
/// @param source Source rank
/// @param recv_tag Receive message tag
/// @return Result containing message status
template <Communicator Comm, typename T>
result<message_status> sendrecv(
    Comm& comm,
    std::span<const T> send_data, rank_t dest, int send_tag,
    std::span<T> recv_data, rank_t source, int recv_tag) {
    // Issue non-blocking receive first
    auto recv_req = comm.irecv(recv_data.data(), recv_data.size() * sizeof(T), source, recv_tag);

    // Then blocking send
    comm.send(send_data.data(), send_data.size() * sizeof(T), dest, send_tag);

    // Wait for receive
    comm.wait(recv_req);

    message_status status;
    // Resolve wildcard values to actual values
    status.source = (source == any_source) ? rank_t{0} : source;
    status.tag = (recv_tag == any_tag) ? 0 : recv_tag;
    status.count = recv_data.size();
    status.error = 0;
    return status;
}

/// @brief Send-receive with replace (in-place)
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param data Buffer for send/receive
/// @param dest Destination rank
/// @param send_tag Send message tag
/// @param source Source rank
/// @param recv_tag Receive message tag
/// @return Result containing message status
template <Communicator Comm, typename T>
result<message_status> sendrecv_replace(
    Comm& comm,
    std::span<T> data, rank_t dest, int send_tag,
    rank_t source, int recv_tag) {
    comm.sendrecv_replace(data.data(), data.size() * sizeof(T),
                           dest, send_tag, source, recv_tag);
    message_status status;
    // Resolve wildcard values to actual values
    status.source = (source == any_source) ? rank_t{0} : source;
    status.tag = (recv_tag == any_tag) ? 0 : recv_tag;
    status.count = data.size();
    status.error = 0;
    return status;
}

}  // namespace dtl
