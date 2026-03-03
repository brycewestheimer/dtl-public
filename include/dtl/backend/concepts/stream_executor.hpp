// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file stream_executor.hpp
/// @brief StreamExecutor concept for GPU-style async execution
/// @details Defines requirements for stream-based execution (CUDA/HIP style).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/executor.hpp>
#include <dtl/backend/concepts/event.hpp>

#include <concepts>
#include <functional>

namespace dtl {

// ============================================================================
// Stream Priority
// ============================================================================

/// @brief Priority level for streams
enum class stream_priority {
    low,      ///< Background/low priority work
    normal,   ///< Default priority
    high      ///< High priority (may preempt)
};

// ============================================================================
// Stream Properties
// ============================================================================

/// @brief Properties describing a stream
struct stream_properties {
    /// @brief Stream priority
    stream_priority priority = stream_priority::normal;

    /// @brief Whether stream operations are serialized
    bool serialized = true;

    /// @brief Whether stream supports concurrent kernel execution
    bool concurrent_kernels = false;

    /// @brief Device index this stream is bound to
    int device_id = 0;
};

// ============================================================================
// Stream Executor Concept
// ============================================================================

/// @brief Core stream executor concept for GPU-style async execution
/// @details Defines requirements for executing work on a stream/queue.
///
/// @par Required Operations:
/// - execute(): Queue work on the stream
/// - synchronize(): Wait for all queued work to complete
/// - record_event(): Record an event on the stream
template <typename T>
concept StreamExecutor = Executor<T> &&
    requires(T& exec, const T& cexec, std::function<void()> f) {
    // Synchronize stream
    { exec.synchronize() } -> std::same_as<void>;

    // Query if stream is idle
    { cexec.is_idle() } -> std::same_as<bool>;

    // Get stream properties
    { cexec.properties() } -> std::same_as<stream_properties>;
};

// ============================================================================
// Event Recording Stream Concept
// ============================================================================

/// @brief Stream executor that can record events
template <typename T, typename Event>
concept EventRecordingStream = StreamExecutor<T> && dtl::Event<Event> &&
    requires(T& exec, Event& event) {
    // Record event on stream
    { exec.record(event) } -> std::same_as<void>;

    // Wait for event before executing further work
    { exec.wait_event(event) } -> std::same_as<void>;
};

// ============================================================================
// Multi-Stream Executor Concept
// ============================================================================

/// @brief Executor managing multiple streams
template <typename T>
concept MultiStreamExecutor = StreamExecutor<T> &&
    requires(T& exec, const T& cexec, size_type stream_id) {
    // Get number of streams
    { cexec.num_streams() } -> std::same_as<size_type>;

    // Select stream for subsequent operations
    { exec.set_stream(stream_id) } -> std::same_as<void>;

    // Get current stream index
    { cexec.current_stream() } -> std::same_as<size_type>;

    // Synchronize specific stream
    { exec.synchronize_stream(stream_id) } -> std::same_as<void>;

    // Synchronize all streams
    { exec.synchronize_all() } -> std::same_as<void>;
};

// ============================================================================
// Stream Executor Traits
// ============================================================================

/// @brief Traits for stream executor types
template <typename Exec>
struct stream_executor_traits {
    /// @brief Whether executor manages GPU streams
    static constexpr bool is_gpu = false;

    /// @brief Whether executor supports multiple streams
    static constexpr bool is_multi_stream = false;

    /// @brief Whether executor supports event recording
    static constexpr bool supports_events = false;

    /// @brief Whether executor supports stream priorities
    static constexpr bool supports_priorities = false;

    /// @brief Maximum number of streams (0 = unlimited)
    static constexpr size_type max_streams = 1;
};

// ============================================================================
// Stream Tag Types
// ============================================================================

/// @brief Tag for CUDA streams
struct cuda_stream_tag {};

/// @brief Tag for HIP streams
struct hip_stream_tag {};

/// @brief Tag for SYCL queues
struct sycl_queue_tag {};

/// @brief Tag for host-side simulated streams
struct host_stream_tag {};

// ============================================================================
// Host Stream Executor
// ============================================================================

/// @brief Host-side sequential stream executor
/// @details Simulates stream semantics on the host.
class host_stream_executor {
public:
    using tag_type = host_stream_tag;

    /// @brief Execute callable (synchronously for host streams)
    template <typename F>
    void execute(F&& f) {
        std::forward<F>(f)();
    }

    /// @brief Get executor name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "host_stream";
    }

    /// @brief Synchronize (no-op for host)
    void synchronize() noexcept {}

    /// @brief Check if idle (always true for synchronous host)
    [[nodiscard]] static constexpr bool is_idle() noexcept {
        return true;
    }

    /// @brief Get stream properties
    [[nodiscard]] static constexpr stream_properties properties() noexcept {
        return stream_properties{
            .priority = stream_priority::normal,
            .serialized = true,
            .concurrent_kernels = false,
            .device_id = -1  // Host
        };
    }

    /// @brief Record a null event (no-op)
    void record(null_event& /*event*/) noexcept {}

    /// @brief Wait for a null event (no-op)
    void wait_event(null_event& /*event*/) noexcept {}
};

/// @brief Traits specialization for host_stream_executor
template <>
struct stream_executor_traits<host_stream_executor> {
    static constexpr bool is_gpu = false;
    static constexpr bool is_multi_stream = false;
    static constexpr bool supports_events = true;
    static constexpr bool supports_priorities = false;
    static constexpr size_type max_streams = 1;
};

// ============================================================================
// Stream Utilities
// ============================================================================

/// @brief Execute work on a stream and return an event
/// @tparam Exec Stream executor type
/// @tparam Event Event type
/// @tparam F Callable type
/// @param exec The stream executor
/// @param f The work to execute
/// @return Event representing completion
template <typename Exec, typename Event, typename F>
    requires StreamExecutor<Exec> && dtl::Event<Event>
Event execute_async(Exec& exec, F&& f) {
    exec.execute(std::forward<F>(f));
    Event event;
    if constexpr (EventRecordingStream<Exec, Event>) {
        exec.record(event);
    }
    return event;
}

/// @brief Synchronize between two streams via event
/// @tparam Exec Stream executor type
/// @tparam Event Event type
/// @param producer Stream that produces the event
/// @param consumer Stream that waits on the event
template <typename Exec, typename Event>
    requires EventRecordingStream<Exec, Event>
void stream_synchronize(Exec& producer, Exec& consumer) {
    Event event;
    producer.record(event);
    consumer.wait_event(event);
}

}  // namespace dtl
