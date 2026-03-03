// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file dynamic_handler.hpp
/// @brief Dynamic handler creation with type erasure for RPC dispatch
/// @details Provides fully functional dynamic handlers that deserialize
///          arguments, invoke functions, and serialize results.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/remote/action.hpp>
#include <dtl/remote/argument_pack.hpp>
#include <dtl/serialization/serializer.hpp>

#include <cstring>
#include <exception>
#include <type_traits>
#include <vector>

namespace dtl::remote {

// ============================================================================
// Dynamic Handler Result
// ============================================================================

/// @brief Result of a dynamic handler invocation
struct handler_result {
    /// @brief Number of bytes written to response buffer
    size_type bytes_written = 0;
    
    /// @brief Error code (0 = success)
    int error_code = 0;
    
    /// @brief Whether invocation succeeded
    [[nodiscard]] bool ok() const noexcept { 
        return error_code == 0; 
    }
    
    /// @brief Create success result
    static handler_result make_ok(size_type bytes) noexcept {
        return {bytes, 0};
    }
    
    /// @brief Create error result
    static handler_result make_error(int code) noexcept {
        return {0, code};
    }
    
    /// @brief Error codes
    enum error : int {
        err_none = 0,
        err_buffer_too_small = 1,
        err_deserialization_failed = 2,
        err_invocation_failed = 3,
        err_invalid_handler = 4
    };
};

// ============================================================================
// Dynamic Handler Implementation
// ============================================================================

namespace detail {

/// @brief Create the handler function for an action type
/// @tparam A Action type satisfying the Action concept
/// @details This creates a static function that:
///          1. Deserializes arguments from the request buffer
///          2. Invokes the action's function with those arguments
///          3. Serializes the result to the response buffer
///          4. Returns the number of bytes written (0 on error or void return)
template <Action A>
struct handler_factory {
    using request_type = typename A::request_type;
    using response_type = typename A::response_type;
    using pack_type = argument_pack_for<request_type>;
    
    /// @brief The actual handler function
    /// @param request_data Serialized request arguments
    /// @param request_size Size of request data
    /// @param response_buffer Buffer for serialized response
    /// @param response_capacity Capacity of response buffer
    /// @return Size of serialized response (0 if void, error, or insufficient space)
    static size_type invoke(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity) noexcept {
        
        try {
            // Deserialize arguments
            request_type args = pack_type::deserialize(request_data, request_size);
            
            // Invoke the function
            if constexpr (std::is_void_v<response_type>) {
                A::invoke_tuple(args);
                return 0;  // Void return
            } else {
                response_type result = A::invoke_tuple(args);
                
                // Check if we have space for the result
                size_type needed = serialized_size(result);
                if (needed > response_capacity) {
                    return 0;  // Insufficient space
                }
                
                // Serialize result
                return serialize(result, response_buffer);
            }
        } catch (...) {
            // Serialization or invocation failed
            return 0;
        }
    }
};

/// @brief Create handler with error reporting
/// @tparam A Action type
template <Action A>
struct handler_factory_with_errors {
    using request_type = typename A::request_type;
    using response_type = typename A::response_type;
    using pack_type = argument_pack_for<request_type>;
    
    /// @brief Invoke with detailed error reporting
    static handler_result invoke_with_result(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity) noexcept {
        
        try {
            // Deserialize arguments
            request_type args = pack_type::deserialize(request_data, request_size);
            
            // Invoke the function
            if constexpr (std::is_void_v<response_type>) {
                A::invoke_tuple(args);
                return handler_result::make_ok(0);
            } else {
                response_type result = A::invoke_tuple(args);
                
                // Check if we have space for the result
                size_type needed = serialized_size(result);
                if (needed > response_capacity) {
                    return handler_result::make_error(handler_result::err_buffer_too_small);
                }
                
                // Serialize result
                size_type written = serialize(result, response_buffer);
                return handler_result::make_ok(written);
            }
        } catch (const std::exception&) {
            return handler_result::make_error(handler_result::err_invocation_failed);
        } catch (...) {
            return handler_result::make_error(handler_result::err_invocation_failed);
        }
    }
};

}  // namespace detail

// ============================================================================
// Extended Action Handler
// ============================================================================

/// @brief Extended handler with additional metadata and error reporting
/// @details Wraps the basic handler_fn with richer functionality.
class extended_action_handler {
public:
    /// @brief Handler function type (same as action_handler::handler_fn)
    using handler_fn = size_type(*)(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity);
    
    /// @brief Handler with result type
    using handler_with_result_fn = handler_result(*)(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity);
    
    /// @brief Default constructor (invalid handler)
    extended_action_handler() noexcept = default;
    
    /// @brief Construct with handler functions
    extended_action_handler(
        action_id id,
        handler_fn fn,
        handler_with_result_fn fn_with_result,
        size_type arity,
        bool is_void_return) noexcept
        : id_(id)
        , handler_(fn)
        , handler_with_result_(fn_with_result)
        , arity_(arity)
        , is_void_return_(is_void_return) {}
    
    /// @brief Get action ID
    [[nodiscard]] action_id id() const noexcept { return id_; }
    
    /// @brief Check if handler is valid
    [[nodiscard]] bool valid() const noexcept { return handler_ != nullptr; }
    
    /// @brief Get function arity
    [[nodiscard]] size_type arity() const noexcept { return arity_; }
    
    /// @brief Check if function returns void
    [[nodiscard]] bool is_void() const noexcept { return is_void_return_; }
    
    /// @brief Invoke the handler (simple version)
    [[nodiscard]] size_type invoke(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity) const noexcept {
        
        if (!handler_) return 0;
        return handler_(request_data, request_size, 
                       response_buffer, response_capacity);
    }
    
    /// @brief Invoke with detailed result
    [[nodiscard]] handler_result invoke_with_result(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity) const noexcept {
        
        if (!handler_with_result_) {
            return handler_result::make_error(handler_result::err_invalid_handler);
        }
        return handler_with_result_(request_data, request_size,
                                   response_buffer, response_capacity);
    }
    
private:
    action_id id_ = invalid_action_id;
    handler_fn handler_ = nullptr;
    handler_with_result_fn handler_with_result_ = nullptr;
    size_type arity_ = 0;
    bool is_void_return_ = false;
};

// ============================================================================
// Handler Creation Functions
// ============================================================================

/// @brief Create a dynamic handler for an action type
/// @tparam A Action type satisfying the Action concept
/// @return Handler that can invoke the action dynamically
template <Action A>
[[nodiscard]] action_handler make_dynamic_handler() noexcept {
    return action_handler(
        A::id(),
        &detail::handler_factory<A>::invoke
    );
}

/// @brief Create an extended dynamic handler with metadata
/// @tparam A Action type satisfying the Action concept
/// @return Extended handler with metadata and error reporting
template <Action A>
[[nodiscard]] extended_action_handler make_extended_handler() noexcept {
    return extended_action_handler(
        A::id(),
        &detail::handler_factory<A>::invoke,
        &detail::handler_factory_with_errors<A>::invoke_with_result,
        A::arity,
        A::is_void
    );
}

// ============================================================================
// Handler Invocation Utilities
// ============================================================================

/// @brief Invoke a handler with typed arguments and result
/// @tparam Result Expected result type
/// @tparam Args Argument types
/// @param handler The handler to invoke
/// @param args Arguments to pass
/// @return Result wrapped in dtl::result
template <typename Result, typename... Args>
[[nodiscard]] result<Result> invoke_handler(
    const action_handler& handler,
    Args&&... args) {
    
    if (!handler.valid()) {
        return error_status(status_code::invalid_argument, no_rank, "Invalid handler");
    }
    
    // Serialize arguments
    using pack_t = argument_pack<std::decay_t<Args>...>;
    auto request = pack_t::serialize_to_vector(std::forward<Args>(args)...);
    
    // Prepare response buffer
    std::vector<std::byte> response;
    if constexpr (!std::is_void_v<Result>) {
        // Estimate size - for fixed-size types, use serialized_size()
        response.resize(sizeof(Result) * 2 + 256);  // Conservative estimate
    }
    
    // Invoke
    size_type written = handler.invoke(
        request.data(), request.size(),
        response.data(), response.size());
    
    if constexpr (std::is_void_v<Result>) {
        return {};
    } else {
        if (written == 0) {
            return error_status(status_code::operation_failed, no_rank, "Handler invocation failed");
        }
        return deserialize<Result>(response.data(), written);
    }
}

/// @brief Invoke handler with a pre-serialized request
/// @tparam Result Expected result type
/// @param handler The handler to invoke
/// @param request_data Serialized request
/// @param request_size Request size
/// @return Result wrapped in dtl::result
template <typename Result>
[[nodiscard]] result<Result> invoke_handler_raw(
    const action_handler& handler,
    const std::byte* request_data,
    size_type request_size) {
    
    if (!handler.valid()) {
        return error_status(status_code::invalid_argument, no_rank, "Invalid handler");
    }
    
    // Prepare response buffer
    std::vector<std::byte> response;
    if constexpr (!std::is_void_v<Result>) {
        response.resize(sizeof(Result) * 2 + 256);
    }
    
    // Invoke
    size_type written = handler.invoke(
        request_data, request_size,
        response.data(), response.size());
    
    if constexpr (std::is_void_v<Result>) {
        return {};
    } else {
        if (written == 0) {
            return error_status(status_code::operation_failed, no_rank, "Handler invocation failed");
        }
        return deserialize<Result>(response.data(), written);
    }
}

}  // namespace dtl::remote
