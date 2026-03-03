// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file action_registry.hpp
/// @brief Compile-time action registry for RPC dispatch
/// @details Provides mechanisms for registering and looking up actions
///          at compile-time, avoiding runtime registry overhead.
///          Dynamic handlers are now fully functional with serialization.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/remote/action.hpp>
#include <dtl/remote/argument_pack.hpp>
#include <dtl/serialization/serializer.hpp>

#include <array>
#include <optional>
#include <type_traits>
#include <utility>

namespace dtl::remote {

// ============================================================================
// Action List (Compile-Time)
// ============================================================================

/// @brief A compile-time list of actions
/// @tparam Actions Action types to include in the list
template <typename... Actions>
struct action_list {
    static constexpr size_type size = sizeof...(Actions);

    /// @brief Get action IDs as a compile-time array
    static constexpr std::array<action_id, size> ids() noexcept {
        return {Actions::id()...};
    }

    /// @brief Check if an action ID is in this list
    static constexpr bool contains(action_id id) noexcept {
        for (auto list_id : ids()) {
            if (list_id == id) return true;
        }
        return false;
    }

    /// @brief Get index of action ID in list (-1 if not found)
    static constexpr int index_of(action_id id) noexcept {
        auto id_array = ids();
        for (size_type i = 0; i < size; ++i) {
            if (id_array[i] == id) return static_cast<int>(i);
        }
        return -1;
    }
};

// ============================================================================
// Type-Erased Action Handler
// ============================================================================

/// @brief Type-erased action handler for runtime dispatch
/// @details Wraps an action's invoke functionality with serialization.
class action_handler {
public:
    /// @brief Handler function type
    /// @param request_data Serialized request arguments
    /// @param request_size Size of request data
    /// @param response_buffer Buffer for serialized response
    /// @param response_capacity Capacity of response buffer
    /// @return Size of serialized response (0 if void or error)
    using handler_fn = size_type(*)(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity);

    /// @brief Default constructor (invalid handler)
    action_handler() noexcept = default;

    /// @brief Construct from handler function
    explicit action_handler(action_id id, handler_fn fn) noexcept
        : id_(id), handler_(fn) {}

    /// @brief Get action ID
    [[nodiscard]] action_id id() const noexcept { return id_; }

    /// @brief Check if valid
    [[nodiscard]] bool valid() const noexcept { return handler_ != nullptr; }

    /// @brief Invoke the handler
    [[nodiscard]] size_type invoke(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity) const {
        if (!handler_) return 0;
        return handler_(request_data, request_size, response_buffer, response_capacity);
    }

private:
    action_id id_ = invalid_action_id;
    handler_fn handler_ = nullptr;
};

// ============================================================================
// Action Registry
// ============================================================================

/// @brief Compile-time action registry
/// @tparam N Maximum number of actions
/// @details Provides lookup by action_id at runtime while keeping
///          registration entirely at compile-time.
template <size_type N>
class action_registry {
public:
    /// @brief Default constructor (empty registry)
    constexpr action_registry() noexcept = default;

    /// @brief Construct from action handlers
    template <typename... Handlers>
        requires (sizeof...(Handlers) <= N)
    constexpr action_registry(Handlers... handlers) noexcept
        : handlers_{handlers...}
        , count_(sizeof...(Handlers)) {}

    /// @brief Look up a handler by action ID
    /// @param id Action ID to find
    /// @return Handler if found, nullopt otherwise
    [[nodiscard]] std::optional<action_handler> find(action_id id) const noexcept {
        for (size_type i = 0; i < count_; ++i) {
            if (handlers_[i].id() == id) {
                return handlers_[i];
            }
        }
        return std::nullopt;
    }

    /// @brief Check if registry contains an action
    [[nodiscard]] bool contains(action_id id) const noexcept {
        return find(id).has_value();
    }

    /// @brief Get number of registered actions
    [[nodiscard]] constexpr size_type size() const noexcept {
        return count_;
    }

    /// @brief Get capacity
    [[nodiscard]] static constexpr size_type capacity() noexcept {
        return N;
    }

private:
    // Allow registry_builder to access private members for construction
    template <size_type Capacity>
    friend class registry_builder;

    std::array<action_handler, N> handlers_{};
    size_type count_ = 0;
};

// ============================================================================
// Registry Builder
// ============================================================================

/// @brief Builder for creating action registries
/// @details Accumulates actions and produces a registry.
template <size_type Capacity = 64>
class registry_builder {
public:
    /// @brief Add an action to the registry
    /// @tparam A Action type
    /// @return Reference for chaining
    template <Action A>
    registry_builder& add() {
        if (count_ < Capacity) {
            handlers_[count_++] = make_handler<A>();
        }
        return *this;
    }

    /// @brief Add an action from an action instance
    template <auto Func>
    registry_builder& add(const action<Func>&) {
        return add<action<Func>>();
    }

    /// @brief Build the registry
    [[nodiscard]] action_registry<Capacity> build() const noexcept {
        action_registry<Capacity> result;
        // Copy handlers (constexpr-friendly in C++20)
        for (size_type i = 0; i < count_; ++i) {
            result.handlers_[i] = handlers_[i];
        }
        result.count_ = count_;
        return result;
    }

    /// @brief Get current count
    [[nodiscard]] size_type size() const noexcept { return count_; }

private:
    /// @brief Create a type-erased handler for an action
    /// @tparam A Action type satisfying the Action concept
    /// @details Creates a handler that:
    ///          1. Deserializes arguments from the request buffer
    ///          2. Invokes the action's function with those arguments
    ///          3. Serializes the result to the response buffer
    ///          4. Returns the number of bytes written
    template <Action A>
    static action_handler make_handler() {
        // Real implementation: create handler with serialization/dispatch
        return action_handler(A::id(), &invoke_action<A>);
    }

    /// @brief Type-erased invocation function for an action
    /// @tparam A Action type
    template <Action A>
    static size_type invoke_action(
        const std::byte* request_data,
        size_type request_size,
        std::byte* response_buffer,
        size_type response_capacity) noexcept {
        
        using request_type = typename A::request_type;
        using response_type = typename A::response_type;
        using pack_type = argument_pack_for<request_type>;
        
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
                    return 0;  // Insufficient space - caller should retry
                }
                
                // Serialize result
                return serialize(result, response_buffer);
            }
        } catch (...) {
            // Serialization or invocation failed
            return 0;
        }
    }

    std::array<action_handler, Capacity> handlers_{};
    size_type count_ = 0;
};

// ============================================================================
// Static Action Table
// ============================================================================

/// @brief Compile-time action table for static dispatch
/// @tparam Actions Variadic list of action types
template <typename... Actions>
class static_action_table {
public:
    static constexpr size_type size = sizeof...(Actions);

    /// @brief Dispatch to an action by ID
    /// @tparam F Callable to invoke with the action
    /// @param id Action ID to dispatch
    /// @param f Callable object
    /// @return true if action was found and invoked
    template <typename F>
    static bool dispatch(action_id id, F&& f) {
        return dispatch_impl<Actions...>(id, std::forward<F>(f));
    }

private:
    template <typename First, typename... Rest, typename F>
    static bool dispatch_impl(action_id id, F&& f) {
        if (First::id() == id) {
            f(First{});
            return true;
        }
        if constexpr (sizeof...(Rest) > 0) {
            return dispatch_impl<Rest...>(id, std::forward<F>(f));
        }
        return false;
    }

    // Base case: no more actions
    template <typename F>
    static bool dispatch_impl(action_id, F&&) {
        return false;
    }
};

}  // namespace dtl::remote
