// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file action.hpp
/// @brief Compile-time action type for RPC dispatch
/// @details Defines action templates that enable type-safe RPC with
///          compile-time registration (no runtime plugin registry).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <cstdint>
#include <functional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace dtl::remote {

// ============================================================================
// Action ID Type
// ============================================================================

/// @brief Unique identifier for an action
/// @details Computed at compile-time from the function signature.
using action_id = std::uint64_t;

/// @brief Sentinel for invalid/unknown action
inline constexpr action_id invalid_action_id = 0;

// ============================================================================
// Function Traits
// ============================================================================

namespace detail {

/// @brief Extract function traits from a function pointer
template <typename Func>
struct function_traits;

/// @brief Specialization for regular function pointers
template <typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_type arity = sizeof...(Args);
};

/// @brief Specialization for member function pointers
template <typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...)> {
    using return_type = R;
    using class_type = C;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_type arity = sizeof...(Args);
};

/// @brief Specialization for const member function pointers
template <typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    using return_type = R;
    using class_type = C;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_type arity = sizeof...(Args);
};

/// @brief Specialization for noexcept function pointers
template <typename R, typename... Args>
struct function_traits<R(*)(Args...) noexcept> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_type arity = sizeof...(Args);
};

// ============================================================================
// Compile-Time String Hash (FNV-1a)
// ============================================================================

/// @brief FNV-1a hash constant for 64-bit
inline constexpr action_id fnv1a_basis = 14695981039346656037ULL;
inline constexpr action_id fnv1a_prime = 1099511628211ULL;

/// @brief Compute FNV-1a hash of a string at compile time
constexpr action_id fnv1a_hash(std::string_view str) noexcept {
    action_id hash = fnv1a_basis;
    for (char c : str) {
        hash ^= static_cast<action_id>(static_cast<unsigned char>(c));
        hash *= fnv1a_prime;
    }
    return hash;
}

/// @brief Mix two hashes together
constexpr action_id hash_combine(action_id h1, action_id h2) noexcept {
    return h1 ^ (h2 * fnv1a_prime + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

// ============================================================================
// Type Name Hash
// ============================================================================

/// @brief Get a compile-time hash for a type
/// @details Uses __PRETTY_FUNCTION__ or equivalent for type name
template <typename T>
constexpr action_id type_hash() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return fnv1a_hash(__PRETTY_FUNCTION__);
#elif defined(_MSC_VER)
    return fnv1a_hash(__FUNCSIG__);
#else
    // Fallback: hash sizeof and alignof (less unique but portable)
    return hash_combine(sizeof(T), alignof(T));
#endif
}

/// @brief Get hash for a tuple of types (variadic)
template <typename... Ts>
constexpr action_id tuple_type_hash() noexcept {
    if constexpr (sizeof...(Ts) == 0) {
        return fnv1a_basis;
    } else {
        action_id hashes[] = {type_hash<Ts>()...};
        action_id result = fnv1a_basis;
        for (auto h : hashes) {
            result = hash_combine(result, h);
        }
        return result;
    }
}

}  // namespace detail

// ============================================================================
// Action Template
// ============================================================================

/// @brief Compile-time action descriptor
/// @tparam Func Function pointer (must be auto-deducible)
/// @details Provides type-safe metadata for RPC dispatch. Each instantiation
///          gets a unique action_id based on the function signature.
///
/// @par Example usage:
/// @code
/// int add(int a, int b) { return a + b; }
///
/// // Create action for the function
/// constexpr auto add_action = dtl::remote::action<&add>{};
///
/// // Use the action
/// static_assert(add_action.id() != 0);
/// using args = decltype(add_action)::request_type;  // std::tuple<int, int>
/// using result = decltype(add_action)::response_type;  // int
/// @endcode
template <auto Func>
struct action {
    /// @brief Function pointer type
    using func_type = decltype(Func);

    /// @brief Function traits
    using traits = detail::function_traits<func_type>;

    /// @brief Request type (tuple of argument types)
    using request_type = typename traits::args_tuple;

    /// @brief Response type (return type)
    using response_type = typename traits::return_type;

    /// @brief Number of arguments
    static constexpr size_type arity = traits::arity;

    /// @brief Check if this is a void action
    static constexpr bool is_void = std::is_void_v<response_type>;

    /// @brief Get the action ID
    /// @return Unique identifier for this action
    /// @note Not constexpr because function pointer address is used
    [[nodiscard]] static action_id id() noexcept {
        static const action_id cached_id = compute_id();
        return cached_id;
    }

    /// @brief Get the function pointer
    /// @return The wrapped function pointer
    [[nodiscard]] static constexpr func_type function() noexcept {
        return Func;
    }

    /// @brief Invoke the function with given arguments
    /// @tparam Args Argument types
    /// @param args Function arguments
    /// @return Function result
    template <typename... Args>
    static decltype(auto) invoke(Args&&... args) {
        return Func(std::forward<Args>(args)...);
    }

    /// @brief Invoke the function with a tuple of arguments
    /// @param args_tuple Tuple containing arguments
    /// @return Function result
    static decltype(auto) invoke_tuple(const request_type& args_tuple) {
        return std::apply(Func, args_tuple);
    }

private:
    /// @brief Compute unique action ID from function signature
    /// @note Not constexpr because it uses reinterpret_cast
    static action_id compute_id() noexcept {
        // Hash based on:
        // 1. Return type hash
        // 2. Argument types hash
        // 3. Function pointer value (for uniqueness)

        action_id ret_hash = detail::type_hash<response_type>();

        action_id args_hash = []<typename... Ts>(std::tuple<Ts...>*) {
            return detail::tuple_type_hash<Ts...>();
        }(static_cast<request_type*>(nullptr));

        // Mix in the function address for uniqueness
        // (different functions with same signature get different IDs)
        const auto func_addr = reinterpret_cast<std::uintptr_t>(
            reinterpret_cast<void(*)()>(Func));
        action_id ptr_hash{};
        if constexpr (std::is_same_v<std::uintptr_t, action_id>) {
            ptr_hash = func_addr;
        } else {
            ptr_hash = static_cast<action_id>(func_addr);
        }

        return detail::hash_combine(
            detail::hash_combine(ret_hash, args_hash),
            ptr_hash);
    }
};

// ============================================================================
// Action Traits
// ============================================================================

/// @brief Check if a type is an action
template <typename T>
struct is_action : std::false_type {};

template <auto Func>
struct is_action<action<Func>> : std::true_type {};

/// @brief Helper that strips cv-qualifiers for is_action
template <typename T>
inline constexpr bool is_action_v = is_action<std::remove_cv_t<T>>::value;

/// @brief Concept for action types
template <typename T>
concept Action = is_action_v<T>;

/// @brief Extract action ID from an action type
template <Action A>
action_id get_action_id() noexcept {
    return A::id();
}

/// @brief Extract action ID from an action instance
template <auto Func>
action_id get_action_id(const action<Func>&) noexcept {
    return action<Func>::id();
}

}  // namespace dtl::remote

// ============================================================================
// Registration Macro
// ============================================================================

/// @brief Register an action with automatic name generation
/// @param func Function to register
/// @details Creates a constexpr action instance named action_<func>.
///
/// @par Example:
/// @code
/// int my_function(int x) { return x * 2; }
/// DTL_REGISTER_ACTION(my_function);
///
/// // Now available as:
/// auto id = dtl::remote::action_my_function.id();
/// @endcode
#define DTL_REGISTER_ACTION(func) \
    inline constexpr ::dtl::remote::action<&func> action_##func{}

/// @brief Register an action with a custom name
/// @param name Custom name for the action
/// @param func Function to register
#define DTL_REGISTER_ACTION_AS(name, func) \
    inline constexpr ::dtl::remote::action<&func> name{}
