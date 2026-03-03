// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file continuation.hpp
/// @brief Future continuation chaining (.then()) and combinators
/// @details Provides continuation support for distributed futures using
///          progress-based execution instead of detached threads.
///
///          **Execution model (ADR: inline on progress engine):**
///          Continuations registered via `.then()` execute inline within the
///          progress engine's poll loop. When the input future becomes ready,
///          the continuation callback is invoked directly on the polling thread.
///
///          **Implications:**
///          - Lightweight callbacks (~microseconds) are executed efficiently
///            with no scheduling overhead.
///          - Long-running callbacks WILL block the progress engine, preventing
///            other futures and continuations from advancing. Keep `.then()`
///            callbacks short; offload heavy work to a separate thread or use
///            `callback_executor` (see callback_executor.hpp) directly.
///
///          **Design rationale (Option B — Document):**
///          The `callback_executor` class provides isolated execution on a
///          dedicated thread/pool. However, integrating it transparently into
///          `.then()` would add thread-hop latency for the common case of fast
///          callbacks and introduce exception-forwarding complexity across
///          thread boundaries. The inline model was retained for simplicity
///          and performance. Users requiring isolation can enqueue work onto
///          `global_callback_executor()` from within their `.then()` callback.
///
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>
#include <dtl/futures/completion.hpp>

#include <functional>
#include <thread>
#include <type_traits>
#include <tuple>
#include <variant>
#include <vector>

namespace dtl::futures {

// when_any_result is already defined in backend/concepts/distributed_future.hpp

// ============================================================================
// Continuation Traits
// ============================================================================

/// @brief Traits for continuation result types
template <typename F, typename T>
struct continuation_result {
    using type = std::invoke_result_t<F, T>;
};

/// @brief Specialization for void input
template <typename F>
struct continuation_result<F, void> {
    using type = std::invoke_result_t<F>;
};

/// @brief Helper alias
template <typename F, typename T>
using continuation_result_t = typename continuation_result<F, T>::type;

// ============================================================================
// Continuation Implementation
// ============================================================================

namespace detail {

/// @brief Execute continuation when future is ready
/// @tparam T Input type
/// @tparam F Continuation function type
/// @tparam R Result type
template <typename T, typename F, typename R>
void execute_continuation(
    std::shared_ptr<shared_state<T>> input_state,
    std::shared_ptr<shared_state<R>> output_state,
    F func) {
    try {
        input_state->wait();

        if (input_state->has_error()) {
            // Propagate error
            output_state->set_error(status(status_code::operation_failed,
                                           no_rank, "Continuation input failed"));
        } else {
            if constexpr (std::is_void_v<T>) {
                if constexpr (std::is_void_v<R>) {
                    func();
                    output_state->set_value();
                } else {
                    output_state->set_value(func());
                }
            } else {
                if constexpr (std::is_void_v<R>) {
                    func(input_state->get());
                    output_state->set_value();
                } else {
                    output_state->set_value(func(input_state->get()));
                }
            }
        }
    } catch (const std::exception& e) {
        output_state->set_error(status(status_code::operation_failed, no_rank, e.what()));
    } catch (...) {
        output_state->set_error(status(status_code::unknown_error, no_rank, "Unknown exception"));
    }
}

/// @brief Progress-based continuation executor
/// @details Registers with the progress engine and executes when the input is ready
template <typename T, typename F, typename R>
class continuation_executor {
public:
    continuation_executor(
        std::shared_ptr<shared_state<T>> input,
        std::shared_ptr<shared_state<R>> output,
        F func)
        : input_(std::move(input))
        , output_(std::move(output))
        , func_(std::move(func))
        , executed_(false) {}

    /// @brief Poll for completion (returns false when done)
    bool operator()() {
        if (executed_) return false;

        if (input_->is_ready()) {
            execute_continuation<T, F, R>(input_, output_, std::move(func_));
            executed_ = true;
            return false;  // Done
        }
        return true;  // Still pending
    }

private:
    std::shared_ptr<shared_state<T>> input_;
    std::shared_ptr<shared_state<R>> output_;
    F func_;
    bool executed_;
};

}  // namespace detail

// ============================================================================
// Then Implementation for distributed_future<T>
// ============================================================================

template <typename T>
template <typename F>
auto distributed_future<T>::then(F&& func) {
    using R = continuation_result_t<F, T>;

    auto output_state = std::make_shared<shared_state<R>>();
    auto input_state = state_;

    if (!input_state) {
        output_state->set_error(status(status_code::invalid_state, no_rank, "Invalid future"));
        return distributed_future<R>(std::move(output_state));
    }

    // If already ready, execute immediately
    if (input_state->is_ready()) {
        detail::execute_continuation<T, std::decay_t<F>, R>(
            input_state, output_state, std::forward<F>(func));
    } else {
        // Register with progress engine for deferred execution
        auto executor = std::make_shared<detail::continuation_executor<T, std::decay_t<F>, R>>(
            input_state, output_state, std::forward<F>(func));

        futures::progress_engine::instance().register_callback([executor]() {
            return (*executor)();
        });
    }

    return distributed_future<R>(std::move(output_state));
}

template <typename F>
auto distributed_future<void>::then(F&& func) {
    using R = continuation_result_t<F, void>;

    auto output_state = std::make_shared<shared_state<R>>();
    auto input_state = state_;

    if (!input_state) {
        output_state->set_error(status(status_code::invalid_state, no_rank, "Invalid future"));
        return distributed_future<R>(std::move(output_state));
    }

    // If already ready, execute immediately
    if (input_state->is_ready()) {
        detail::execute_continuation<void, std::decay_t<F>, R>(
            input_state, output_state, std::forward<F>(func));
    } else {
        // Register with progress engine for deferred execution
        auto executor = std::make_shared<detail::continuation_executor<void, std::decay_t<F>, R>>(
            input_state, output_state, std::forward<F>(func));

        futures::progress_engine::instance().register_callback([executor]() {
            return (*executor)();
        });
    }

    return distributed_future<R>(std::move(output_state));
}

// ============================================================================
// When_All Implementation
// ============================================================================

namespace detail {

/// @brief Helper to get value from future (non-void)
template <typename T>
    requires (!std::is_void_v<T>)
auto get_future_value(distributed_future<T>& f) -> T {
    return f.get();
}

/// @brief Helper to get value from void future
inline auto get_future_value(distributed_future<void>& f) -> std::monostate {
    f.get();
    return std::monostate{};
}

/// @brief Helper to build when_all result tuple
template <typename... Ts, size_type... Is>
auto build_when_all_result(std::tuple<distributed_future<Ts>...>& futures,
                           std::index_sequence<Is...>) {
    return std::make_tuple(std::get<Is>(futures).get()...);
}

/// @brief Type to use in tuple for void futures
template <typename T>
struct when_all_value_type {
    using type = T;
};

template <>
struct when_all_value_type<void> {
    using type = std::monostate;
};

template <typename T>
using when_all_value_type_t = typename when_all_value_type<T>::type;

}  // namespace detail

namespace detail {

/// @brief Helper to check if all futures in a tuple are ready
template <typename Tuple, size_type... Is>
bool all_futures_ready(Tuple& futures_tuple, std::index_sequence<Is...>) {
    return (std::get<Is>(futures_tuple).is_ready() && ...);
}

/// @brief Helper to get results from all futures in a tuple
template <typename ResultTuple, typename FutureTuple, size_type... Is>
ResultTuple collect_future_results(FutureTuple& futures_tuple, std::index_sequence<Is...>) {
    return ResultTuple{
        [&]() {
            using future_type = std::tuple_element_t<Is, std::decay_t<FutureTuple>>;
            if constexpr (std::is_void_v<typename future_type::value_type>) {
                std::get<Is>(futures_tuple).get();
                return std::monostate{};
            } else {
                return std::get<Is>(futures_tuple).get();
            }
        }()...};
}

}  // namespace detail

/// @brief Wait for all futures to complete (variadic, heterogeneous types)
/// @tparam Futures Future types
/// @param futures Futures to wait for
/// @return Future containing tuple of results
template <typename... Futures>
    requires (sizeof...(Futures) > 0)
auto when_all(Futures&&... futures) {
    // Determine result tuple type
    using result_type = std::tuple<
        detail::when_all_value_type_t<typename std::decay_t<Futures>::value_type>...>;
    using futures_tuple_type = std::tuple<std::decay_t<Futures>...>;

    auto state = std::make_shared<shared_state<result_type>>();

    // Store futures in a shared_ptr to avoid moving issues in lambda
    auto futures_ptr = std::make_shared<futures_tuple_type>(std::forward<Futures>(futures)...);

    constexpr size_type count = sizeof...(Futures);

    // Register progress callback
    futures::progress_engine::instance().register_callback(
        [state, futures_ptr]() mutable {
            // Check readiness
            if (!detail::all_futures_ready(*futures_ptr, std::make_index_sequence<count>{})) {
                return true;  // Keep polling
            }

            // All ready - collect results
            try {
                auto results = detail::collect_future_results<result_type>(
                    *futures_ptr, std::make_index_sequence<count>{});
                state->set_value(std::move(results));
            } catch (const std::exception& e) {
                state->set_error(status(status_code::operation_failed, no_rank, e.what()));
            } catch (...) {
                state->set_error(status(status_code::unknown_error, no_rank, "Unknown exception"));
            }
            return false;  // Done
        });

    return distributed_future<result_type>(std::move(state));
}

/// @brief Wait for all futures in a vector to complete (homogeneous type)
/// @tparam T Value type of futures
/// @param futures Vector of futures to wait for
/// @return Future containing vector of results
template <typename T>
auto when_all(std::vector<distributed_future<T>> futures) {
    using result_type = std::vector<T>;

    auto state = std::make_shared<shared_state<result_type>>();

    if (futures.empty()) {
        state->set_value(result_type{});
        return distributed_future<result_type>(std::move(state));
    }

    auto futures_ptr = std::make_shared<std::vector<distributed_future<T>>>(std::move(futures));

    futures::progress_engine::instance().register_callback(
        [state, futures_ptr]() mutable {
            // Check if all futures are ready
            bool all_ready = true;
            for (auto& f : *futures_ptr) {
                if (!f.is_ready()) {
                    all_ready = false;
                    break;
                }
            }

            if (all_ready) {
                try {
                    result_type results;
                    results.reserve(futures_ptr->size());
                    for (auto& f : *futures_ptr) {
                        results.push_back(f.get());
                    }
                    state->set_value(std::move(results));
                } catch (const std::exception& e) {
                    state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                } catch (...) {
                    state->set_error(status(status_code::unknown_error, no_rank, "Unknown exception"));
                }
                return false;  // Done
            }
            return true;  // Still waiting
        });

    return distributed_future<result_type>(std::move(state));
}

/// @brief Wait for all void futures in a vector to complete
/// @param futures Vector of void futures
/// @return Void future that completes when all complete
inline auto when_all(std::vector<distributed_future<void>> futures) {
    auto state = std::make_shared<shared_state<void>>();

    if (futures.empty()) {
        state->set_value();
        return distributed_future<void>(std::move(state));
    }

    auto futures_ptr = std::make_shared<std::vector<distributed_future<void>>>(std::move(futures));

    futures::progress_engine::instance().register_callback(
        [state, futures_ptr]() mutable {
            bool all_ready = true;
            for (auto& f : *futures_ptr) {
                if (!f.is_ready()) {
                    all_ready = false;
                    break;
                }
            }

            if (all_ready) {
                try {
                    for (auto& f : *futures_ptr) {
                        f.get();  // May throw
                    }
                    state->set_value();
                } catch (const std::exception& e) {
                    state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                } catch (...) {
                    state->set_error(status(status_code::unknown_error, no_rank, "Unknown exception"));
                }
                return false;
            }
            return true;
        });

    return distributed_future<void>(std::move(state));
}

// ============================================================================
// When_Any Implementation
// ============================================================================

/// @brief Wait for any future to complete
/// @tparam T Value type (all futures must have same type)
/// @param futures Vector of futures (taken by value to avoid use-after-free)
/// @return Future with first completed value and its index
template <typename T>
auto when_any(std::vector<distributed_future<T>> futures) {
    using result_type = when_any_result<T>;

    auto state = std::make_shared<shared_state<result_type>>();

    if (futures.empty()) {
        state->set_error(status(status_code::invalid_argument, no_rank, "when_any: empty futures vector"));
        return distributed_future<result_type>(std::move(state));
    }

    auto completed = std::make_shared<std::atomic<bool>>(false);
    auto shared_futures = std::make_shared<std::vector<distributed_future<T>>>(std::move(futures));

    futures::progress_engine::instance().register_callback(
        [state, shared_futures, completed]() mutable {
            if (completed->load()) return false;

            for (size_type i = 0; i < shared_futures->size(); ++i) {
                if ((*shared_futures)[i].is_ready()) {
                    bool expected = false;
                    if (completed->compare_exchange_strong(expected, true)) {
                        try {
                            result_type result;
                            result.index = i;
                            result.value = (*shared_futures)[i].get();
                            state->set_value(std::move(result));
                        } catch (const std::exception& e) {
                            state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                        }
                    }
                    return false;
                }
            }
            return true;  // Keep polling
        });

    return distributed_future<result_type>(std::move(state));
}

/// @brief Wait for any void future to complete
/// @param futures Vector of void futures (taken by value to avoid use-after-free)
/// @return Future with index of first completed
inline auto when_any(std::vector<distributed_future<void>> futures) {
    using result_type = when_any_result<void>;

    auto state = std::make_shared<shared_state<result_type>>();

    if (futures.empty()) {
        state->set_error(status(status_code::invalid_argument, no_rank, "when_any: empty futures vector"));
        return distributed_future<result_type>(std::move(state));
    }

    auto completed = std::make_shared<std::atomic<bool>>(false);
    auto shared_futures = std::make_shared<std::vector<distributed_future<void>>>(std::move(futures));

    futures::progress_engine::instance().register_callback(
        [state, shared_futures, completed]() mutable {
            if (completed->load()) return false;

            for (size_type i = 0; i < shared_futures->size(); ++i) {
                if ((*shared_futures)[i].is_ready()) {
                    bool expected = false;
                    if (completed->compare_exchange_strong(expected, true)) {
                        try {
                            (*shared_futures)[i].get();  // May throw
                            result_type result;
                            result.index = i;
                            state->set_value(std::move(result));
                        } catch (const std::exception& e) {
                            state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                        }
                    }
                    return false;
                }
            }
            return true;
        });

    return distributed_future<result_type>(std::move(state));
}

// ============================================================================
// Error Recovery
// ============================================================================

/// @brief Add error recovery handler to a future
/// @tparam T Value type
/// @tparam Handler Error handler function type
/// @param future The future to add recovery to
/// @param handler Function called on error: (const status&) -> T
/// @return Future that uses handler result on error
template <typename T, typename Handler>
    requires std::invocable<Handler, const status&>
auto on_error(distributed_future<T> future, Handler&& handler) {
    using handler_result = std::invoke_result_t<Handler, const status&>;
    static_assert(std::is_convertible_v<handler_result, T> || std::is_same_v<handler_result, T>,
                  "on_error handler must return a type convertible to T");

    auto state = std::make_shared<shared_state<T>>();
    auto input_state = future.get_state();

    if (!input_state) {
        try {
            state->set_value(handler(status(status_code::invalid_state, no_rank, "Invalid future")));
        } catch (const std::exception& e) {
            state->set_error(status(status_code::operation_failed, no_rank, e.what()));
        }
        return distributed_future<T>(std::move(state));
    }

    if (input_state->is_ready()) {
        if (input_state->has_error()) {
            try {
                // Get the error and call handler
                auto result = input_state->get_result();
                state->set_value(handler(result.error()));
            } catch (const std::exception& e) {
                state->set_error(status(status_code::operation_failed, no_rank, e.what()));
            }
        } else {
            try {
                state->set_value(input_state->get());
            } catch (const std::exception& e) {
                state->set_error(status(status_code::operation_failed, no_rank, e.what()));
            }
        }
    } else {
        futures::progress_engine::instance().register_callback(
            [state, input_state, h = std::forward<Handler>(handler)]() mutable {
                if (!input_state->is_ready()) return true;

                if (input_state->has_error()) {
                    try {
                        auto result = input_state->get_result();
                        state->set_value(h(result.error()));
                    } catch (const std::exception& e) {
                        state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                    }
                } else {
                    try {
                        state->set_value(input_state->get());
                    } catch (const std::exception& e) {
                        state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                    }
                }
                return false;
            });
    }

    return distributed_future<T>(std::move(state));
}

// ============================================================================
// Utility Functions
// ============================================================================

namespace detail {

/// @brief Chain helper - base case (single function)
template <typename T, typename F>
auto chain_impl(distributed_future<T> future, F&& func) {
    return future.then(std::forward<F>(func));
}

/// @brief Chain helper - recursive case
template <typename T, typename F, typename... Rest>
    requires (sizeof...(Rest) > 0)
auto chain_impl(distributed_future<T> future, F&& func, Rest&&... rest) {
    return chain_impl(future.then(std::forward<F>(func)), std::forward<Rest>(rest)...);
}

}  // namespace detail

/// @brief Chain multiple continuations
/// @tparam T Initial type
/// @tparam Fs Continuation function types
/// @param future Initial future
/// @param funcs Continuation functions
/// @return Final future
template <typename T, typename... Fs>
    requires (sizeof...(Fs) > 0)
auto chain(distributed_future<T> future, Fs&&... funcs) {
    return detail::chain_impl(std::move(future), std::forward<Fs>(funcs)...);
}

/// @brief Map a function over a future
/// @tparam T Value type
/// @tparam F Function type
/// @param future Input future
/// @param func Mapping function
/// @return Mapped future
template <typename T, typename F>
auto fmap(distributed_future<T> future, F&& func) {
    return future.then(std::forward<F>(func));
}

/// @brief Flatten nested futures
/// @tparam T Inner value type
/// @param future Nested future
/// @return Flattened future
template <typename T>
distributed_future<T> flatten(distributed_future<distributed_future<T>> future) {
    auto state = std::make_shared<shared_state<T>>();

    auto outer_state = future.get_state();

    if (!outer_state) {
        state->set_error(status(status_code::invalid_state, no_rank, "Invalid outer future"));
        return distributed_future<T>(std::move(state));
    }

    futures::progress_engine::instance().register_callback(
        [state, outer_state]() mutable {
            if (!outer_state->is_ready()) return true;

            try {
                auto inner = outer_state->get();
                if (!inner.valid()) {
                    state->set_error(status(status_code::invalid_state, no_rank, "Invalid inner future"));
                    return false;
                }

                if (inner.is_ready()) {
                    try {
                        state->set_value(inner.get());
                    } catch (const std::exception& e) {
                        state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                    }
                    return false;
                }

                // Inner not ready yet - re-register
                futures::progress_engine::instance().register_callback(
                    [state, inner_future = std::move(inner)]() mutable {
                        if (!inner_future.is_ready()) return true;
                        try {
                            state->set_value(inner_future.get());
                        } catch (const std::exception& e) {
                            state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                        }
                        return false;
                    });
                return false;
            } catch (const std::exception& e) {
                state->set_error(status(status_code::operation_failed, no_rank, e.what()));
                return false;
            }
        });

    return distributed_future<T>(std::move(state));
}

/// @brief Monadic bind (flatMap)
/// @tparam T Input value type
/// @tparam F Function returning future
/// @param future Input future
/// @param func Function to apply
/// @return Result future
template <typename T, typename F>
auto flat_map(distributed_future<T> future, F&& func) {
    return flatten(future.then(std::forward<F>(func)));
}

}  // namespace dtl::futures
