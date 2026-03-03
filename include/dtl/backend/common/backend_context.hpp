// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file backend_context.hpp
/// @brief RAII context manager for backends
/// @details Provides lifecycle management for backend resources.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/common/backend_traits.hpp>

#include <memory>
#include <functional>
#include <unordered_map>
#include <string>
#include <mutex>

namespace dtl {

// ============================================================================
// Context State
// ============================================================================

/// @brief State of a backend context
enum class context_state {
    uninitialized,  ///< Not yet initialized
    initializing,   ///< Initialization in progress
    active,         ///< Ready for use
    finalizing,     ///< Finalization in progress
    finalized       ///< Finalization complete
};

// ============================================================================
// Backend Context Base
// ============================================================================

/// @brief Base class for backend context managers
/// @details Provides RAII-style lifetime management for backend resources.
class backend_context_base {
public:
    /// @brief Virtual destructor
    virtual ~backend_context_base() = default;

    /// @brief Get context state
    [[nodiscard]] virtual context_state state() const noexcept = 0;

    /// @brief Check if context is active
    [[nodiscard]] bool is_active() const noexcept {
        return state() == context_state::active;
    }

    /// @brief Initialize the backend
    /// @return Result indicating success or error
    virtual result<void> initialize() = 0;

    /// @brief Finalize the backend
    /// @return Result indicating success or error
    virtual result<void> finalize() = 0;

    /// @brief Get backend name
    [[nodiscard]] virtual const char* backend_name() const noexcept = 0;

protected:
    /// @brief Protected default constructor
    backend_context_base() = default;

    /// @brief Non-copyable
    backend_context_base(const backend_context_base&) = delete;
    backend_context_base& operator=(const backend_context_base&) = delete;

    /// @brief Movable
    backend_context_base(backend_context_base&&) = default;
    backend_context_base& operator=(backend_context_base&&) = default;
};

// ============================================================================
// Backend Context Template
// ============================================================================

/// @brief Template for typed backend contexts
/// @tparam Backend Backend tag type
template <typename Backend>
class backend_context : public backend_context_base {
public:
    using backend_type = Backend;
    using traits_type = backend_traits<Backend>;

    /// @brief Default constructor (uninitialized)
    backend_context() : state_(context_state::uninitialized) {}

    /// @brief Destructor (auto-finalizes)
    ~backend_context() override {
        if (state_ == context_state::active) {
            finalize();
        }
    }

    /// @brief Move constructor (transfers ownership, leaves source finalized)
    backend_context(backend_context&& other) noexcept
        : backend_context_base(std::move(other)), state_(other.state_) {
        other.state_ = context_state::finalized;
    }

    /// @brief Move assignment (finalizes current, transfers from source)
    backend_context& operator=(backend_context&& other) noexcept {
        if (this != &other) {
            if (state_ == context_state::active) {
                finalize();
            }
            backend_context_base::operator=(std::move(other));
            state_ = other.state_;
            other.state_ = context_state::finalized;
        }
        return *this;
    }

    /// @brief Non-copyable (backend resources are not shareable)
    backend_context(const backend_context&) = delete;
    backend_context& operator=(const backend_context&) = delete;

    /// @brief Get context state
    [[nodiscard]] context_state state() const noexcept override {
        return state_;
    }

    /// @brief Initialize the backend
    /// @details Base class provides a default no-op initialization that
    ///          transitions the context from uninitialized to active.
    ///          Concrete backend specializations (e.g., MPI, CUDA, NCCL)
    ///          should override this method to perform backend-specific
    ///          resource acquisition such as library initialization,
    ///          device selection, or communicator creation.
    /// @return Result indicating success or error
    result<void> initialize() override {
        if (state_ != context_state::uninitialized) {
            return make_error(status_code::invalid_state,
                             "Context already initialized");
        }

        state_ = context_state::initializing;
        // Default no-op: concrete backends override to perform
        // library initialization, device selection, etc.
        state_ = context_state::active;
        return {};
    }

    /// @brief Finalize the backend
    /// @details Base class provides a default no-op finalization that
    ///          transitions the context from active to finalized.
    ///          Concrete backend specializations should override this
    ///          method to release backend-specific resources such as
    ///          communicator handles, device contexts, or library state.
    ///          The destructor calls finalize() automatically if the
    ///          context is still active.
    /// @return Result indicating success or error
    result<void> finalize() override {
        if (state_ != context_state::active) {
            return make_error(status_code::invalid_state,
                             "Context not active");
        }

        state_ = context_state::finalizing;
        // Default no-op: concrete backends override to release
        // communicator handles, device contexts, library state, etc.
        state_ = context_state::finalized;
        return {};
    }

    /// @brief Get backend name
    [[nodiscard]] const char* backend_name() const noexcept override {
        return traits_type::name;
    }

private:
    context_state state_;
};

// ============================================================================
// Scoped Context
// ============================================================================

/// @brief RAII guard for backend context lifetime
/// @tparam Context Context type
template <typename Context>
class scoped_context {
public:
    /// @brief Construct and initialize context
    scoped_context() {
        auto result = context_.initialize();
        if (!result) {
            throw std::runtime_error("Failed to initialize context");
        }
    }

    /// @brief Construct with existing context (takes ownership)
    explicit scoped_context(Context&& ctx) : context_(std::move(ctx)) {
        if (!context_.is_active()) {
            auto result = context_.initialize();
            if (!result) {
                throw std::runtime_error("Failed to initialize context");
            }
        }
    }

    /// @brief Destructor finalizes context
    ~scoped_context() {
        if (context_.is_active()) {
            context_.finalize();
        }
    }

    /// @brief Non-copyable
    scoped_context(const scoped_context&) = delete;
    scoped_context& operator=(const scoped_context&) = delete;

    /// @brief Movable
    scoped_context(scoped_context&&) = default;
    scoped_context& operator=(scoped_context&&) = default;

    /// @brief Access the context
    [[nodiscard]] Context& get() noexcept { return context_; }
    [[nodiscard]] const Context& get() const noexcept { return context_; }

    /// @brief Pointer-like access
    [[nodiscard]] Context* operator->() noexcept { return &context_; }
    [[nodiscard]] const Context* operator->() const noexcept { return &context_; }

private:
    Context context_;
};

// ============================================================================
// Context Registry
// ============================================================================

/// @brief Global registry for active backend contexts
class context_registry {
public:
    /// @brief Get the singleton registry
    [[nodiscard]] static context_registry& instance() {
        static context_registry registry;
        return registry;
    }

    /// @brief Register a context
    /// @param name Context name (must be unique)
    /// @param ctx Context pointer (non-owning)
    void register_context(const char* name, backend_context_base* ctx) {
        std::lock_guard<std::mutex> lock(mtx_);
        contexts_[std::string(name)] = ctx;
    }

    /// @brief Unregister a context
    /// @param name Context name
    void unregister_context(const char* name) {
        std::lock_guard<std::mutex> lock(mtx_);
        contexts_.erase(std::string(name));
    }

    /// @brief Get a registered context
    /// @param name Context name
    /// @return Pointer to context or nullptr if not found
    [[nodiscard]] backend_context_base* get_context(const char* name) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = contexts_.find(std::string(name));
        if (it != contexts_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /// @brief Check if a context is registered
    /// @param name Context name
    /// @return true if registered
    [[nodiscard]] bool contains(const char* name) {
        std::lock_guard<std::mutex> lock(mtx_);
        return contexts_.find(std::string(name)) != contexts_.end();
    }

    /// @brief Get number of registered contexts
    [[nodiscard]] size_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return contexts_.size();
    }

    /// @brief Finalize all registered contexts
    /// @details Iterates and calls finalize() on each active context.
    ///          Clears the registry afterward.
    void finalize_all() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& [name, ctx] : contexts_) {
            if (ctx && ctx->is_active()) {
                ctx->finalize();
            }
        }
        contexts_.clear();
    }

private:
    context_registry() = default;
    ~context_registry() { finalize_all(); }

    // Non-copyable
    context_registry(const context_registry&) = delete;
    context_registry& operator=(const context_registry&) = delete;

    std::mutex mtx_;
    std::unordered_map<std::string, backend_context_base*> contexts_;
};

// ============================================================================
// Helper Functions
// ============================================================================

/// @brief Create and initialize a backend context
/// @tparam Backend Backend tag type
/// @return Result containing initialized context
template <typename Backend>
[[nodiscard]] result<backend_context<Backend>> make_context() {
    backend_context<Backend> ctx;
    auto init_result = ctx.initialize();
    if (!init_result) {
        return make_error(init_result.error().code(),
                         init_result.error().message());
    }
    return ctx;
}

/// @brief Create a scoped context
/// @tparam Backend Backend tag type
/// @return Scoped context guard
template <typename Backend>
[[nodiscard]] scoped_context<backend_context<Backend>> make_scoped_context() {
    return scoped_context<backend_context<Backend>>();
}

}  // namespace dtl
