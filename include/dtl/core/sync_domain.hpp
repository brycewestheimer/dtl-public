// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file sync_domain.hpp
/// @brief Dirty-state and synchronization domain model
/// @details Implements explicit dirty tracking and sync domains for
///          distributed containers, enforcing coherency requirements.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <atomic>
#include <bitset>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace dtl {

// ============================================================================
// Synchronization Domain Enumeration
// ============================================================================

/// @brief Synchronization domains for distributed data
/// @details Defines the scope within which data modifications are visible.
///          Used to track which portions of data may be stale.
enum class sync_domain : uint8_t {
    /// @brief No modifications pending
    /// @details Data is fully synchronized across all domains.
    clean = 0,

    /// @brief Local modifications only
    /// @details Data has been modified locally but not communicated.
    ///          Safe to read locally, may be stale elsewhere.
    local_dirty = 1,

    /// @brief Halo region modifications
    /// @details Halo exchange needed; boundary data may be stale.
    ///          Interior data is consistent.
    halo = 2,

    /// @brief Global modifications
    /// @details Data needs global synchronization.
    ///          Collective operation required before reads.
    global_dirty = 3
};

/// @brief Check if domain requires communication to sync
[[nodiscard]] constexpr bool requires_communication(sync_domain domain) noexcept {
    return domain != sync_domain::clean && domain != sync_domain::local_dirty;
}

/// @brief Check if domain is at least as dirty as another
[[nodiscard]] constexpr bool is_at_least_as_dirty(sync_domain a, sync_domain b) noexcept {
    return static_cast<uint8_t>(a) >= static_cast<uint8_t>(b);
}

/// @brief Get the dirtier of two domains
[[nodiscard]] constexpr sync_domain max_domain(sync_domain a, sync_domain b) noexcept {
    return static_cast<uint8_t>(a) > static_cast<uint8_t>(b) ? a : b;
}

// ============================================================================
// Dirty State Flags
// ============================================================================

/// @brief Bit flags for fine-grained dirty tracking
/// @details Used when more granularity than sync_domain is needed.
struct dirty_flags {
    /// @brief Local data modified
    bool local_modified : 1 = false;

    /// @brief Halo data stale (need halo exchange)
    bool halo_stale : 1 = false;

    /// @brief Remote data may differ (global sync needed)
    bool remote_stale : 1 = false;

    /// @brief Structure changed (resize, redistribute)
    bool structure_changed : 1 = false;

    /// @brief Check if any flag is set
    [[nodiscard]] bool any() const noexcept {
        return local_modified || halo_stale || remote_stale || structure_changed;
    }

    /// @brief Check if clean (no flags set)
    [[nodiscard]] bool is_clean() const noexcept {
        return !any();
    }

    /// @brief Clear all flags
    void clear() noexcept {
        local_modified = false;
        halo_stale = false;
        remote_stale = false;
        structure_changed = false;
    }

    /// @brief Convert to sync_domain
    [[nodiscard]] sync_domain to_domain() const noexcept {
        if (remote_stale || structure_changed) return sync_domain::global_dirty;
        if (halo_stale) return sync_domain::halo;
        if (local_modified) return sync_domain::local_dirty;
        return sync_domain::clean;
    }
};

// ============================================================================
// Sync State Tracker
// ============================================================================

/// @brief Thread-safe synchronization state tracker for containers
/// @details Tracks dirty state and provides sync operations.
///          Can be embedded in distributed containers.
class sync_state {
public:
    /// @brief Default constructor (clean state)
    sync_state() noexcept = default;

    /// @brief Get current sync domain
    [[nodiscard]] sync_domain domain() const noexcept {
        return domain_.load(std::memory_order_acquire);
    }

    /// @brief Check if clean
    [[nodiscard]] bool is_clean() const noexcept {
        return domain() == sync_domain::clean;
    }

    /// @brief Check if dirty
    [[nodiscard]] bool is_dirty() const noexcept {
        return !is_clean();
    }

    /// @brief Check if requires communication to sync
    [[nodiscard]] bool needs_communication() const noexcept {
        return requires_communication(domain());
    }

    /// @brief Mark local modification
    void mark_local_modified() noexcept {
        auto current = domain_.load(std::memory_order_relaxed);
        if (current == sync_domain::clean) {
            domain_.store(sync_domain::local_dirty, std::memory_order_release);
        }
    }

    /// @brief Mark halo as stale
    void mark_halo_stale() noexcept {
        auto current = domain_.load(std::memory_order_relaxed);
        if (current < sync_domain::halo) {
            domain_.store(sync_domain::halo, std::memory_order_release);
        }
    }

    /// @brief Mark as needing global sync
    void mark_global_dirty() noexcept {
        domain_.store(sync_domain::global_dirty, std::memory_order_release);
    }

    /// @brief Mark as clean (after sync operation)
    void mark_clean() noexcept {
        domain_.store(sync_domain::clean, std::memory_order_release);
    }

    /// @brief Mark halo as synced (after halo exchange)
    void mark_halo_synced() noexcept {
        auto current = domain_.load(std::memory_order_relaxed);
        if (current == sync_domain::halo) {
            domain_.store(sync_domain::local_dirty, std::memory_order_release);
        }
    }

    /// @brief Set domain directly
    void set_domain(sync_domain d) noexcept {
        domain_.store(d, std::memory_order_release);
    }

private:
    std::atomic<sync_domain> domain_{sync_domain::clean};
};

// ============================================================================
// Sync Guard (RAII for tracking modifications)
// ============================================================================

/// @brief RAII guard that marks container dirty on destruction
/// @tparam Container Container type with sync_state
/// @details Use when beginning a modification scope that should
///          mark the container dirty when complete.
template <typename Container>
class sync_guard {
public:
    /// @brief Construct guard for container
    explicit sync_guard(Container& container) noexcept
        : container_(&container) {}

    /// @brief Non-copyable
    sync_guard(const sync_guard&) = delete;
    sync_guard& operator=(const sync_guard&) = delete;

    /// @brief Movable
    sync_guard(sync_guard&& other) noexcept
        : container_(other.container_), committed_(other.committed_) {
        other.container_ = nullptr;
    }

    sync_guard& operator=(sync_guard&& other) noexcept {
        if (this != &other) {
            commit();
            container_ = other.container_;
            committed_ = other.committed_;
            other.container_ = nullptr;
        }
        return *this;
    }

    /// @brief Mark dirty on destruction
    ~sync_guard() {
        commit();
    }

    /// @brief Commit changes (mark dirty)
    void commit() noexcept {
        if (container_ && !committed_) {
            container_->mark_local_modified();
            committed_ = true;
        }
    }

    /// @brief Cancel without marking dirty
    void cancel() noexcept {
        committed_ = true;  // Pretend we committed to suppress mark
    }

private:
    Container* container_ = nullptr;
    bool committed_ = false;
};

// ============================================================================
// Stale Data Policy
// ============================================================================

/// @brief Policy for handling access to stale data
enum class stale_policy : uint8_t {
    /// @brief Allow access to stale data (user responsible for coherency)
    allow,

    /// @brief Warn about stale access (debug mode)
    warn,

    /// @brief Return error on stale access
    error,

    /// @brief Automatically sync before access
    auto_sync
};

/// @brief Result type for checked operations on potentially stale data
template <typename T>
using sync_result = result<T>;

// ============================================================================
// Sync Requirements Concept
// ============================================================================

/// @brief Concept for types that support sync state tracking
template <typename T>
concept Syncable = requires(T& t, const T& ct) {
    { ct.sync_state() } -> std::same_as<const sync_state&>;
    { t.sync_state() } -> std::same_as<sync_state&>;
    { t.is_dirty() } -> std::convertible_to<bool>;
    { t.is_clean() } -> std::convertible_to<bool>;
    { t.mark_clean() } -> std::same_as<void>;
};

/// @brief Concept for types that support explicit sync operations
template <typename T>
concept ExplicitlySyncable = Syncable<T> && requires(T& t) {
    { t.sync() } -> std::same_as<result<void>>;
    { t.sync_halo() } -> std::same_as<result<void>>;
};

// ============================================================================
// Checked Access Wrapper
// ============================================================================

/// @brief Wrapper that enforces sync state checks before access
/// @tparam Container Container type with sync support
/// @details Used to implement "refuse-by-default" semantics for stale data.
template <typename Container>
class checked_access {
public:
    using value_type = typename Container::value_type;
    using reference = typename Container::reference;

    /// @brief Construct with container and policy
    explicit checked_access(Container& container,
                           stale_policy policy = stale_policy::error)
        : container_(&container), policy_(policy) {}

    /// @brief Access local element with sync check
    /// @param local_idx Local index
    /// @return Result with reference or error if stale
    [[nodiscard]] sync_result<reference> local(size_type local_idx) {
        if (auto check = check_sync(); !check) {
            return check.error();
        }
        return container_->local(local_idx);
    }

    /// @brief Get local view with sync check
    [[nodiscard]] auto local_view() {
        check_sync_or_throw();
        return container_->local_view();
    }

private:
    result<void> check_sync() {
        if constexpr (Syncable<Container>) {
            if (container_->is_dirty()) {
                switch (policy_) {
                case stale_policy::allow:
                    return {};
                case stale_policy::warn:
                    // Would log warning in full implementation
                    return {};
                case stale_policy::error:
                    return make_error<void>(status_code::invalid_state,
                        "Container data is stale; call sync() first");
                case stale_policy::auto_sync:
                    if constexpr (ExplicitlySyncable<Container>) {
                        return container_->sync();
                    } else {
                        return make_error<void>(status_code::not_supported,
                            "Container does not support auto-sync");
                    }
                }
            }
        }
        return {};
    }

    void check_sync_or_throw() {
        auto result = check_sync();
        if (!result) {
            throw std::runtime_error(result.error().to_string());
        }
    }

    Container* container_;
    stale_policy policy_;
};

// ============================================================================
// Helper Functions
// ============================================================================

/// @brief Create a checked access wrapper for a container
template <typename Container>
[[nodiscard]] checked_access<Container> make_checked(
    Container& container,
    stale_policy policy = stale_policy::error) {
    return checked_access<Container>(container, policy);
}

/// @brief Require container to be clean, returning error otherwise
template <Syncable Container>
[[nodiscard]] result<void> require_clean(const Container& container) {
    if (container.is_dirty()) {
        return make_error<void>(status_code::invalid_state,
            "Operation requires clean container state");
    }
    return {};
}

/// @brief Require container to be at least locally consistent
template <Syncable Container>
[[nodiscard]] result<void> require_local_consistent(const Container& container) {
    auto domain = container.sync_state().domain();
    if (domain == sync_domain::global_dirty) {
        return make_error<void>(status_code::invalid_state,
            "Operation requires at least local consistency");
    }
    return {};
}

}  // namespace dtl
