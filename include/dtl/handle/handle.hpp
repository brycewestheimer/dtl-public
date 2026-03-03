// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file handle.hpp
/// @brief Public runtime/context/communicator handle contracts
/// @since 0.1.0

#pragma once

#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/runtime/runtime_registry.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace dtl::handle {

class runtime_handle {
public:
    runtime_handle() = default;

    runtime_handle(std::shared_ptr<const std::atomic<std::uint64_t>> token,
                   std::uint64_t generation) noexcept
        : token_(std::move(token))
        , generation_(generation) {}

    [[nodiscard]] static runtime_handle current() {
        auto& registry = runtime::runtime_registry::instance();
        return runtime_handle{registry.lifetime_generation_token(),
                              registry.lifetime_generation()};
    }

    [[nodiscard]] bool valid() const noexcept {
        if (!token_) {
            return true;
        }
        return token_->load(std::memory_order_acquire) == generation_;
    }

    [[nodiscard]] bool expired() const noexcept {
        return !valid();
    }

    [[nodiscard]] std::uint64_t generation() const noexcept {
        return generation_;
    }

private:
    std::shared_ptr<const std::atomic<std::uint64_t>> token_;
    std::uint64_t generation_ = 0;
};

class comm_handle {
public:
    using barrier_fn = std::function<result<void>()>;

    comm_handle() = default;

    comm_handle(rank_t rank, rank_t size, barrier_fn barrier, runtime_handle runtime)
        : rank_(rank)
        , size_(size)
        , barrier_(std::move(barrier))
        , runtime_(std::move(runtime)) {}

    [[nodiscard]] static comm_handle local(runtime_handle runtime = runtime_handle::current()) {
        return comm_handle{0, 1, nullptr, std::move(runtime)};
    }

    [[nodiscard]] static comm_handle unbound(rank_t rank, rank_t size,
                                             runtime_handle runtime = runtime_handle::current()) {
        return comm_handle{rank, size, nullptr, std::move(runtime)};
    }

    [[nodiscard]] bool valid() const noexcept {
        if (size_ <= 0 || rank_ < 0 || rank_ >= size_) {
            return false;
        }
        return runtime_.valid();
    }

    [[nodiscard]] bool has_collective_path() const noexcept {
        return size_ <= 1 || static_cast<bool>(barrier_);
    }

    [[nodiscard]] rank_t rank() const noexcept {
        return rank_;
    }

    [[nodiscard]] rank_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] const runtime_handle& runtime() const noexcept {
        return runtime_;
    }

    [[nodiscard]] result<void> barrier() const {
        if (!runtime_.valid()) {
            return result<void>::failure(status{status_code::invalid_state, no_rank,
                                                "Runtime is finalized; communicator handle expired"});
        }

        if (size_ <= 1) {
            return result<void>::success();
        }

        if (!barrier_) {
            return result<void>::failure(status{status_code::invalid_state, rank_,
                                                "Multi-rank operation requires a valid communicator handle"});
        }

        return barrier_();
    }

private:
    rank_t rank_ = 0;
    rank_t size_ = 0;
    barrier_fn barrier_;
    runtime_handle runtime_{};
};

class context_handle {
public:
    context_handle() = default;

    context_handle(comm_handle comm, runtime_handle runtime,
                   determinism_options determinism = determinism_options{})
        : comm_(std::move(comm))
        , runtime_(std::move(runtime))
        , determinism_(determinism) {}

    [[nodiscard]] bool valid() const noexcept {
        return runtime_.valid() && comm_.valid();
    }

    [[nodiscard]] const comm_handle& communicator() const noexcept {
        return comm_;
    }

    [[nodiscard]] const runtime_handle& runtime() const noexcept {
        return runtime_;
    }

    [[nodiscard]] const determinism_options& determinism() const noexcept {
        return determinism_;
    }

private:
    comm_handle comm_{};
    runtime_handle runtime_{};
    determinism_options determinism_{};
};

template <typename Ctx>
[[nodiscard]] inline comm_handle make_comm_handle(const Ctx& ctx) {
    if constexpr (requires { { ctx.handle() } -> std::same_as<context_handle>; }) {
        return ctx.handle().communicator();
    } else if constexpr (requires { { ctx.communicator_handle() } -> std::same_as<comm_handle>; }) {
        return ctx.communicator_handle();
    } else if constexpr (requires(const Ctx& c) {
                                     { c.rank() } -> std::convertible_to<rank_t>;
                                     { c.size() } -> std::convertible_to<rank_t>;
                                 }) {
        return comm_handle::unbound(static_cast<rank_t>(ctx.rank()),
                                    static_cast<rank_t>(ctx.size()),
                                    runtime_handle::current());
    } else {
        return comm_handle::local(runtime_handle::current());
    }
}

}  // namespace dtl::handle
