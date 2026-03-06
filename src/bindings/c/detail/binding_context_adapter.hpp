// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file binding_context_adapter.hpp
/// @brief Adapter wrapping dtl_context_s* to satisfy C++ context concepts
/// @since 0.1.0

#pragma once

#include "../dtl_internal.hpp"
#include <dtl/core/types.hpp>
#include <dtl/core/config.hpp>

namespace dtl::c::detail {

/// @brief Lightweight cuda_domain adapter for the binding context
///
/// Satisfies the requirements of ctx.get<cuda_domain>():
///   - valid()     -> bool
///   - device_id() -> int
struct binding_cuda_domain {
    int dev_id;

    [[nodiscard]] constexpr bool valid() const noexcept { return dev_id >= 0; }
    [[nodiscard]] constexpr int device_id() const noexcept { return dev_id; }
};

/// @brief Context adapter wrapping dtl_context_s* for C++ container construction
///
/// Satisfies the distributed_vector context concept:
///   requires(const Ctx& c) {
///       { c.rank() } -> std::convertible_to<rank_t>;
///       { c.size() } -> std::convertible_to<rank_t>;
///   }
///
/// Also supports GPU domain extraction via get<>/has<> template methods,
/// which are used by device_only_runtime and unified_memory placements.
class binding_context {
public:
    explicit binding_context(const dtl_context_s* ctx) noexcept
        : rank_(ctx->rank)
        , size_(ctx->size)
        , device_id_(ctx->device_id)
        , domain_flags_(ctx->domain_flags) {}

    [[nodiscard]] dtl::rank_t rank() const noexcept { return rank_; }
    [[nodiscard]] dtl::rank_t size() const noexcept { return size_; }

    /// @brief Check if the context has a GPU domain
    template <typename Domain>
    [[nodiscard]] bool has() const noexcept {
        return (domain_flags_ & dtl_context_s::HAS_CUDA) && device_id_ >= 0;
    }

    /// @brief Get the GPU domain adapter
    template <typename Domain>
    [[nodiscard]] binding_cuda_domain get() const noexcept {
        return binding_cuda_domain{device_id_};
    }

private:
    dtl::rank_t rank_;
    dtl::rank_t size_;
    int device_id_;
    uint32_t domain_flags_;
};

}  // namespace dtl::c::detail
