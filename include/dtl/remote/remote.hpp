// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file remote.hpp
/// @brief Master include for DTL remote support
/// @details Provides single-header access to all remote types including
///          RPC, active messages, and progress management.
/// @since 0.1.0

#pragma once

// Action registration
#include <dtl/remote/action.hpp>
#include <dtl/remote/action_registry.hpp>

// Argument serialization
#include <dtl/remote/argument_pack.hpp>

// Request management
#include <dtl/remote/rpc_request.hpp>

// RPC interfaces
#include <dtl/remote/rpc.hpp>

// Active messages
#include <dtl/remote/active_message.hpp>

// Progress management
#include <dtl/remote/progress.hpp>

namespace dtl::remote {

// ============================================================================
// Remote Module Summary
// ============================================================================
//
// The remote module provides infrastructure for remote procedure calls and
// active messages in DTL. It follows the design principles:
//
// 1. **Compile-time Action Registration** (no runtime plugin registry)
//    - Actions are defined via `action<&func>` template
//    - Registration uses DTL_REGISTER_ACTION macro
//    - Action IDs are computed at compile-time from function signatures
//
// 2. **Manual Progress** (explicit)
//    - Remote operations require explicit progress calls
//    - Integrates with dtl::futures::progress_engine
//    - No background threads for message processing
//
// 3. **Type-safe RPC with Automatic Serialization**
//    - Uses existing dtl::serializer trait
//    - Arguments are packed into tuples and serialized
//    - Return values are automatically deserialized
//
// ============================================================================
// Quick Start
// ============================================================================
//
// 1. Define your remote functions:
//
//    int add(int a, int b) { return a + b; }
//    void notify(std::string msg) { /* ... */ }
//
// 2. Register them as actions:
//
//    DTL_REGISTER_ACTION(add);
//    DTL_REGISTER_ACTION(notify);
//
// 3. Make RPC calls:
//
//    // Asynchronous call
//    auto future = dtl::remote::call<&add>(target_rank, 10, 20);
//
//    // Drive progress until complete
//    while (!future.is_ready()) {
//        dtl::remote::make_all_progress();
//    }
//
//    int result = future.get();  // 30
//
//    // Fire-and-forget (for void functions)
//    dtl::remote::send<&notify>(target_rank, "Hello!");
//
// 4. Synchronous (blocking) calls:
//
//    auto result = dtl::remote::call_sync<&add>(target_rank, 10, 20);
//    if (result) {
//        std::cout << "Result: " << result.value() << "\n";
//    }
//
// ============================================================================
// Active Messages
// ============================================================================
//
// For lower-level control, use active messages:
//
//    // Define handler ID (can use any action or custom ID)
//    constexpr action_id my_handler = 12345;
//
//    // Register handler
//    dtl::remote::am_registry::instance().register_handler(my_handler,
//        [](rank_t source, const std::byte* payload, size_type size) {
//            // Process payload
//        });
//
//    // Send active message
//    std::vector<std::byte> data = /* ... */;
//    dtl::remote::send_am(target, my_handler, data);
//
// ============================================================================
// Progress Model
// ============================================================================
//
// DTL uses explicit progress (no background threads):
//
//    // Drive async operations
//    dtl::futures::make_progress();
//
//    // Drive remote message processing
//    dtl::remote::make_remote_progress();
//
//    // Or both together
//    dtl::remote::make_all_progress();
//
// ============================================================================

}  // namespace dtl::remote
