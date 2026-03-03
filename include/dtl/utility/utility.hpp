// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file utility.hpp
/// @brief Master include for DTL utility module
/// @details Provides single-header access to all utility types.
/// @since 0.1.0

#pragma once

// Environment management
#include <dtl/utility/environment.hpp>

// Debugging utilities
#include <dtl/utility/debugging.hpp>

namespace dtl {

// ============================================================================
// Utility Module Summary
// ============================================================================
//
// The utility module provides support functionality for DTL applications,
// including environment management, debugging, and diagnostics.
//
// ============================================================================
// Environment Management (Legacy)
// ============================================================================
//
// NOTE: The classes in this module are deprecated. Use dtl::environment from
// <dtl/core/environment.hpp> for modern environment lifecycle management.
//
// Legacy RAII management (dtl::legacy:: namespace):
//
// @code
// #include <dtl/core/environment.hpp>
//
// int main() {
//     // Modern RAII environment initialization
//     dtl::environment env;
//
//     // Query backend availability (instance method)
//     if (env.has_mpi()) { /* ... */ }
//
//     // Environment automatically finalized when env goes out of scope
//     return 0;
// }
// @endcode
//
// ============================================================================
// Global Environment Access
// ============================================================================
//
// Access environment info from anywhere after initialization:
//
// @code
// // In any function after environment is initialized
// if (dtl::environment_initialized()) {
//     auto rank = dtl::world_rank();
//     auto size = dtl::world_size();
//
//     if (dtl::is_root()) {
//         // Root-only operations
//     }
// }
// @endcode
//
// ============================================================================
// Debug Output
// ============================================================================
//
// Configurable debug output with levels:
//
// @code
// // Set debug level
// dtl::set_debug_level(dtl::debug_level::debug);
// dtl::set_debug_rank(dtl::world_rank());
//
// // Output at different levels
// dtl::debug_error("Error message");    // Always shown
// dtl::debug_warning("Warning");        // level >= warning
// dtl::debug_info("Info");              // level >= info
// dtl::debug_msg("Debug message");      // level >= debug
// dtl::debug_trace("Trace message");    // level >= trace
//
// // Root-only output
// dtl::debug_root("Only from rank 0");
//
// // Rank-specific output
// dtl::debug_rank(2, "Only from rank 2");
// @endcode
//
// ============================================================================
// Assertions
// ============================================================================
//
// DTL provides assertion macros with source location tracking:
//
// @code
// DTL_ASSERT(ptr != nullptr);
// DTL_ASSERT_MSG(x > 0, "x must be positive");
//
// // Debug-only assertions (disabled in release builds)
// DTL_DEBUG_ASSERT(expensive_check());
//
// // Semantic assertions
// DTL_PRECONDITION(size > 0);   // Function precondition
// DTL_POSTCONDITION(result >= 0); // Function postcondition
// DTL_INVARIANT(valid_state());   // Class invariant
//
// // Unreachable code
// switch (value) {
//     case 1: return "one";
//     case 2: return "two";
//     default: DTL_UNREACHABLE();
// }
// @endcode
//
// ============================================================================
// Performance Timing
// ============================================================================
//
// Scoped timer for performance measurement:
//
// @code
// void expensive_operation() {
//     DTL_TIMED_SCOPE("expensive_operation");
//
//     // ... do work ...
//
// }  // Outputs: expensive_operation took 0.123s
//
// // Or manually
// {
//     dtl::scoped_timer timer("my_section");
//
//     // Check elapsed time mid-execution
//     if (timer.elapsed() > 1.0) {
//         dtl::debug_warning("Taking longer than expected");
//     }
// }
// @endcode
//
// ============================================================================
// Value Inspection
// ============================================================================
//
// Debug macros for inspecting values:
//
// @code
// int x = 42;
// double y = 3.14;
// std::string s = "hello";
//
// DTL_INSPECT(x);           // x = 42
// DTL_INSPECT2(x, y);       // x = 42, y = 3.14
// DTL_INSPECT3(x, y, s);    // x = 42, y = 3.14, s = hello
// @endcode
//
// ============================================================================
// Diagnostics
// ============================================================================
//
// Track memory and communication statistics:
//
// @code
// // Get memory diagnostics
// auto mem = dtl::get_memory_diagnostics();
// std::cout << "Allocated: " << mem.total_allocated << " bytes" << std::endl;
// std::cout << "Peak: " << mem.peak_allocated << " bytes" << std::endl;
//
// // Get communication diagnostics
// auto comm = dtl::get_communication_diagnostics();
// std::cout << "Bytes sent: " << comm.bytes_sent << std::endl;
// std::cout << "Messages: " << comm.messages_sent << std::endl;
//
// // Reset counters
// dtl::reset_memory_diagnostics();
// dtl::reset_communication_diagnostics();
// @endcode
//
// ============================================================================
// Debug Configuration
// ============================================================================
//
// Configure debug output behavior:
//
// @code
// // Show/hide rank and location in output
// dtl::configure_debug(true, true);   // show rank, show location
// dtl::configure_debug(true, false);  // show rank, hide location
// dtl::configure_debug(false, true);  // hide rank, show location
//
// // Redirect debug output
// std::ofstream logfile("debug.log");
// dtl::set_debug_stream(logfile);
// @endcode
//
// ============================================================================
// Usage Examples
// ============================================================================
//
// @code
// #include <dtl/utility/utility.hpp>
//
// int main(int argc, char** argv) {
//     // Initialize environment
//     dtl::scoped_environment<> env(argc, argv);
//
//     // Set up debugging
//     dtl::set_debug_level(dtl::debug_level::info);
//     dtl::set_debug_rank(env.rank());
//
//     dtl::debug_root("Starting application with ", env.size(), " processes");
//
//     {
//         DTL_TIMED_SCOPE("main computation");
//
//         // Your distributed computation here
//         DTL_ASSERT(env.size() > 0);
//
//         int local_result = env.rank() * 10;
//         DTL_INSPECT(local_result);
//     }
//
//     dtl::debug_root("Application complete");
//     return 0;
// }
// @endcode
//
// ============================================================================

}  // namespace dtl
