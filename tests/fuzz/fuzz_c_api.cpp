// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file fuzz_c_api.cpp
/// @brief Fuzz target for DTL C API layer
/// @details Exercises the C API's type dispatch, handle lifecycle, and
///          input validation using libFuzzer or AFL.
///
/// Build with:
///   clang++ -fsanitize=fuzzer,address -std=c++20 \
///     -DDTL_BUILD_C_BINDINGS=ON \
///     -I include -I backends \
///     tests/fuzz/fuzz_c_api.cpp \
///     -o fuzz_c_api -lm
///
/// Run:
///   ./fuzz_c_api corpus/

#include <dtl/bindings/c/dtl.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_status.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

// Fuzzer entry point
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Need at least 2 bytes: 1 for operation selector, 1+ for parameters
    if (size < 2) return 0;

    uint8_t op = data[0];
    const uint8_t* params = data + 1;
    size_t param_size = size - 1;

    // Create a default context for all operations
    dtl_context_t ctx = nullptr;
    dtl_status status = dtl_context_create_default(&ctx);
    if (status != DTL_SUCCESS || ctx == nullptr) {
        return 0;  // Context creation failed; nothing to fuzz
    }

    switch (op % 8) {
        case 0: {
            // Fuzz vector creation with varying sizes
            if (param_size >= sizeof(uint32_t)) {
                uint32_t vec_size;
                std::memcpy(&vec_size, params, sizeof(uint32_t));
                // Cap size to avoid OOM
                vec_size = vec_size % 100'000;

                dtl_vector_t vec = nullptr;
                status = dtl_vector_create_f64(&vec, ctx, vec_size);
                if (status == DTL_SUCCESS && vec != nullptr) {
                    // Query properties
                    (void)dtl_vector_size(vec);
                    (void)dtl_vector_local_size(vec);
                    dtl_vector_destroy(vec);
                }
            }
            break;
        }

        case 1: {
            // Fuzz context queries
            (void)dtl_context_rank(ctx);
            (void)dtl_context_size(ctx);
            (void)dtl_context_is_root(ctx);
            (void)dtl_context_device_id(ctx);
            (void)dtl_context_has_device(ctx);
            (void)dtl_context_is_valid(ctx);
            break;
        }

        case 2: {
            // Fuzz context duplication
            dtl_context_t dup = nullptr;
            status = dtl_context_dup(ctx, &dup);
            if (status == DTL_SUCCESS && dup != nullptr) {
                (void)dtl_context_rank(dup);
                dtl_context_destroy(dup);
            }
            break;
        }

        case 3: {
            // Fuzz NULL handle safety
            (void)dtl_context_is_valid(nullptr);
            dtl_context_destroy(nullptr);
            break;
        }

        case 4: {
            // Fuzz vector with create + fill + destroy cycle
            if (param_size >= 2 * sizeof(uint32_t)) {
                uint32_t vec_size;
                std::memcpy(&vec_size, params, sizeof(uint32_t));
                vec_size = vec_size % 10'000;

                int32_t fill_val;
                std::memcpy(&fill_val, params + sizeof(uint32_t),
                            sizeof(int32_t));

                dtl_vector_t vec = nullptr;
                status = dtl_vector_create_i32(&vec, ctx, vec_size);
                if (status == DTL_SUCCESS && vec != nullptr) {
                    // Fill operation (if API supports it)
                    (void)dtl_vector_size(vec);
                    dtl_vector_destroy(vec);
                }
            }
            break;
        }

        case 5: {
            // Fuzz domain queries
            (void)dtl_context_has_mpi(ctx);
            (void)dtl_context_has_cuda(ctx);
            (void)dtl_context_has_nccl(ctx);
            (void)dtl_context_has_shmem(ctx);
            break;
        }

        case 6: {
            // Fuzz barrier/fence (single-rank, should be no-ops)
            (void)dtl_context_barrier(ctx);
            (void)dtl_context_fence(ctx);
            break;
        }

        case 7: {
            // Fuzz status message for arbitrary codes
            if (param_size >= sizeof(uint16_t)) {
                uint16_t code;
                std::memcpy(&code, params, sizeof(uint16_t));
                (void)dtl_status_message(static_cast<dtl_status>(code));
            }
            break;
        }
    }

    dtl_context_destroy(ctx);
    return 0;
}
