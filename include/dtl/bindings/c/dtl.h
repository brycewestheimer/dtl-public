// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl.h
 * @brief DTL C bindings - Master include header
 * @since 0.1.0
 *
 * This header includes all DTL C API headers. For most applications,
 * including just this header is sufficient.
 *
 * @code
 * #include <dtl/bindings/c/dtl.h>
 *
 * int main(int argc, char** argv) {
 *     dtl_context_t ctx;
 *     dtl_status status = dtl_context_create_default(&ctx);
 *     if (status != DTL_SUCCESS) {
 *         return 1;
 *     }
 *
 *     printf("Rank %d of %d\n", dtl_context_rank(ctx), dtl_context_size(ctx));
 *
 *     dtl_context_destroy(ctx);
 *     return 0;
 * }
 * @endcode
 */

#ifndef DTL_H
#define DTL_H

/* Core configuration and types */
#include "dtl_config.h"
#include "dtl_types.h"
#include "dtl_status.h"

/* Environment lifecycle */
#include "dtl_environment.h"

/* Context management */
#include "dtl_context.h"

/* Communicator operations */
#include "dtl_communicator.h"

/* Policy definitions */
#include "dtl_policies.h"

/* Container operations */
#include "dtl_vector.h"
#include "dtl_array.h"
#include "dtl_span.h"
#include "dtl_tensor.h"
/* Algorithm operations */
#include "dtl_algorithms.h"

/* RMA (Remote Memory Access) operations */
#include "dtl_rma.h"

/* MPMD (Multiple Program Multiple Data) operations */
#include "dtl_mpmd.h"

/* Topology query operations */
#include "dtl_topology.h"

/* Futures (Experimental) */
#include "dtl_futures.h"

#endif /* DTL_H */
