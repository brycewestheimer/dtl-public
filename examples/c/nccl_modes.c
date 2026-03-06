#include <dtl/bindings/c/dtl.h>

#include <stdio.h>

static void print_status(const char* label, dtl_status status) {
    printf("%s: %s (%d)\n", label, dtl_status_message(status), (int)status);
}

static void print_capabilities(const char* label, dtl_context_t ctx) {
    const int mode = dtl_context_nccl_mode(ctx);
    const int native_allreduce = dtl_context_nccl_supports_native(ctx, DTL_NCCL_OP_ALLREDUCE);
    const int native_scan = dtl_context_nccl_supports_native(ctx, DTL_NCCL_OP_SCAN);
    const int hybrid_scan = dtl_context_nccl_supports_hybrid(ctx, DTL_NCCL_OP_SCAN);

    printf("%s\n", label);
    printf("  nccl_mode: %d\n", mode);
    printf("  supports_native(ALLREDUCE): %d\n", native_allreduce);
    printf("  supports_native(SCAN):      %d\n", native_scan);
    printf("  supports_hybrid(SCAN):      %d\n", hybrid_scan);
}

int main(void) {
    dtl_context_t base = NULL;
    dtl_context_t native_ctx = NULL;
    dtl_context_t hybrid_ctx = NULL;
    dtl_context_t split_hybrid_ctx = NULL;

    dtl_status status = dtl_context_create_default(&base);
    if (!dtl_status_ok(status)) {
        print_status("dtl_context_create_default", status);
        return 1;
    }

    printf("Rank %d/%d\n", (int)dtl_context_rank(base), (int)dtl_context_size(base));
    printf("has_mpi=%d has_cuda=%d has_nccl=%d\n",
           dtl_context_has_mpi(base),
           dtl_context_has_cuda(base),
           dtl_context_has_nccl(base));

    if (!dtl_context_has_mpi(base) || !dtl_context_has_cuda(base)) {
        printf("Skipping NCCL mode demo: MPI and CUDA domains are required.\n");
        dtl_context_destroy(base);
        return 0;
    }

    int device_id = dtl_context_device_id(base);
    if (device_id < 0) {
        device_id = 0;
    }

    status = dtl_context_with_nccl_ex(base, device_id,
                                      DTL_NCCL_MODE_NATIVE_ONLY,
                                      &native_ctx);
    if (!dtl_status_ok(status)) {
        print_status("dtl_context_with_nccl_ex(native_only)", status);
        dtl_context_destroy(base);
        return 0;  // Graceful skip when NCCL runtime is unavailable
    }

    status = dtl_context_with_nccl_ex(base, device_id,
                                      DTL_NCCL_MODE_HYBRID_PARITY,
                                      &hybrid_ctx);
    if (!dtl_status_ok(status)) {
        print_status("dtl_context_with_nccl_ex(hybrid_parity)", status);
        dtl_context_destroy(native_ctx);
        dtl_context_destroy(base);
        return 0;
    }

    print_capabilities("Native-only context:", native_ctx);
    print_capabilities("Hybrid-parity context:", hybrid_ctx);

    status = dtl_context_split_nccl_ex(hybrid_ctx,
                                       0,  // color: all ranks same subgroup
                                       (int)dtl_context_rank(hybrid_ctx),
                                       device_id,
                                       DTL_NCCL_MODE_HYBRID_PARITY,
                                       &split_hybrid_ctx);
    if (!dtl_status_ok(status)) {
        print_status("dtl_context_split_nccl_ex", status);
    } else {
        print_capabilities("Split hybrid context:", split_hybrid_ctx);
    }

    dtl_context_destroy(split_hybrid_ctx);
    dtl_context_destroy(hybrid_ctx);
    dtl_context_destroy(native_ctx);
    dtl_context_destroy(base);
    return 0;
}
