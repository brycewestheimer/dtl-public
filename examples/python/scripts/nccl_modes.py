#!/usr/bin/env python3
"""Demonstrate explicit NCCL mode selection and capability queries."""

from __future__ import annotations

import dtl


def describe_ctx(name: str, ctx: dtl.Context) -> None:
    print(name)
    print(f"  nccl_mode: {ctx.nccl_mode}")
    print(
        "  supports_native(ALLREDUCE): "
        f"{ctx.nccl_supports_native(dtl.DTL_NCCL_OP_ALLREDUCE)}"
    )
    print(
        "  supports_native(SCAN):      "
        f"{ctx.nccl_supports_native(dtl.DTL_NCCL_OP_SCAN)}"
    )
    print(
        "  supports_hybrid(SCAN):      "
        f"{ctx.nccl_supports_hybrid(dtl.DTL_NCCL_OP_SCAN)}"
    )


def main() -> int:
    with dtl.Context() as base:
        print(f"Rank {base.rank}/{base.size}")
        print(
            f"has_mpi={base.has_mpi} has_cuda={base.has_cuda} "
            f"has_nccl={base.has_nccl}"
        )

        if not base.has_mpi or not base.has_cuda:
            print("Skipping NCCL mode demo: MPI and CUDA domains are required.")
            return 0

        device_id = base.device_id if base.device_id >= 0 else 0

        try:
            native = base.with_nccl(
                device_id=device_id,
                mode=dtl.DTL_NCCL_MODE_NATIVE_ONLY,
            )
            hybrid = base.with_nccl(
                device_id=device_id,
                mode=dtl.DTL_NCCL_MODE_HYBRID_PARITY,
            )
        except RuntimeError as exc:
            # Graceful skip when NCCL runtime/driver is unavailable.
            print(f"Unable to create NCCL contexts: {exc}")
            return 0

        describe_ctx("Native-only context:", native)
        describe_ctx("Hybrid-parity context:", hybrid)

        split_hybrid = hybrid.split_nccl(
            color=base.rank % 2,
            key=base.rank,
            device_id=device_id,
            mode=dtl.DTL_NCCL_MODE_HYBRID_PARITY,
        )
        describe_ctx("Split hybrid context:", split_hybrid)

        base.barrier()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
