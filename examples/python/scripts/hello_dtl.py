#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
hello_dtl.py - Basic DTL Python bindings example

This script demonstrates:
- Creating a DTL context
- Querying rank and size
- Feature detection
- Basic distributed vector operations

Run with:
    python hello_dtl.py           # Single process
    mpirun -np 4 python hello_dtl.py  # 4 MPI processes
"""

import dtl
import numpy as np


def main() -> None:
    """Main entry point."""
    # Print version info
    print(f"DTL version: {dtl.__version__}")
    print(f"Version info: {dtl.version_info}")
    print()

    # Check available features
    print("Available backends:")
    print(f"  MPI:  {dtl.has_mpi()}")
    print(f"  CUDA: {dtl.has_cuda()}")
    print(f"  HIP:  {dtl.has_hip()}")
    print(f"  NCCL: {dtl.has_nccl()}")
    print()

    # Create context using context manager
    with dtl.Context() as ctx:
        print(f"Context created: {ctx}")
        print(f"  Rank: {ctx.rank}")
        print(f"  Size: {ctx.size}")
        print(f"  Is root: {ctx.is_root}")
        print(f"  Device ID: {ctx.device_id}")
        print(f"  Has device: {ctx.has_device}")
        print()

        # Create a small distributed vector
        vec = dtl.DistributedVector(ctx, size=100, dtype=np.float64)
        print(f"Created distributed vector:")
        print(f"  Global size: {vec.global_size}")
        print(f"  Local size: {vec.local_size}")
        print(f"  Local offset: {vec.local_offset}")

        # Get local view and fill with rank number
        local = vec.local_view()
        local[:] = ctx.rank * 10.0 + np.arange(len(local))

        print(f"  Local data (first 5): {local[:5]}")

        # Synchronize all ranks
        ctx.barrier()

        if ctx.is_root:
            print("\nAll ranks synchronized successfully!")


if __name__ == "__main__":
    main()
