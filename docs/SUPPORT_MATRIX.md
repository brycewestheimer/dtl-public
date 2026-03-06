# DTL Support Matrix

Feature support across language bindings.

## Container Placements

| Feature | C++ | C | Python | Fortran |
|---|---|---|---|---|
| Host placement | Supported | Supported | Supported | Supported |
| Device placement (CUDA) | Supported | Supported | Partial | Partial |
| Unified placement (CUDA) | Supported | Supported | Partial | Partial |
| Device Preferred | Unsupported | Unsupported | Unsupported | Unsupported |

## MPI Collectives (Host)

| Feature | C++ | C | Python | Fortran |
|---|---|---|---|---|
| Barrier | Supported | Supported | Supported | Supported |
| Broadcast | Supported | Supported | Supported | Supported |
| Allreduce | Supported | Supported | Supported | Supported |
| Send/Recv | Supported | Supported | Supported | Supported |
| RMA Put/Get | Supported | Supported | Partial | Partial |

## NCCL Collectives (Device)

| Feature | C++ | C | Python | Fortran |
|---|---|---|---|---|
| Allreduce | Supported | Deferred | Deferred | Deferred |
| Broadcast | Supported | Deferred | Deferred | Deferred |
| Reduce | Supported | Deferred | Deferred | Deferred |
| Allgather | Supported | Deferred | Deferred | Deferred |
| Send/Recv (P2P) | Supported | Deferred | Deferred | Deferred |
| Isend/Irecv | Supported | Deferred | Deferred | Deferred |
| Barrier | Supported | Deferred | Deferred | Deferred |
| Scan/Exscan | Unsupported | Unsupported | Unsupported | Unsupported |
| Variable-size collectives | Unsupported | Unsupported | Unsupported | Unsupported |

## CUDA Container Operations

| Operation | C++ Device | C++ Unified | C Device | C Unified |
|---|---|---|---|---|
| Create | Supported | Supported | Supported | Supported |
| Fill | Supported | Supported | Supported | Supported |
| Resize (vector) | Supported | Supported | Supported | Supported |
| Reduce (sum/min/max) | Supported | Supported | Supported | Supported |
| Sort | Supported | Supported | Supported | Supported |
| Copy to/from host | Supported | Supported | Supported | Supported |
| Direct host access | Unsupported | Supported | Unsupported | Supported |
| Device view/pointer | Supported | Supported | Supported | Supported |
| ND tensor access | Unsupported | Supported | Unsupported | Supported |
| Tensor reshape | Unsupported | Supported | Unsupported | Supported |
| Redistribute | Unsupported | Unsupported | Unsupported | Unsupported |

## Legend

- **Supported**: Fully implemented and tested
- **Partial**: API exists but may not cover all placements/operations
- **Deferred**: Planned but not yet implemented
- **Unsupported**: Not planned or deliberately excluded
