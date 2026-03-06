# NCCL/CUDA Audit

This audit records the supported surface and the current remediation status for
the CUDA and NCCL backends as of 2026-03-06.

## Supported Operations Matrix

| Surface | MPI | CUDA | NCCL |
|---|---|---|---|
| Generic distributed algorithms | Supported | Local execution only | Not selected implicitly |
| Context rank/size for generic distributed code | Primary | N/A | Not a selector |
| Device memory allocation | N/A | Supported | N/A |
| Device local algorithms | N/A | Supported | N/A |
| `with_nccl(device_id)` | N/A | Requires CUDA | Supported |
| `split_nccl(...)` | Bootstrap via MPI split | Requires CUDA | Supported in C++ only |
| Point-to-point | Supported | N/A | Supported for device buffers |
| Broadcast | Supported | N/A | Supported for device buffers |
| Gather / Scatter | Supported | N/A | Supported for fixed-size device buffers |
| Allgather / Alltoall | Supported | N/A | Supported for fixed-size device buffers |
| Reduce / Allreduce | Supported | N/A | Supported for explicit device-buffer paths |
| Variable-size collectives | Supported | N/A | Unsupported |
| Scan / Exscan | Supported | Local-only helpers | Unsupported on NCCL adapter |
| Scalar convenience reductions | Supported | N/A | Unsupported |
| Logical reductions | Supported | N/A | Unsupported |
| Host-buffer collectives | Supported | N/A | Unsupported |
| RMA / one-sided | Supported | N/A | Unsupported |

## Bug List

### High Severity

| Subsystem | Issue | Status |
|---|---|---|
| Generic algorithm dispatch | Context-based NCCL auto-selection routed generic host-oriented algorithms through NCCL | Remediated by removing umbrella exposure and restoring MPI-primary dispatch |
| NCCL adapter semantics | Blocking methods returned before CUDA stream completion | Remediated by explicit synchronization before return |
| NCCL adapter memory contract | Host scalars and host buffers could enter NCCL collectives | Remediated by device-buffer validation and removal of scalar helpers |
| Scan/exscan parity | Host scratch emulation for NCCL scan/exscan was invalid | Remediated by removing/marking unsupported |
| Async completion | `wait()` / `test()` did not propagate CUDA errors | Remediated |

### Medium Severity

| Subsystem | Issue | Status |
|---|---|---|
| Documentation | NCCL docs overstated MPI-like parity and implicit context behavior | Remediated in backend docs |
| Public API scope | `split_nccl(...)` support was not clearly documented as C++-only | Remediated in docs/comments |
| Test coverage | No explicit contract tests for host-buffer rejection and blocking completion | Remediated with explicit NCCL adapter contract coverage |

### Low Severity

| Subsystem | Issue | Status |
|---|---|---|
| Warning hygiene | NCCL communicator still emits reorder/sign-conversion warnings | Open |
| Broader CUDA docs | Some higher-level docs outside the backend pages still describe NCCL optimistically | Open follow-up |

## Parity Gaps

### Must-Fix Correctness

- Keep generic distributed algorithms MPI-primary until there is a device-aware
  distributed container and algorithm path.
- Preserve the rule that NCCL only accepts CUDA device buffers.
- Keep blocking and async completion semantics explicit and error-reporting.

### Must-Document Unsupported

- Scalar convenience reductions on the NCCL adapter
- Scan/exscan on the NCCL adapter
- Variable-size collectives
- Logical reductions
- Host-buffer communication
- `split_nccl(...)` as C++-only/publicly limited

### Future Parity Work

- Explicit device-resident distributed container support
- Opt-in NCCL-aware algorithm entry points
- Binding parity for any retained public NCCL APIs beyond `with_nccl(...)`
- Additional multi-rank GPU integration coverage once MPI-enabled CI/build
  environments are available consistently
