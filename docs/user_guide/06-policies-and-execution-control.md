# 6. Policies and Execution Control

## Policy system overview

DTL policies provide orthogonal control over behavior.

Primary axes:

- partition
- placement
- execution
- consistency
- error

## Partition policies

Determine how global indices map to ranks (e.g., block/cyclic/replicated variants depending on API/backend support).

Use partition policy to balance:

- load distribution
- communication locality
- downstream collective behavior

## Placement policies

Control where data resides:

- host-only
- unified memory
- device-only/device-preferred

Choose placement based on kernel location and transfer characteristics.

## Execution policies

Select local/parallel execution semantics for algorithms.

Align policy with backend capability and desired reproducibility/performance tradeoff.

## Consistency policies

Define synchronization visibility and ordering behavior for distributed mutations.

Be explicit when using relaxed or deferred synchronization modes.

## Error policies

Determine failure behavior (result/status propagation, throwing behavior, callback-based handling depending on surface/API).

## Policy composition guidance

- Start with conservative defaults.
- Move to specialized policy sets for measured bottlenecks.
- Keep policy selection explicit in performance-critical interfaces.

## Validation checklist

- policy set supported by target backend
- no conflicting assumptions across layers (C++/C/Python/Fortran)
- documented behavior for non-default policies in team code

## Next step

Continue to [Chapter 7](07-algorithms-collectives-and-remote-operations.md).

## Deep-dive reference

- [Legacy Deep-Dive: Policies](policies.md)
