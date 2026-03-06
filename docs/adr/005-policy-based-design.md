# ADR-005: Variadic Policy-Based Container Design

**Status:** Accepted
**Date:** 2026-01-15

## Context

Distributed containers need configuration along multiple orthogonal axes:
how data is partitioned, where it resides in memory, what consistency
guarantees are provided, how operations execute, and how errors are reported.
Hard-coding these choices limits flexibility; inheritance hierarchies create
combinatorial explosion.

## Decision

DTL containers accept a variadic policy parameter pack:

```cpp
template <typename T, typename... Policies>
class distributed_vector { ... };

// Usage:
distributed_vector<double, cyclic_partition<>, device_only<0>, async{}>
```

Policies are extracted by category using `extract_*_policy_t` type aliases.
Missing policies use defaults. Policy composition is validated at compile
time with `static_assert` messages.

Five orthogonal policy axes:
1. **Partition** — how data divides across ranks
2. **Placement** — where data resides (host/device/unified)
3. **Consistency** — synchronization model
4. **Execution** — how local work executes
5. **Error** — how errors are reported

## Consequences

**Positive:**
- Policies are orthogonal and independently selectable
- Order-independent specification (`<cyclic, async>` == `<async, cyclic>`)
- Compile-time validation with clear error messages
- Zero runtime overhead (all dispatch is static)
- Defaults make simple cases simple: `distributed_vector<int>` just works

**Negative:**
- Longer type names for complex configurations
- Template error messages can be verbose despite concepts
- Cannot change policies after construction (compile-time choice)

**Mitigations:**
- `make_policy_set` provides a canonical policy set from any order
- `policy_count::validate()` gives targeted error messages
- Type aliases simplify common configurations
