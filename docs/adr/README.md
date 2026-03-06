# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) that document
key design decisions in DTL. Each ADR explains the context, decision, and
consequences of a significant architectural choice.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-result-over-exceptions.md) | result&lt;T&gt; over exceptions | Accepted |
| [002](002-concepts-over-crtp.md) | C++20 concepts over CRTP | Accepted |
| [003](003-per-instance-comm-dup.md) | Per-instance MPI_Comm_dup | Accepted |
| [004](004-remote-ref-no-implicit-conversion.md) | remote_ref&lt;T&gt; no implicit conversion | Accepted |
| [005](005-policy-based-design.md) | Variadic policy-based container design | Accepted |

## Format

Each ADR follows this structure:
- **Status:** Proposed, Accepted, Deprecated, or Superseded
- **Context:** What is the issue that motivates this decision?
- **Decision:** What is the change being proposed or made?
- **Consequences:** What are the trade-offs and results?
