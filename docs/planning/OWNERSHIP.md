# Ownership and Locking

Ownership is by area to minimize merge conflicts and unclear responsibility.

| Area | Paths | Primary Role | Backup Role |
|---|---|---|---|
| Core Model | `src/model/**` | Modeling Agent | Validation Agent |
| Tree Search | `src/search/**` | Search Agent | Modeling Agent |
| Data + Partitions | `src/data/**` | Data Agent | Modeling Agent |
| GPU Kernels | `src/gpu/**` | Performance Agent | Search Agent |
| Evaluation | `src/eval/**`, `benchmarks/**` | Validation Agent | Data Agent |
| Tests | `tests/**` | Validation Agent | Area owner of touched module |
| Docs/Planning | `docs/**` | Coordination Agent | Any agent |

## File Lock Protocol
- If you need to edit outside your area, claim a temporary lock.
- Temporary lock format:

`LOCK | owner=<name> | paths=<glob> | task=<task-id> | started=<YYYY-MM-DD>`

- Remove lock when task moves to `Review` or `Done`.

## Active Locks
- None.
