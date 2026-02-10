# AGENTS.md

This repository uses a multi-agent workflow. Follow this file as the operating contract.

## Collaboration Principles
- Keep tasks small, explicit, and testable.
- One agent owns a task at a time.
- One agent edits a file at a time.
- Always leave a clean handoff for the next agent.

## Source-of-Truth Planning Docs
- `docs/planning/README.md`: how planning artifacts are used.
- `docs/planning/ROADMAP.md`: milestone plan (M0-M5).
- `docs/planning/TASK_BOARD.md`: current task status and assignments.
- `docs/planning/OWNERSHIP.md`: component/file ownership and lock rules.
- `docs/planning/HANDOFF_TEMPLATE.md`: required handoff structure.

## Start-of-Task Checklist
1. Read `docs/planning/TASK_BOARD.md` and claim one task.
2. Confirm ownership scope in `docs/planning/OWNERSHIP.md`.
3. Create a branch named `codex/<agent>/<task-id-short>`.
4. Add an in-progress note with scope and expected files.

## File Ownership and Locking
- If your task requires files outside your ownership area, record a temporary lock in `docs/planning/OWNERSHIP.md`.
- If another active task already holds the lock, do not edit those files.
- Resolve collisions by splitting work into separate files or sequence the tasks.

## Implementation Rules
- Keep changes minimal to the claimed task scope.
- Include tests or validation artifacts for behavior changes.
- Do not bundle unrelated refactors with feature work.
- Do not rewrite another agent's in-progress area without explicit reassignment.

## Handoff Requirements
Before unclaiming a task, provide a handoff using `docs/planning/HANDOFF_TEMPLATE.md` including:
- what changed
- exact files touched
- tests/commands run and results
- known risks, TODOs, and next recommended owner

## Completion Criteria
A task is `Done` only when all are true:
- acceptance criteria met
- validation completed
- documentation updated (if behavior changed)
- task row updated in `docs/planning/TASK_BOARD.md`

## Dispute/Conflict Protocol
- If two agents need the same file, pause and re-scope in `docs/planning/TASK_BOARD.md`.
- Prefer additive extension points over deep shared-file edits.
- Escalate unresolved ownership conflicts by creating a blocker entry in the task board.
