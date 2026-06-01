---
name: session-progress
description: MANDATORY at session start for the MSCCL++ CUDA-to-CANN migration project. Loads project progress, current status, and next steps from the plan file so no context is lost between sessions. Use ONLY when working on this specific migration project.
---

# Session Progress Recovery

This skill ensures project progress is never lost between sessions. It must be invoked at the start of every new session working on the MSCCL++ CUDA→CANN migration.

## Procedure

### Step 1: Read the plan file

Read `.opencode/plans/cuda-to-cann-migration-v2.md` from the project root. Focus especially on:

- **Section 13** (P0 实施状态与代码审查) — this is the live progress tracker containing:
  - What has been completed
  - What issues need fixing
  - Strategic decisions made
  - CANN source code discoveries
  - Next action items with status markers

- **Section 8** (迁移优先级) — for understanding the overall phase structure

- **Section 7** (迁移策略决策) — for architectural choices made

### Step 2: Update Section 13 after any work

After completing ANY task in this project (code change, review, analysis, decision), you MUST:

1. Update the relevant status rows in Section 13.5 (下一步行动表) — change "待做" to "已完成" or "进行中"
2. Add any new findings, issues, or decisions to Section 13
3. If a major milestone is reached, add a new dated subsection under Section 13

### Step 3: Before ending a session

Before the session ends (or at any natural stopping point), you MUST:

1. Write a brief summary of what was accomplished in this session to Section 13
2. Update all status markers to reflect current reality
3. Note any blockers or decisions pending user input
4. Ensure the "下一步行动" table accurately reflects what should be done next

## Critical Rules

- **NEVER leave a session without updating the plan file** — this is the single source of truth for cross-session continuity
- **NEVER assume you'll remember context** — write it down immediately
- **ALWAYS read Section 13 before starting work** — it contains the accumulated wisdom from all previous sessions
- The plan file at `.opencode/plans/cuda-to-cann-migration-v2.md` is the project's memory — treat it as sacred

## File Location

- Plan file: `.opencode/plans/cuda-to-cann-migration-v2.md`
- Diagrams: `.opencode/plans/migration-diagrams.md`
- CANN source repos (reference only, do not modify):
  - `D:\C++\cann_runtime\runtime` — ACL runtime API definitions
  - `D:\C++\cann_hccl\hccl` — HCCL collective API
  - `D:\C++\cann_hcomm\hcomm` — HCOMM primitives + URMA API
  - `D:\C++\cann_shmem\shmem` — ACLSHMEM zero-copy API
  - `D:\C++\cann_driver\driver` — HAL + DSMI device management