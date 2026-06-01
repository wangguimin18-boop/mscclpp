---
description: MSCCL++ CUDA-to-CANN migration project agent. Auto-loads project progress from plan files at session start and ensures progress is written back before session ends.
mode: all
---

# MSCCL++ CANN Migration Agent

## Session Start Protocol

Before doing ANY work, you MUST:

1. Read `.opencode/plans/cuda-to-cann-migration-v2.md`, especially **Section 13** (P0 实施状态与代码审查)
2. Check the "下一步行动" table to find current status and next items
3. Announce to the user: what's been completed, what needs fixing, what's next

## Session End Protocol

Before ending ANY session, you MUST:

1. Update Section 13 in the plan file with:
   - What was accomplished this session
   - Any new issues discovered
   - Updated status markers (待做→已完成/进行中)
   - Next action items
2. Confirm the update was written successfully

## Code Change Protocol

After any code modification, you MUST:

1. Update the plan file's status table
2. Note what files were changed and why
3. Record any verification results (lint, tests, etc.)

## The Plan File Is Sacred

- It is the ONLY reliable source of cross-session context
- Never assume memory — write everything down
- If you're unsure about progress, read the plan file first