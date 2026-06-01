---
name: knowledge-first
description: MANDATORY before any analysis, code review, or architecture investigation. This skill ensures you ALWAYS check the existing knowledge base and plan files BEFORE doing fresh analysis. Never re-analyze from scratch when prior analysis exists. Also ensures all new analysis results are archived as documents and diagrams. Use this skill whenever you need to understand, trace, or analyze any code flow, architecture, API dependency, or migration status.
---

# Knowledge First — 查知识库再分析

## Why This Exists

Previous sessions have produced extensive analysis documents, diagrams, and plans. Re-analyzing from scratch wastes time and loses accumulated insights. This skill prevents that.

## Mandatory Procedure

### Before ANY analysis task:

1. **Check the knowledge base** — read files in `.opencode/knowledge/` directory first
2. **Check the plan files** — read `.opencode/plans/cuda-to-cann-migration-v2.md` (especially Section 13 for current status)
3. **Check the diagrams** — read `.opencode/plans/migration-diagrams.md` for existing Mermaid diagrams
4. **Check the v1 plan** — read `.opencode/cuda-to-cann-migration.md` for additional detail
5. **Check the session-progress skill** — read `.opencode/skills/session-progress/SKILL.md` for session protocol

If relevant prior analysis exists, **USE IT**. Extend it, correct it, or reference it — do not duplicate it.

### After ANY analysis task:

1. **Archive results** — write new analysis to `.opencode/knowledge/<topic>-analysis.md`
2. **Include diagrams** — add Mermaid sequence diagrams, flow charts, or architecture diagrams to the document
3. **Update the plan file** — update Section 13 in `.opencode/plans/cuda-to-cann-migration-v2.md` with findings
4. **Cross-reference** — in the knowledge document, link to relevant plan sections and diagram numbers

## Knowledge Base Structure

All analysis documents live in `.opencode/knowledge/`:

```
.opencode/knowledge/
├── bootstrap-cann-analysis.md    # Bootstrap & communicator CANN compatibility
├── ipc-mechanism-analysis.md     # IPC memory sharing analysis (to be created)
├── channel-architecture.md       # MemoryChannel/PortChannel/SwitchChannel analysis
├── ...                           # More topics as needed
```

Each document MUST contain:
- **Date** of analysis
- **Conclusion** section at the top (TL;DR)
- **Mermaid diagrams** (sequence, flow, architecture)
- **Compatibility table** (what works, what doesn't, severity level)
- **Cross-references** to plan file sections and diagrams

## Project Context

**Target repo**: `D:\C++\mscclpp\mscclpp` — ONLY repo we modify and commit to

**Reference repos** (READ ONLY): All directories under `D:\C++` except `mscclpp`. Discover them by listing `D:\C++` at session start. Common ones: `cann_*`, `umdk`, `superpower`.

## Existing Knowledge Sources

| File | Content | Lines |
|------|---------|-------|
| `.opencode/plans/cuda-to-cann-migration-v2.md` | Complete CUDA→CANN migration plan, API scans, HCCL analysis, UMDK analysis | 1062+ |
| `.opencode/cuda-to-cann-migration.md` | v1 migration plan (simplified) | 658 |
| `.opencode/plans/migration-diagrams.md` | 15 Mermaid diagrams (architecture, sequence, flow) | 813 |
| `.opencode/agents/cann-migration.md` | Session start/end protocol | 39 |
| `.opencode/knowledge/` | Analysis documents archive (new) | varies |

## Anti-patterns (DO NOT)

- DO NOT analyze from scratch when prior analysis exists in the knowledge base
- DO NOT produce analysis without archiving it to `.opencode/knowledge/`
- DO NOT skip adding Mermaid diagrams — visual documentation is critical for complex flows
- DO NOT forget to cross-reference existing plan sections and diagrams
- DO NOT forget to update the plan file (Section 13) after analysis