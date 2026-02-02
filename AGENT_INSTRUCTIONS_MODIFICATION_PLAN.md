# sAImone Agent Instructions — Modification Plan

## Purpose

Apply relevant GPT-5.2 prompting best practices to the existing sAImone agent
instructions while preserving the current objectives: a MAPS-aligned medical
affairs strategic/tactical agent with 5-phase workflow, Hard Logic integration,
silent tool execution, and regulatory compliance.

This plan maps each GPT-5.2 recommendation to specific files and describes the
change, the rationale, and any dependencies.

---

## Files in Scope

| File | Role |
|---|---|
| `system_instructions.txt` | Primary behavioural instructions (279 lines) |
| `assistant_config.py` | Reasoning effort & verbosity defaults |
| `session_manager.py` | Silent instructions, context builder |
| `core_assistant.py` | API call orchestration, instruction injection |
| `prompt_cache.py` | Static context block assembly |
| `tool_config.py` | Tool result limits, token budgets |

---

## Modification 1 — Output Verbosity Spec

### What GPT-5.2 guide says
Section 3.1: Give clear, concrete length constraints. Default 3–6 sentences,
compact bullets, structured sections. Avoid long narrative paragraphs.

### Current state
- `assistant_config.py` sets `DEFAULT_VERBOSITY = "medium"` (API-level control).
- `system_instructions.txt` line 51 says "Use matrices/tables wherever possible"
  and line 61 says "Keep meta commentary to a bare minimum".
- No explicit output shape or length constraints in the instructions.

### Proposed change
Add an `<output_verbosity_spec>` block to **`system_instructions.txt`** in the
GENERAL RULES section (after line 63). This gives the model explicit formatting
discipline that complements the API-level verbosity knob.

```
<output_verbosity_spec>
- Default responses: structured sections with ≤5 bullets per section.
- Simple factual questions: ≤3 sentences + source link.
- Multi-phase outputs (Phases 1–5): 1 short overview paragraph, then matrix/table
  output, then ≤5 tagged bullets (What changed, Key findings, Risks, Next steps,
  Open questions).
- Prefer compact bullets, tables, and short sections over narrative paragraphs.
- Do not rephrase the user's request unless it changes semantics.
- Keep technical/meta notes in a standalone "Notes" section at the end.
</output_verbosity_spec>
```

### Rationale
GPT-5.2 builds more deliberate scaffolding by default. Without explicit length
constraints it tends to over-elaborate, especially in multi-phase workflows.
This block ensures the 5-phase output stays scannable without losing depth.

### Dependencies
None. Pure instruction-layer change.

---

## Modification 2 — Scope Discipline Block

### What GPT-5.2 guide says
Section 3.2: Explicitly forbid extra features, uncontrolled expansions. Choose
simplest valid interpretation for ambiguous instructions.

### Current state
- Line 52: "No generic frameworks; outputs must be context-specific."
- Line 60: "prefer a width over depth unless prompted to."
- No explicit scope-clamping instruction.

### Proposed change
Add a `<scope_discipline>` block to **`system_instructions.txt`** in the
GENERAL RULES section.

```
<scope_discipline>
- Deliver EXACTLY what the user asked for. Do not add unrequested analyses,
  extra frameworks, or speculative sections.
- If the user asks a targeted question, answer it directly — do not trigger
  the full 5-phase workflow unless the query warrants it.
- If an instruction is ambiguous, choose the simplest valid interpretation
  and state the assumption briefly.
- When you identify adjacent work that could be valuable, mention it as an
  optional next step — do not execute it unprompted.
</scope_discipline>
```

### Rationale
GPT-5.2 is more structured but can produce more content than needed. In a
tool-heavy medical affairs agent that already has a 5-phase pipeline, scope
creep inflates token usage and dilutes actionable output.

### Dependencies
Interacts with the Phase prompts. Phase instructions already say "build from
previous phases and add fields as necessary" — the scope block ensures "as
necessary" is interpreted conservatively.

---

## Modification 3 — Long-Context Handling

### What GPT-5.2 guide says
Section 3.3: For inputs >10k tokens, produce an internal outline, restate
constraints, anchor claims to sections.

### Current state
- `session_manager.py` manages token budgets (24K–96K) and builds context via
  `adaptive_context_builder()`.
- `prompt_cache.py` handles the static prefix.
- No instruction to the model for how to handle accumulated multi-phase context.

### Proposed change
Add a `<long_context_handling>` block to **`system_instructions.txt`** after the
GENERAL RULES section, before Phase 1.

```
<long_context_handling>
- When operating in later phases (Phase 3+) or when the accumulated conversation
  exceeds ~10k tokens:
  - Internally summarise the key findings from prior phases before generating
    the new phase output.
  - Restate the user's core objective, therapy area, region, and timeline
    constraints at the start of each phase.
  - Anchor all claims to their source: "Per the Phase 1 landscape analysis…",
    "From [Source Name]…".
- If the answer depends on specific data points (dates, thresholds, regulatory
  deadlines), quote or paraphrase them rather than referencing generically.
</long_context_handling>
```

### Rationale
The 5-phase workflow accumulates substantial context. GPT-5.2's improved
recall still benefits from explicit re-grounding to avoid "lost in the scroll"
errors — especially when Phase 5 needs to reference Phase 1 findings.

### Dependencies
- Works with `session_manager.py` checkpoint summaries (the model instruction
  complements the code-level context management).
- If Modification 7 (Compaction) is implemented, this block becomes even more
  important as compacted context is lossy.

---

## Modification 4 — Uncertainty & Hallucination Safeguards

### What GPT-5.2 guide says
Section 3.4: Handle ambiguity explicitly — clarifying questions or labeled
interpretations. Never fabricate figures. Self-check for high-risk outputs.

### Current state
- Line 54: "Flag any data limitations, compliance risks, or uncertainties."
- Line 56: "Do not infer or use industry averages—only verifiable sources."
- Line 58: Tag unverified data as `[Requires Verification]`.
- These are scattered rules, not a cohesive block.

### Proposed change
**A)** Consolidate and strengthen into an `<uncertainty_and_ambiguity>` block in
**`system_instructions.txt`** (GENERAL RULES section, replacing/absorbing the
scattered rules at lines 54, 56, 58).

```
<uncertainty_and_ambiguity>
- If the query is ambiguous or underspecified:
  - Ask 1–3 precise clarifying questions, OR
  - Present 2–3 plausible interpretations with clearly labeled assumptions.
- Never fabricate exact figures, regulatory dates, clinical endpoints, or
  external references. If uncertain, state so explicitly.
- When external facts may have changed (drug approvals, pricing, guidelines)
  and live search returns no results: answer in general terms and flag with
  [Requires Verification].
- Prefer "Based on available data…" or "Per [Source]…" over absolute claims.
- Do not infer or use industry averages — only verifiable sources.
</uncertainty_and_ambiguity>
```

**B)** Add a `<high_risk_self_check>` block specifically for the medical affairs
context.

```
<high_risk_self_check>
Before finalising any output involving regulatory status, clinical safety data,
compliance recommendations, or market access claims:
- Re-scan your output for:
  - Unstated assumptions about approval status or indication scope
  - Specific numbers not grounded in a cited source
  - Overly strong language ("guaranteed", "always", "no risk")
- If found, qualify them and explicitly state the assumption.
</high_risk_self_check>
```

### Rationale
Medical affairs is a high-risk domain. The existing scattered rules work but
lack the structured emphasis GPT-5.2 responds well to. Consolidating them into
tagged blocks improves instruction adherence.

### Dependencies
None. Absorbs existing lines 54, 56, 58 — those can be removed to avoid
duplication.

---

## Modification 5 — User Updates Spec (Agentic Steerability)

### What GPT-5.2 guide says
Section 5: Send brief updates only on phase changes or plan changes. Avoid
narrating routine tool calls. Include at least one concrete outcome per update.

### Current state
- Line 62: "CRITICAL: Never describe your internal tool-calling process…"
- This is binary (never describe) — there's no spec for what the model
  *should* say during multi-step work.

### Proposed change
Add a `<user_updates_spec>` block to **`system_instructions.txt`** in the
GENERAL RULES section. This replaces/supplements the line-62 rule.

```
<user_updates_spec>
- All tool execution (web search, file search, query_hard_logic, code
  interpreter) happens silently. Never announce or narrate tool calls.
- Send a brief update (1–2 sentences) ONLY when:
  - You begin a new phase of the workflow.
  - You discover a finding that changes the plan or contradicts prior data.
- Each update must include at least one concrete outcome: "Confirmed X",
  "Found discrepancy in Y", "Updated Z based on live search".
- Do not expand the task beyond what the user asked. If you identify
  additional valuable work, call it out as optional.
</user_updates_spec>
```

### Rationale
The existing "never describe" rule is necessary but incomplete. GPT-5.2
benefits from knowing *when* to communicate, not just what to suppress.
This spec gives the model a positive pattern for user-facing updates.

### Dependencies
Replaces the existing CRITICAL rule at line 62. The intent is identical but
the framing is more complete.

---

## Modification 6 — Tool Usage Rules

### What GPT-5.2 guide says
Section 6: Describe tools crisply. Encourage parallelism. Require verification
for high-impact operations.

### Current state
- Lines 12–30: Detailed Hard Logic tool descriptions.
- Lines 46–48: Web search and file_search instructions.
- Lines 151–174: search_medical_links workflow.
- No explicit parallelism guidance or post-write verification pattern.

### Proposed change
Add a `<tool_usage_rules>` block to **`system_instructions.txt`** after the
existing tool descriptions (after line 30, before GENERAL RULES).

```
<tool_usage_rules>
- Prefer tools over internal knowledge whenever you need:
  - Fresh or user-specific data (drug status, regulatory filings, clinical
    trial results, market pricing).
  - Specific identifiers (NCT numbers, NDA numbers, product codes, KOL IDs).
- Parallelise independent tool calls when possible:
  - Run web_search_preview + file_search + query_hard_logic in parallel for
    landscape queries.
  - Run multiple search_medical_links calls (PubMed + FDA + EMA) in parallel.
- After any data-modifying tool call or code interpreter execution, briefly
  restate: what was computed/changed, the key result, and any follow-up
  validation performed.
- For code interpreter outputs: always include the numerical result and a
  one-line interpretation in the response.
</tool_usage_rules>
```

### Rationale
GPT-5.2 takes additional tool actions in interactive flows vs. GPT-5.1.
Explicit parallelism guidance reduces latency. Post-action verification
ensures auditability — critical in regulated environments.

### Dependencies
The Responses API already supports parallel tool calls. This instruction
guides the model to use that capability. No code change needed.

---

## Modification 7 — Compaction Integration (Code Change)

### What GPT-5.2 guide says
Section 4: Use the `/responses/compact` endpoint for long-running, tool-heavy
workflows that exceed the context window.

### Current state
- `session_manager.py` manages context via `adaptive_context_builder()` and
  token budgets.
- `core_assistant.py` uses `previous_response_id` for chaining.
- Checkpoint summaries via `gpt-4.1-mini` provide session continuity.
- No compaction integration.

### Proposed change
**A)** Add a `compact_response()` function to **`session_manager.py`** that
calls the `/responses/compact` endpoint after major milestones (e.g., end of
each phase).

**B)** Integrate the compaction call into **`core_assistant.py`** at the end of
each tool-call round when the accumulated context exceeds a threshold (e.g.,
200K tokens).

**C)** Add a `COMPACTION_THRESHOLD` to **`tool_config.py`** (suggested: 200K
tokens, or 50% of the 400K window).

### Rationale
The 5-phase workflow with multiple tool calls per phase is exactly the
scenario where compaction adds value. It extends the effective context window
without hitting limits, especially for deep research sessions.

### Dependencies
- Requires OpenAI SDK support for the compact endpoint.
- Must preserve `previous_response_id` chain after compaction.
- Checkpoint summaries remain as a fallback for cross-session continuity.

### Risk
Compacted items are opaque. If the agent needs to reference exact Phase 1
data in Phase 5, compaction may lose detail. Mitigation: only compact after
Phases 2 and 4 (not after every phase), and always keep the latest phase
uncompacted.

---

## Modification 8 — Structured Extraction Spec

### What GPT-5.2 guide says
Section 7: Always provide a schema for extraction. Distinguish required vs.
optional fields. Handle missing fields as null.

### Current state
- The agent processes PDFs via file_search and extracts data for SWOT, TOWS,
  stakeholder mapping, etc.
- No explicit extraction schema guidance in the instructions.
- `tool_config.py` has clinical outcome extraction configuration.

### Proposed change
Add an `<extraction_spec>` block to **`system_instructions.txt`** for when the
agent extracts structured data from PDFs, labels, or clinical documents.

```
<extraction_spec>
When extracting structured data from PDFs, drug labels, clinical papers, or
regulatory documents:
- Follow any user-provided schema exactly. Do not add extra fields.
- If a field is not present in the source, set it to null or mark as
  "Not found in source" — never guess.
- Before returning extracted data, re-scan the source for missed fields.
- For multi-document extraction: serialise results per document with a stable
  identifier (filename, document title, page range).
- Present extracted data in Markdown tables when practical.
</extraction_spec>
```

### Rationale
GPT-5.2 shows strong improvements in structured extraction. Explicit guidance
makes it more reliable for regulatory document analysis — a core sAImone use
case.

### Dependencies
None. Works with the existing file_search and code interpreter tools.

---

## Modification 9 — Reasoning Effort Tuning

### What GPT-5.2 guide says
Section 8: Default reasoning for GPT-5.2 is `none`. Pin reasoning effort
explicitly. Use the migration mapping table.

### Current state
- `assistant_config.py` line 82: `DEFAULT_REASONING_EFFORT = "medium"`.
- Line 83: `HIGH_REASONING_EFFORT = "high"`.
- Auto-escalation for MC/stats queries via `needs_high_reasoning()`.

### Proposed change
**A)** Review `DEFAULT_REASONING_EFFORT`. The current `"medium"` is appropriate
for a medical affairs agent (not a simple chat). Confirm via eval that `"medium"`
outperforms `"none"` and `"low"` on the typical query mix. No change expected
unless evals show regression.

**B)** Add a `"low"` reasoning tier in `assistant_config.py` for simple
factual/lookup queries (e.g., "What is the FDA approval date for X?"). Detect
via keyword heuristics similar to `needs_high_reasoning()`.

```python
LOW_REASONING_KEYWORDS = [
    "what is", "when was", "who is", "define", "list",
    "approval date", "status of", "price of",
]

def get_reasoning_effort(user_input: str) -> str:
    """Return appropriate reasoning effort for the query."""
    q = user_input.lower()
    if any(kw in q for kw in HIGH_REASONING_KEYWORDS):
        return HIGH_REASONING_EFFORT
    if any(kw in q for kw in LOW_REASONING_KEYWORDS):
        return "low"
    return DEFAULT_REASONING_EFFORT
```

### Rationale
The current binary (medium/high) works but wastes tokens on simple lookups.
A three-tier approach (low/medium/high) aligns with GPT-5.2's `reasoning_effort`
knob and reduces latency + cost on straightforward queries.

### Dependencies
- `core_assistant.py` line 1485-1494: Update to use `get_reasoning_effort()`
  instead of the current binary check.

---

## Modification 10 — Silent Instructions Refinement

### What GPT-5.2 guide says
Sections 3.1 + 5: Clamp verbosity of updates. Make scope discipline explicit.

### Current state
- `session_manager.py` has `COMPACT_SILENT_INSTRUCTIONS` and
  `FULL_SILENT_INSTRUCTIONS` injected dynamically based on query complexity.

### Proposed change
Update both instruction blocks in **`session_manager.py`** to incorporate
the verbosity and scope discipline patterns from Modifications 1 and 2.

**COMPACT version** — add:
```
SCOPE: Answer exactly what was asked. Flag optional adjacent work separately.
OUTPUT: Structured sections, ≤5 bullets each. Tables for comparisons.
```

**FULL version** — add:
```
SCOPE & VERBOSITY:
- Deliver exactly what was asked. Do not add unrequested analyses.
- Structured sections with ≤5 bullets each. Tables for comparisons/data.
- Keep meta-commentary in a standalone "Notes" section at the end.
- If adjacent work is valuable, mention as optional — do not execute.
```

### Rationale
The silent instructions are the dynamic complement to the static system
instructions. They need to reinforce the same verbosity and scope constraints
to prevent drift, especially on complex queries where FULL instructions are
injected.

### Dependencies
Must align with the `<output_verbosity_spec>` and `<scope_discipline>` blocks
added to `system_instructions.txt` (Modifications 1 and 2).

---

## Implementation Priority

| Priority | Mod # | Description | Effort | Impact |
|---|---|---|---|---|
| P0 | 1 | Output verbosity spec | Low | High — directly reduces token waste |
| P0 | 2 | Scope discipline | Low | High — prevents scope creep |
| P0 | 4 | Uncertainty safeguards | Low | High — critical for medical affairs |
| P1 | 5 | User updates spec | Low | Medium — improves UX |
| P1 | 6 | Tool usage rules | Low | Medium — reduces latency via parallelism |
| P1 | 3 | Long-context handling | Low | Medium — improves multi-phase coherence |
| P1 | 8 | Structured extraction | Low | Medium — improves PDF/label processing |
| P1 | 10 | Silent instructions update | Low | Medium — reinforces P0 changes |
| P2 | 9 | Reasoning effort tuning | Medium | Medium — cost/latency optimisation |
| P2 | 7 | Compaction integration | High | Medium — extends effective context |

---

## Implementation Order

1. **Batch 1 (P0 — instruction-only, no code):** Mods 1, 2, 4, 5
   - All changes to `system_instructions.txt`.
   - No code changes. Can be deployed immediately.
   - Estimated token impact on system_instructions.txt: +~800 tokens.
   - Verify prompt cache remains above 1024-token threshold (it will — current
     file is ~4000 tokens).

2. **Batch 2 (P1 — instruction + minor code):** Mods 3, 6, 8, 10
   - Mods 3, 6, 8: changes to `system_instructions.txt`.
   - Mod 10: changes to `session_manager.py` (silent instructions).

3. **Batch 3 (P2 — code changes + eval):** Mods 9, 7
   - Mod 9: changes to `assistant_config.py` + `core_assistant.py`.
   - Mod 7: changes to `session_manager.py` + `core_assistant.py` + `tool_config.py`.
   - Requires eval runs to validate reasoning effort tiers and compaction
     behaviour.

---

## Validation Approach

Per GPT-5.2 guide Section 8 (migration steps):

1. Apply Batch 1 instruction changes.
2. Run existing eval suite with `DEFAULT_REASONING_EFFORT = "medium"` unchanged.
3. If results are stable or improved, ship Batch 1.
4. Apply Batch 2. Re-run evals.
5. Apply Batch 3. Re-run evals with low/medium/high reasoning tiers.
6. Measure: token usage, response latency, output quality (factual accuracy,
   scope adherence, formatting compliance).

---

## What NOT to Change

- **5-phase workflow structure**: The GPT-5.2 guide doesn't suggest changing
  workflow architecture. The phases are domain-specific and should remain.
- **Hard Logic integration**: Already well-structured. No changes needed.
- **Prompt caching architecture**: Already implements the best practice of
  immutable static prefix. No changes needed.
- **Circuit breaker pattern**: Infrastructure resilience — orthogonal to
  prompting changes.
- **Tool configurations (tool_config.py)**: Result limits and tiering are
  already optimised (v4.2). Only add `COMPACTION_THRESHOLD` in Batch 3.
