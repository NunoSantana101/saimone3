"""
pipeline.py  –  3-Stage MedAffairs Inference Pipeline
=====================================================

Architecture:  Plan → Research → Synthesize

Replaces the monolithic "God Prompt" with a serial pipeline that
maximises reasoning fidelity (GPT-5.2), ensures auditability of
medical claims, and enforces strict context hygiene.

Data Flow
---------
User Query
  → [Stage 1: Architect]  → ResearchPlan  (JSON)
  → [Stage 2: Researcher] → FactSheet     (structured)
  → [Stage 3: Synthesizer] → Final Response

Stage 1 – The Architect (Reasoning Engine)
    Role:  Request Router & Logic Planner
    Input: Raw User Query + System State
    Output: ResearchPlan (typed JSON)
    Constraint: **No tools, no retrieval** – pure reasoning.

Stage 2 – The Researcher (Execution & Filtration)
    Role:  Headless Browser / Data Extractor
    Input: ResearchPlan from Stage 1
    Output: FactSheet (structured, cited)
    Constraint: Must cite Source ID for every fact.
                Missing data → null / "Data Not Available".

Stage 3 – The Synthesizer (Compliance & Tone)
    Role:  Medical Writer / Compliance Officer
    Input: Original User Query + FactSheet from Stage 2
    Output: Final natural-language response
    Constraint: FactSheet is the "Ground Truth Universe".

v9.0 – Initial 3-stage pipeline architecture.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai

_logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Schema Definitions
# ─────────────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    """Categorisation of the incoming user query."""
    EFFICACY = "efficacy"
    SAFETY = "safety"
    REGULATORY = "regulatory"
    COMPETITIVE = "competitive"
    KOL = "kol"
    MARKET_ACCESS = "market_access"
    CLINICAL_TRIAL = "clinical_trial"
    PUBLICATION = "publication"
    STAKEHOLDER = "stakeholder"
    STRATEGIC = "strategic"
    GENERAL = "general"


@dataclass
class ResearchStep:
    """A single retrieval / tool-call step to be executed in Stage 2."""
    tool: str                         # e.g. "web_search_preview", "file_search", "query_hard_logic"
    action: str                       # human-readable description of what to retrieve
    parameters: Dict[str, Any]        # tool-specific params (query, dataset, filters, …)
    priority: int = 1                 # 1 = highest, lower priority may be skipped
    fallback_tool: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {
            "tool": self.tool,
            "action": self.action,
            "parameters": self.parameters,
            "priority": self.priority,
        }
        if self.fallback_tool:
            d["fallback_tool"] = self.fallback_tool
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchStep":
        return cls(
            tool=d["tool"],
            action=d.get("action", ""),
            parameters=d.get("parameters", {}),
            priority=d.get("priority", 1),
            fallback_tool=d.get("fallback_tool"),
        )


@dataclass
class ResearchPlan:
    """Output contract of Stage 1 – The Architect.

    Strictly typed research plan that Stage 2 executes verbatim.
    """
    intent: str                                # QueryIntent value
    query_decomposition: str                   # Plain-English restatement
    required_data_points: List[str]            # e.g. ["hazard ratio", "p-value", "OS median"]
    steps: List[ResearchStep]                  # ordered tool calls
    ambiguities: List[str] = field(default_factory=list)  # flagged unclear aspects
    constraints: List[str] = field(default_factory=list)   # e.g. "sub-population X only"
    confidence_notes: str = ""                 # architect's self-assessment

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "query_decomposition": self.query_decomposition,
            "required_data_points": self.required_data_points,
            "steps": [s.to_dict() for s in self.steps],
            "ambiguities": self.ambiguities,
            "constraints": self.constraints,
            "confidence_notes": self.confidence_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchPlan":
        return cls(
            intent=d.get("intent", "general"),
            query_decomposition=d.get("query_decomposition", ""),
            required_data_points=d.get("required_data_points", []),
            steps=[ResearchStep.from_dict(s) for s in d.get("steps", [])],
            ambiguities=d.get("ambiguities", []),
            constraints=d.get("constraints", []),
            confidence_notes=d.get("confidence_notes", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Fact:
    """A single retrieved datum with mandatory provenance."""
    data_point: str          # which required_data_point this answers
    value: Any               # the extracted value (str, number, None)
    source_id: str           # citation key (URL, DOI, dataset name, …)
    source_label: str        # human-readable source name
    confidence: str = "high"  # "high" | "medium" | "low" | "not_available"
    excerpt: str = ""        # verbatim supporting quote (≤300 chars)

    def to_dict(self) -> dict:
        return {
            "data_point": self.data_point,
            "value": self.value,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "confidence": self.confidence,
            "excerpt": self.excerpt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            data_point=d.get("data_point", ""),
            value=d.get("value"),
            source_id=d.get("source_id", "unknown"),
            source_label=d.get("source_label", ""),
            confidence=d.get("confidence", "high"),
            excerpt=d.get("excerpt", ""),
        )


@dataclass
class FactSheet:
    """Output contract of Stage 2 – The Researcher.

    Every fact must carry a source_id.  Missing data is represented
    with value=None and confidence="not_available".
    """
    query_intent: str
    facts: List[Fact]
    data_gaps: List[str] = field(default_factory=list)    # requested but not found
    sources_consulted: List[str] = field(default_factory=list)
    retrieval_timestamp: str = ""
    total_tool_calls: int = 0

    def __post_init__(self):
        if not self.retrieval_timestamp:
            self.retrieval_timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "query_intent": self.query_intent,
            "facts": [f.to_dict() for f in self.facts],
            "data_gaps": self.data_gaps,
            "sources_consulted": self.sources_consulted,
            "retrieval_timestamp": self.retrieval_timestamp,
            "total_tool_calls": self.total_tool_calls,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactSheet":
        return cls(
            query_intent=d.get("query_intent", "general"),
            facts=[Fact.from_dict(f) for f in d.get("facts", [])],
            data_gaps=d.get("data_gaps", []),
            sources_consulted=d.get("sources_consulted", []),
            retrieval_timestamp=d.get("retrieval_timestamp", ""),
            total_tool_calls=d.get("total_tool_calls", 0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Render the fact sheet as a compact Markdown block for Stage 3."""
        lines = [
            f"## Fact Sheet  (retrieved {self.retrieval_timestamp})",
            f"**Intent:** {self.query_intent}",
            "",
        ]
        for i, fact in enumerate(self.facts, 1):
            val = fact.value if fact.value is not None else "_Data Not Available_"
            lines.append(
                f"{i}. **{fact.data_point}**: {val}  "
                f"[{fact.source_label}]({fact.source_id})  "
                f"(confidence: {fact.confidence})"
            )
            if fact.excerpt:
                lines.append(f"   > {fact.excerpt}")
        if self.data_gaps:
            lines.append("")
            lines.append("### Data Gaps")
            for gap in self.data_gaps:
                lines.append(f"- {gap}")
        return "\n".join(lines)


@dataclass
class PipelineAuditTrail:
    """Full audit trail for a single pipeline execution."""
    pipeline_id: str
    user_query: str
    started_at: str
    stage_1_plan: Optional[dict] = None
    stage_1_duration_ms: int = 0
    stage_2_fact_sheet: Optional[dict] = None
    stage_2_duration_ms: int = 0
    stage_2_tool_calls: List[dict] = field(default_factory=list)
    stage_3_duration_ms: int = 0
    total_duration_ms: int = 0
    final_response_length: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "user_query": self.user_query,
            "started_at": self.started_at,
            "stage_1_plan": self.stage_1_plan,
            "stage_1_duration_ms": self.stage_1_duration_ms,
            "stage_2_fact_sheet": self.stage_2_fact_sheet,
            "stage_2_duration_ms": self.stage_2_duration_ms,
            "stage_2_tool_calls": self.stage_2_tool_calls,
            "stage_3_duration_ms": self.stage_3_duration_ms,
            "total_duration_ms": self.total_duration_ms,
            "final_response_length": self.final_response_length,
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────────
#  Stage 1: The Architect  (Reasoning Engine)
# ─────────────────────────────────────────────────────────────────────

# System prompt for Stage 1 — reasoning + hard logic context.
_ARCHITECT_SYSTEM_PROMPT = """\
You are the Architect stage of a medical affairs inference pipeline.

ROLE: Request Router & Logic Planner.
You receive a raw user query and must produce a structured research plan.

RESPONSIBILITIES:
1. Deconstruct complex medical queries (e.g. "efficacy vs safety in sub-population X").
2. Identify ambiguities and formulate a strict step-by-step retrieval strategy.
3. Determine the specific data points needed (hazard ratios, p-values, regulatory dates, etc.).
4. Specify which tools should be called and with what parameters.
5. Use query_hard_logic to inspect internal config schemas (pillars, metrics,
   tactics, stakeholders, etc.) so your plan references real dataset fields and
   valid filter values.

TOOL ACCESS:
You have access to ONE tool — query_hard_logic — for inspecting the internal
structured configuration data.  Use it to look up available datasets, schema
fields, metric definitions, pillar names, stakeholder tiers, etc. so your
research plan is grounded in real data structures.

CRITICAL CONSTRAINTS:
- Do NOT have access to web search, file search, or any external sources.
- Do NOT attempt to answer the query or provide medical information.
- Do NOT hallucinate data or facts.
- Your ONLY job is to produce a research plan.
- Use query_hard_logic ONLY to inform your plan structure — do not use it
  to answer the user's question directly.

AVAILABLE TOOLS FOR STAGE 2 (reference only — you cannot call them):
- web_search_preview: Live web search for current data (PubMed, FDA, EMA, clinical trials, etc.)
- file_search: Vector store search for internal documents, PDFs, proprietary references
- code_interpreter: Sandboxed Python for statistical analysis, Monte Carlo, Bayesian inference
- query_hard_logic: In-memory pandas DataFrames for structured config data
  (datasets: metrics, pillars, stakeholders, tactics, roles, data_sources, kol,
   pricing_market_access, competitive_intelligence, comparison_guide, auth_sources, vr)
- run_statistical_analysis: Monte Carlo / Bayesian / sensitivity analysis
- monte_carlo_simulation: Specific Monte Carlo simulation
- bayesian_analysis: Bayesian inference for evidence synthesis

OUTPUT FORMAT:
After any query_hard_logic calls, respond with ONLY a valid JSON object matching this schema:
{
  "intent": "<query category: efficacy|safety|regulatory|competitive|kol|market_access|clinical_trial|publication|stakeholder|strategic|general>",
  "query_decomposition": "<plain-English restatement of what the user is asking>",
  "required_data_points": ["<specific metric 1>", "<specific metric 2>", ...],
  "steps": [
    {
      "tool": "<tool_name>",
      "action": "<what to retrieve>",
      "parameters": { <tool-specific params> },
      "priority": <1-3, 1=highest>,
      "fallback_tool": "<optional alternative tool>"
    }
  ],
  "ambiguities": ["<unclear aspect 1>", ...],
  "constraints": ["<scope limitation 1>", ...],
  "confidence_notes": "<self-assessment of plan quality>"
}

Do NOT wrap in markdown code fences.  Return raw JSON only.\
"""


def _get_architect_tools() -> List[dict]:
    """Return the limited tool set for Stage 1 (query_hard_logic only)."""
    from hard_logic import QUERY_HARD_LOGIC_TOOL
    return [QUERY_HARD_LOGIC_TOOL]


def run_stage_1_architect(
    user_query: str,
    *,
    system_state: Optional[Dict[str, Any]] = None,
    client: Optional[Any] = None,
    model: str = "gpt-5.2",
    tool_router: Optional[Callable] = None,
    max_tool_rounds: int = 5,
) -> ResearchPlan:
    """Stage 1: Decompose the user query into a structured research plan.

    Uses GPT-5.2 with HIGH reasoning effort.  Has access ONLY to the
    query_hard_logic tool so it can inspect internal config schemas
    (pillars, metrics, tactics, stakeholders, etc.) and ground the
    plan in real data structures.  No web search, no file search,
    no external retrieval.

    Args:
        user_query: The raw user query.
        system_state: Optional dict with current_date, user_role, etc.
        client: OpenAI client instance (uses get_client() if None).
        model: Model to use (default: gpt-5.2).
        tool_router: Function tool router (defaults to core_assistant's).
        max_tool_rounds: Max hard-logic lookup round-trips (default 5).

    Returns:
        ResearchPlan — strictly typed plan for Stage 2.

    Raises:
        PipelineStageError: If the model fails to produce valid JSON.
    """
    if client is None:
        from core_assistant import get_client
        client = get_client()
    if tool_router is None:
        from core_assistant import _default_tool_router
        tool_router = _default_tool_router

    tools = _get_architect_tools()

    # Build the dynamic input (timestamps, role, etc.)
    state = system_state or {}
    now_iso = datetime.now(timezone.utc).isoformat()
    now_long = datetime.now(timezone.utc).strftime("%B %d, %Y")

    state_block = (
        f"Current Date: {now_long} ({now_iso})\n"
        f"User Role: {state.get('user_role', 'Medical Affairs Professional')}\n"
    )
    if state.get("therapy_area"):
        state_block += f"Therapy Area Context: {state['therapy_area']}\n"
    if state.get("region"):
        state_block += f"Region: {state['region']}\n"

    input_text = (
        f"SYSTEM STATE:\n{state_block}\n"
        f"USER QUERY:\n{user_query}"
    )

    _logger.info("Stage 1 (Architect): planning for query (len=%d)", len(user_query))

    # Note: store=True (default) is required for tool-enabled calls so that
    # previous_response_id chaining works in the tool-call continuation loop.
    try:
        response = client.responses.create(
            model=model,
            instructions=_ARCHITECT_SYSTEM_PROMPT,
            input=input_text,
            tools=tools,
            reasoning={"effort": "high"},
            text={"format": {"type": "json_object"}},
        )
    except openai.APIError as exc:
        raise PipelineStageError("architect", f"API error: {exc}") from exc

    # Tool call loop — only query_hard_logic is available
    for _round in range(max_tool_rounds):
        tool_calls = _get_tool_calls_from_response(response)
        if not tool_calls:
            break

        tool_outputs: List[dict] = []
        for call in tool_calls:
            fn_name = call.name
            call_id = call.call_id
            try:
                args = json.loads(call.arguments)
            except json.JSONDecodeError:
                args = {}

            try:
                output = tool_router(fn_name, args)
            except Exception as exc:
                output = json.dumps({"error": str(exc)})

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })

        try:
            response = client.responses.create(
                model=model,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=tools,
                instructions=_ARCHITECT_SYSTEM_PROMPT,
                reasoning={"effort": "high"},
                text={"format": {"type": "json_object"}},
            )
        except openai.APIError as exc:
            raise PipelineStageError("architect", f"API error in tool loop: {exc}") from exc

    # Extract text from response
    raw_text = _extract_text(response)
    if not raw_text:
        raise PipelineStageError("architect", "Empty response from model")

    # Parse JSON
    try:
        plan_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Try to extract JSON from markdown fences
        plan_data = _extract_json_from_text(raw_text)
        if plan_data is None:
            raise PipelineStageError(
                "architect",
                f"Invalid JSON from model: {exc}\nRaw: {raw_text[:500]}",
            ) from exc

    plan = ResearchPlan.from_dict(plan_data)
    _logger.info(
        "Stage 1 complete: intent=%s, steps=%d, data_points=%d",
        plan.intent, len(plan.steps), len(plan.required_data_points),
    )
    return plan


# ─────────────────────────────────────────────────────────────────────
#  Stage 2: The Researcher  (Execution & Filtration)
# ─────────────────────────────────────────────────────────────────────

# System prompt for Stage 2 — tool execution + context compression.
_RESEARCHER_SYSTEM_PROMPT = """\
You are the Researcher stage of a medical affairs inference pipeline.

ROLE: Headless Data Extractor & Context Compressor.
You receive a research plan (JSON) and must execute it, then produce
a structured Fact Sheet.

RESPONSIBILITIES:
1. Execute the tool calls defined in the research plan.
2. Perform aggressive "Context Compression": extract ONLY the data
   points requested in the plan.  Discard all irrelevant text.
3. Cite the Source ID for every extracted fact.
4. If data is missing, explicitly flag it as "Data Not Available".

CRITICAL CONSTRAINTS:
- Execute tools IN ORDER of priority (priority 1 first).
- If a tool fails, try the fallback_tool if specified.
- Do NOT synthesise, interpret, or editorialize.  Just extract facts.
- Do NOT include data the plan did not ask for.
- Every fact MUST have a source_id (URL, DOI, dataset name, etc.).
- Keep excerpts to ≤300 characters.

OUTPUT FORMAT:
After executing all steps, respond with ONLY a valid JSON object:
{
  "query_intent": "<from the plan>",
  "facts": [
    {
      "data_point": "<which required_data_point this answers>",
      "value": "<extracted value or null if not found>",
      "source_id": "<URL, DOI, or dataset identifier>",
      "source_label": "<human-readable source name>",
      "confidence": "high|medium|low|not_available",
      "excerpt": "<verbatim supporting quote, ≤300 chars>"
    }
  ],
  "data_gaps": ["<requested data_point not found>", ...],
  "sources_consulted": ["<source 1>", "<source 2>", ...]
}

Do NOT wrap in markdown code fences.  Return raw JSON only.\
"""


def run_stage_2_researcher(
    plan: ResearchPlan,
    *,
    client: Optional[Any] = None,
    model: str = "gpt-5.2",
    tool_router: Optional[Callable] = None,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
    timeout: int = 300,
    max_tool_rounds: int = 15,
) -> Tuple[FactSheet, List[dict]]:
    """Stage 2: Execute the research plan and produce a fact sheet.

    The model receives the plan as a JSON prompt and is given access
    to all available tools.  It executes the steps, compresses the
    retrieved context, and outputs a structured FactSheet.

    Args:
        plan: ResearchPlan from Stage 1.
        client: OpenAI client instance.
        model: Model to use.
        tool_router: Function tool router (defaults to core_assistant's).
        on_tool_call: Optional callback for UI feedback.
        timeout: Max seconds for the entire stage.
        max_tool_rounds: Max tool-call round-trips.

    Returns:
        (FactSheet, tool_call_log) — structured facts + audit trail.
    """
    if client is None:
        from core_assistant import get_client
        client = get_client()
    if tool_router is None:
        from core_assistant import _default_tool_router
        tool_router = _default_tool_router

    from core_assistant import get_tools, _get_tool_calls, reset_container

    tools = get_tools()

    input_text = (
        f"RESEARCH PLAN:\n{plan.to_json()}\n\n"
        f"Execute every step above using the available tools. "
        f"Extract ONLY the required_data_points listed.  "
        f"Produce the Fact Sheet JSON as specified."
    )

    _logger.info(
        "Stage 2 (Researcher): executing %d steps for intent=%s",
        len(plan.steps), plan.intent,
    )

    # Import rate-limit backoff schedule from core
    from core_assistant import _RATE_LIMIT_BACKOFF_SCHEDULE, _PER_TOOL_TIMEOUT
    import concurrent.futures

    tool_call_log: List[dict] = []
    start = time.time()

    # Note: store=True (default) is required for tool-enabled calls so that
    # previous_response_id chaining works in the tool-call continuation loop.
    try:
        response = client.responses.create(
            model=model,
            instructions=_RESEARCHER_SYSTEM_PROMPT,
            input=input_text,
            tools=tools,
            reasoning={"effort": "medium"},
            text={"format": {"type": "json_object"}},
        )
    except openai.BadRequestError as exc:
        _logger.warning("Stage 2 BadRequestError — resetting container: %s", exc)
        reset_container()
        tools = get_tools()
        try:
            response = client.responses.create(
                model=model,
                instructions=_RESEARCHER_SYSTEM_PROMPT,
                input=input_text,
                tools=tools,
                reasoning={"effort": "medium"},
                text={"format": {"type": "json_object"}},
            )
        except openai.BadRequestError:
            raise PipelineStageError("researcher", f"API error after retry: {exc}") from exc
    except openai.RateLimitError as exc:
        response = _retry_with_backoff(
            client, model, _RESEARCHER_SYSTEM_PROMPT, input_text,
            tools, _RATE_LIMIT_BACKOFF_SCHEDULE, exc, "researcher",
        )
    except openai.APIError as exc:
        raise PipelineStageError("researcher", f"API error: {exc}") from exc

    # Tool call loop
    for round_num in range(max_tool_rounds):
        if time.time() - start > timeout:
            _logger.warning("Stage 2 timed out after %ds", timeout)
            break

        tool_calls = _get_tool_calls(response)
        if not tool_calls:
            break

        tool_outputs: List[dict] = []
        for call in tool_calls:
            fn_name = call.name
            call_id = call.call_id
            try:
                args = json.loads(call.arguments)
            except json.JSONDecodeError:
                args = {}

            if on_tool_call:
                on_tool_call(fn_name, args)

            # Execute with per-tool timeout
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(tool_router, fn_name, args)
                    output = future.result(timeout=_PER_TOOL_TIMEOUT)
            except concurrent.futures.TimeoutError:
                output = json.dumps({"error": f"Tool {fn_name} timed out"})
            except Exception as exc:
                output = json.dumps({"error": str(exc)})

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })
            tool_call_log.append({
                "name": fn_name,
                "args": args,
                "output_size": len(output),
                "round": round_num,
                "stage": "researcher",
            })

        # Continue with tool outputs
        try:
            response = client.responses.create(
                model=model,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=tools,
                instructions=_RESEARCHER_SYSTEM_PROMPT,
                reasoning={"effort": "medium"},
                text={"format": {"type": "json_object"}},
            )
        except openai.BadRequestError as exc:
            _logger.warning("Stage 2 tool loop BadRequestError — resetting: %s", exc)
            reset_container()
            tools = get_tools()
            try:
                response = client.responses.create(
                    model=model,
                    previous_response_id=response.id,
                    input=tool_outputs,
                    tools=tools,
                    instructions=_RESEARCHER_SYSTEM_PROMPT,
                    reasoning={"effort": "medium"},
                    text={"format": {"type": "json_object"}},
                )
            except openai.BadRequestError:
                raise PipelineStageError("researcher", f"Tool loop error: {exc}") from exc
        except openai.RateLimitError as rl_exc:
            _cont = None
            for delay in _RATE_LIMIT_BACKOFF_SCHEDULE:
                time.sleep(delay)
                try:
                    _cont = client.responses.create(
                        model=model,
                        previous_response_id=response.id,
                        input=tool_outputs,
                        tools=tools,
                        instructions=_RESEARCHER_SYSTEM_PROMPT,
                        reasoning={"effort": "medium"},
                        text={"format": {"type": "json_object"}},
                    )
                    break
                except openai.RateLimitError:
                    continue
            if _cont is None:
                raise PipelineStageError("researcher", f"Rate limited: {rl_exc}") from rl_exc
            response = _cont
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            time.sleep(2)
            try:
                response = client.responses.create(
                    model=model,
                    previous_response_id=response.id,
                    input=tool_outputs,
                    tools=tools,
                    instructions=_RESEARCHER_SYSTEM_PROMPT,
                    reasoning={"effort": "medium"},
                    text={"format": {"type": "json_object"}},
                )
            except Exception as retry_exc:
                raise PipelineStageError("researcher", f"Retry failed: {retry_exc}") from retry_exc

    # Parse the final FactSheet
    raw_text = _extract_text(response)
    if not raw_text:
        raise PipelineStageError("researcher", "Empty response from researcher")

    try:
        sheet_data = json.loads(raw_text)
    except json.JSONDecodeError:
        sheet_data = _extract_json_from_text(raw_text)
        if sheet_data is None:
            raise PipelineStageError(
                "researcher",
                f"Invalid JSON from researcher: {raw_text[:500]}",
            )

    sheet = FactSheet.from_dict(sheet_data)
    sheet.total_tool_calls = len(tool_call_log)

    _logger.info(
        "Stage 2 complete: facts=%d, gaps=%d, tool_calls=%d",
        len(sheet.facts), len(sheet.data_gaps), len(tool_call_log),
    )
    return sheet, tool_call_log


# ─────────────────────────────────────────────────────────────────────
#  Stage 3: The Synthesizer  (Compliance & Tone)
# ─────────────────────────────────────────────────────────────────────

# System prompt for Stage 3 — writing from fact sheet only.
_SYNTHESIZER_SYSTEM_PROMPT = """\
You are the Synthesizer stage of a medical affairs inference pipeline.

ROLE: Medical Writer & Compliance Officer.
You receive the original user query and a verified Fact Sheet, and must
draft the final natural-language response.

RESPONSIBILITIES:
1. Draft a clear, scientifically accurate response using ONLY the
   provided Fact Sheet as your source of truth.
2. Apply the requested tone and formatting.
3. Cite sources inline using the source labels from the Fact Sheet.
4. Flag any Data Gaps transparently.
5. Include a References section at the end.

CRITICAL CONSTRAINTS — THE GROUND TRUTH BOUNDARY:
- The Fact Sheet is your COMPLETE universe of knowledge for this response.
- If a datum is NOT in the Fact Sheet, it DOES NOT EXIST.  Do NOT add
  information from your training data or general knowledge.
- If asked about something not covered by the Fact Sheet, state:
  "This information was not available in the retrieved sources."
- Do NOT speculate, infer, or extrapolate beyond what the Fact Sheet provides.
- Every claim in your response must map to a specific Fact in the sheet.

TONE & FORMATTING:
- Scientific Objectivity: neutral, evidence-based language.
- Structure: use headers, bullet points, and tables where appropriate.
- Standard Response Document format for regulatory/compliance queries.
- Include a "Data Gaps & Limitations" section if any gaps exist.
- End with a "References" section listing all cited sources.
- Include the standard disclaimer footer.

DISCLAIMER FOOTER (always include):
---
*This response was generated by sAImone, an AI-powered Medical Affairs
assistant.  All claims are sourced from the referenced materials.
This is not medical advice.  Verify critical information with primary
sources and consult qualified professionals for clinical decisions.*\
"""


def run_stage_3_synthesizer(
    user_query: str,
    fact_sheet: FactSheet,
    *,
    client: Optional[Any] = None,
    model: str = "gpt-5.2",
    output_type: str = "Standard Response Document",
    response_tone: str = "Scientific Objectivity",
    compliance_level: str = "high",
    previous_response_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Stage 3: Synthesize the final response from the fact sheet.

    The model receives ONLY the original query and the fact sheet.
    It has NO tools and must not introduce information beyond what
    the fact sheet contains.

    Args:
        user_query: Original user query (for context).
        fact_sheet: FactSheet from Stage 2.
        client: OpenAI client instance.
        model: Model to use.
        output_type: Desired output format.
        response_tone: Tone modifier.
        compliance_level: Compliance stringency.
        previous_response_id: For response chaining (optional).

    Returns:
        (response_text, response_id) — final text + ID for chaining.
    """
    if client is None:
        from core_assistant import get_client
        client = get_client()

    # Build the fact sheet as Markdown for the synthesizer
    fact_md = fact_sheet.to_markdown()

    input_text = (
        f"ORIGINAL USER QUERY:\n{user_query}\n\n"
        f"FORMATTING INSTRUCTIONS:\n"
        f"- Output Type: {output_type}\n"
        f"- Response Tone: {response_tone}\n"
        f"- Compliance Level: {compliance_level}\n\n"
        f"FACT SHEET (your ONLY source of truth):\n{fact_md}\n\n"
        f"Draft the final response.  Use ONLY the facts above."
    )

    _logger.info(
        "Stage 3 (Synthesizer): composing response (facts=%d, gaps=%d)",
        len(fact_sheet.facts), len(fact_sheet.data_gaps),
    )

    kwargs: dict = {
        "model": model,
        "instructions": _SYNTHESIZER_SYSTEM_PROMPT,
        "input": input_text,
        # NO tools — writing only
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "medium"},
        "store": False,
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    try:
        response = client.responses.create(**kwargs)
    except openai.APIError as exc:
        raise PipelineStageError("synthesizer", f"API error: {exc}") from exc

    text = _extract_text(response)
    response_id = response.id if hasattr(response, "id") else None

    _logger.info(
        "Stage 3 complete: response_length=%d", len(text),
    )
    return text, response_id


# ─────────────────────────────────────────────────────────────────────
#  Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    *,
    model: str = "gpt-5.2",
    system_state: Optional[Dict[str, Any]] = None,
    output_type: str = "Standard Response Document",
    response_tone: str = "Scientific Objectivity",
    compliance_level: str = "high",
    previous_response_id: Optional[str] = None,
    tool_router: Optional[Callable] = None,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
    on_stage_complete: Optional[Callable[[str, Any], None]] = None,
    timeout: int = 600,
) -> Tuple[str, Optional[str], PipelineAuditTrail]:
    """Run the full 3-stage MedAffairs inference pipeline.

    Orchestrates:  Architect → Researcher → Synthesizer

    Args:
        user_query: Raw user query.
        model: GPT model to use across all stages.
        system_state: Optional context (user_role, therapy_area, etc.).
        output_type: Desired output format for Stage 3.
        response_tone: Tone modifier for Stage 3.
        compliance_level: Compliance stringency for Stage 3.
        previous_response_id: For response chaining (passed to Stage 3).
        tool_router: Function tool router for Stage 2.
        on_tool_call: Callback for UI feedback during Stage 2.
        on_stage_complete: Callback fired after each stage completes,
            receives (stage_name, stage_output).
        timeout: Max total seconds for the entire pipeline.

    Returns:
        (response_text, response_id, audit_trail)
    """
    import uuid

    pipeline_id = f"pipeline-{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(timezone.utc).isoformat()
    pipeline_start = time.time()

    audit = PipelineAuditTrail(
        pipeline_id=pipeline_id,
        user_query=user_query,
        started_at=started_at,
    )

    _logger.info("Pipeline %s started for query (len=%d)", pipeline_id, len(user_query))

    # ── Stage 1: Architect ──
    try:
        s1_start = time.time()
        plan = run_stage_1_architect(
            user_query,
            system_state=system_state,
            model=model,
            tool_router=tool_router,
        )
        audit.stage_1_plan = plan.to_dict()
        audit.stage_1_duration_ms = int((time.time() - s1_start) * 1000)

        if on_stage_complete:
            on_stage_complete("architect", plan)

        _logger.info(
            "Pipeline %s Stage 1 done in %dms",
            pipeline_id, audit.stage_1_duration_ms,
        )
    except PipelineStageError:
        raise
    except Exception as exc:
        audit.error = f"Stage 1 failed: {exc}"
        raise PipelineStageError("architect", str(exc)) from exc

    # ── Check total timeout ──
    if time.time() - pipeline_start > timeout:
        audit.error = "Timeout after Stage 1"
        raise PipelineStageError("pipeline", "Timeout after Stage 1")

    # ── Stage 2: Researcher ──
    try:
        s2_start = time.time()
        remaining_timeout = max(60, timeout - int(time.time() - pipeline_start))
        fact_sheet, tool_log = run_stage_2_researcher(
            plan,
            model=model,
            tool_router=tool_router,
            on_tool_call=on_tool_call,
            timeout=remaining_timeout,
        )
        audit.stage_2_fact_sheet = fact_sheet.to_dict()
        audit.stage_2_tool_calls = tool_log
        audit.stage_2_duration_ms = int((time.time() - s2_start) * 1000)

        if on_stage_complete:
            on_stage_complete("researcher", fact_sheet)

        _logger.info(
            "Pipeline %s Stage 2 done in %dms (%d tool calls)",
            pipeline_id, audit.stage_2_duration_ms, len(tool_log),
        )
    except PipelineStageError:
        raise
    except Exception as exc:
        audit.error = f"Stage 2 failed: {exc}"
        raise PipelineStageError("researcher", str(exc)) from exc

    # ── Check total timeout ──
    if time.time() - pipeline_start > timeout:
        audit.error = "Timeout after Stage 2"
        raise PipelineStageError("pipeline", "Timeout after Stage 2")

    # ── Stage 3: Synthesizer ──
    try:
        s3_start = time.time()
        response_text, response_id = run_stage_3_synthesizer(
            user_query,
            fact_sheet,
            model=model,
            output_type=output_type,
            response_tone=response_tone,
            compliance_level=compliance_level,
            previous_response_id=previous_response_id,
        )
        audit.stage_3_duration_ms = int((time.time() - s3_start) * 1000)
        audit.final_response_length = len(response_text)

        if on_stage_complete:
            on_stage_complete("synthesizer", response_text)

        _logger.info(
            "Pipeline %s Stage 3 done in %dms (response=%d chars)",
            pipeline_id, audit.stage_3_duration_ms, len(response_text),
        )
    except PipelineStageError:
        raise
    except Exception as exc:
        audit.error = f"Stage 3 failed: {exc}"
        raise PipelineStageError("synthesizer", str(exc)) from exc

    # ── Finalize ──
    audit.total_duration_ms = int((time.time() - pipeline_start) * 1000)

    _logger.info(
        "Pipeline %s completed in %dms (S1=%dms, S2=%dms, S3=%dms)",
        pipeline_id,
        audit.total_duration_ms,
        audit.stage_1_duration_ms,
        audit.stage_2_duration_ms,
        audit.stage_3_duration_ms,
    )

    return response_text, response_id, audit


# ─────────────────────────────────────────────────────────────────────
#  Pipeline Error Types
# ─────────────────────────────────────────────────────────────────────

class PipelineStageError(Exception):
    """Error in a specific pipeline stage."""
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"Pipeline stage '{stage}' failed: {message}")


# ─────────────────────────────────────────────────────────────────────
#  Utility Helpers
# ─────────────────────────────────────────────────────────────────────

def _extract_text(response) -> str:
    """Extract text content from a Responses API response object."""
    if not hasattr(response, "output") or not response.output:
        return ""
    parts: List[str] = []
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) == "output_text":
                    parts.append(block.text)
    return "\n".join(parts) if parts else ""


def _get_tool_calls_from_response(response) -> list:
    """Extract function_call items from a Responses API response."""
    calls = []
    if not hasattr(response, "output") or not response.output:
        return calls
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            calls.append(item)
    return calls


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Try to extract a JSON object from text that may be wrapped in fences."""
    import re
    # Try raw parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try to extract from markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find any JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _retry_with_backoff(
    client, model, instructions, input_text, tools,
    backoff_schedule, original_exc, stage_name,
) -> Any:
    """Retry an API call with exponential backoff on rate-limit."""
    response = None
    for delay in backoff_schedule:
        _logger.warning("Stage %s rate limited; backing off %ds", stage_name, delay)
        time.sleep(delay)
        try:
            # store=True (default) required when tools are present so
            # previous_response_id chaining works in tool-call loops.
            kwargs: dict = {
                "model": model,
                "instructions": instructions,
                "input": input_text,
                "reasoning": {"effort": "medium"},
                "text": {"format": {"type": "json_object"}},
            }
            if tools:
                kwargs["tools"] = tools
            response = client.responses.create(**kwargs)
            break
        except openai.RateLimitError:
            continue
    if response is None:
        raise PipelineStageError(
            stage_name,
            f"Rate limited after {len(backoff_schedule)} retries: {original_exc}",
        )
    return response


# ─────────────────────────────────────────────────────────────────────
#  Pipeline Mode Detection
# ─────────────────────────────────────────────────────────────────────

# Queries matching these patterns benefit from the structured pipeline.
# Simple greetings, meta-questions, etc. are better served by the
# monolithic path (faster, less overhead).
_PIPELINE_TRIGGER_KEYWORDS = [
    "efficacy", "safety", "adverse event", "hazard ratio", "p-value",
    "regulatory", "fda", "ema", "nice", "approval", "submission",
    "clinical trial", "phase ", "endpoint", "primary outcome",
    "competitive", "comparator", "head-to-head", "market share",
    "kol", "key opinion leader", "investigator",
    "market access", "reimbursement", "hta", "pricing",
    "publication", "real-world evidence", "rwe",
    "stakeholder", "patient advocacy",
    "strategy", "plan", "strategic",
    "monte carlo", "bayesian", "simulation", "statistical",
    "sub-population", "subgroup", "biomarker",
    "landscape", "analysis", "comprehensive",
]

# Queries below this character length are unlikely to need the pipeline.
_PIPELINE_MIN_QUERY_LENGTH = 30


def should_use_pipeline(user_query: str) -> bool:
    """Determine whether a query should use the 3-stage pipeline.

    Returns True for complex medical queries that benefit from
    structured planning and citation-tracked retrieval.
    Returns False for simple queries, greetings, and meta-questions
    that are better handled by the direct (monolithic) path.
    """
    if len(user_query) < _PIPELINE_MIN_QUERY_LENGTH:
        return False
    q_lower = user_query.lower()
    return any(kw in q_lower for kw in _PIPELINE_TRIGGER_KEYWORDS)
