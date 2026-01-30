"""
core_assistant.py
Pure-python engine shared by the Streamlit UI (assistant.py) and any CLI or
batch runner.  NO Streamlit or console I/O; it just raises exceptions.

v7.3 – mc_rng.py Code Interpreter Registration:
- Registers mc_rng.py (file-8ofv14JxUdJ2pSTjDxuFHT) as a code interpreter
  file via a sandbox container for direct import in Python execution
- Container is lazily created and cached; falls back to system instructions
  file loading if container API is unavailable
- All 6 tools: web_search_preview, file_search, code_interpreter,
  run_statistical_analysis, monte_carlo_simulation, bayesian_analysis

v7.2 – Code Interpreter:
- Activates OpenAI's built-in code_interpreter tool so the model can
  execute Python code in a sandboxed environment for quantitative analysis,
  Monte Carlo simulations, Bayesian inference, and data visualisation
- All 6 tools: web_search_preview, file_search, code_interpreter,
  run_statistical_analysis, monte_carlo_simulation, bayesian_analysis

v7.1 – Vector Store file_search:
- Connects vector store vs_693fe785b1a081918f82e9f903e008ed via built-in
  file_search tool so the Responses API can retrieve config JSONs, PDFs,
  and proprietary reference files at runtime

v7.0 – OpenAI Built-in Web Search (test):
- Replaces custom med_affairs_data / Tavily backend with OpenAI's
  built-in web_search_preview tool (search_context_size="high")
- Model manages all web search internally – no function-call routing
- Removed: get_med_affairs_data, read_webpage, pipeline compression tools
- Retained: statistical analysis function tools (Monte Carlo, Bayesian)
- Purpose: evaluate GPT-5.2 native search quality vs custom Tavily pipeline

v6.0 – GPT-5.2 Upgrade:
- Model upgraded from gpt-4.1 to gpt-5.2
- 400K token context window (up from 1M effective)
- 128K max output tokens (up from 32K)
- Adaptive reasoning with dynamic compute allocation
- Knowledge cutoff: August 31, 2025

v5.0 – OpenAI Responses API Migration:
- Replaces thread-based Assistants API with stateless Responses API
- Single client.responses.create() call per interaction
- Conversation continuity via previous_response_id

Key architecture:
- No threads, no runs, no polling
- response_id replaces thread_id for continuity
- Tool definitions passed inline with each request
- Instructions passed directly (no server-side assistant config)
"""

from __future__ import annotations

import json, time, os, logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import openai
from openai import OpenAI

# v7.0: med_affairs_data backend unhooked – using OpenAI web_search_preview
# from med_affairs_data import extract_clinical_outcomes

# v7.0: Agent data pipeline unhooked – no custom search results to compress
# from agent_data_pipeline import (
#     process_through_pipeline, process_with_hybrid_pipeline,
#     decompress_pipeline_data, search_manifest,
#     cache_pipeline_data, get_cached_pipeline_data,
#     cache_vector_store_info, get_cached_vector_store_info,
#     cleanup_vector_store, handle_pipeline_tool_call,
#     select_pipeline_strategy, PipelineStrategy,
#     MAX_UNCOMPRESSED_OUTPUT, VECTOR_STORE_THRESHOLD,
# )

from tool_config import (
    DEFAULT_MAX_PER_SOURCE,
    DEFAULT_MAX_TOTAL,
    RESULT_FIELD_LIMITS,
    RESULT_TIERS,
    MESH_METADATA_LIMITS,
    INTENT_MESH_LIMITS,
    REFINEMENT_SUGGESTION_CONFIG,
    QueryIntent,
    detect_query_intent,
    get_truncation_limits,
    get_preserve_fields,
    should_extract_clinical_outcomes,
)

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Output Size Limits
# ──────────────────────────────────────────────────────────────────────
MAX_TOOL_OUTPUT_BYTES = 200_000  # 200KB safety limit
PROGRESSIVE_TRUNCATION_THRESHOLDS = [
    (180_000, 0.9),
    (160_000, 0.75),
    (140_000, 0.6),
    (120_000, 0.5),
]

# ──────────────────────────────────────────────────────────────────────
#  OpenAI Client – lazy-init so main.py can set the API key first
# ──────────────────────────────────────────────────────────────────────
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Return a shared OpenAI client (created lazily)."""
    global _client
    if _client is None:
        # Prefer explicit key from openai module (set by main.py),
        # then fall back to OPENAI_API_KEY env var.
        api_key = getattr(openai, "api_key", None) or os.environ.get("OPENAI_API_KEY")
        _client = OpenAI(api_key=api_key)
    return _client


def reset_client():
    """Force re-creation of the client (e.g. after API key change)."""
    global _client
    _client = None


# ──────────────────────────────────────────────────────────────────────
#  Default model & temperature
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0.7

# ──────────────────────────────────────────────────────────────────────
#  Vector Store – OpenAI file_search
# ──────────────────────────────────────────────────────────────────────
VECTOR_STORE_ID = "vs_693fe785b1a081918f82e9f903e008ed"

# ──────────────────────────────────────────────────────────────────────
#  Code Interpreter – mc_rng.py file registration
# ──────────────────────────────────────────────────────────────────────
MC_RNG_FILE_ID = "file-P2EgMJJmLDWSJqZnKBoiBJ"

_ci_container_id: Optional[str] = None


def _get_code_interpreter_container() -> Optional[str]:
    """Lazy-create a sandbox container with mc_rng.py for code interpreter.

    Returns the container ID, or None if creation fails (falls back to
    system-instructions-based file loading).
    """
    global _ci_container_id
    if _ci_container_id is not None:
        return _ci_container_id
    try:
        client = get_client()
        container = client.containers.create(
            name="saimone-mc-rng",
            file_ids=[MC_RNG_FILE_ID],
        )
        _ci_container_id = container.id
        _logger.info(
            "Created CI container %s with mc_rng.py (%s)",
            _ci_container_id,
            MC_RNG_FILE_ID,
        )
        return _ci_container_id
    except Exception as exc:
        _logger.warning(
            "Could not create CI container: %s – "
            "mc_rng.py loaded via instructions fallback",
            exc,
        )
        return None


# ──────────────────────────────────────────────────────────────────────
#  Tool Definitions for Responses API
# ──────────────────────────────────────────────────────────────────────

# v7.0: _load_tool_schema_v3 removed – no longer loading custom tool schema
# Web search is handled by OpenAI's built-in web_search_preview tool.


def build_tools_list() -> List[dict]:
    """
    Build the complete tools list for Responses API.

    v7.3: Registers mc_rng.py (MC_RNG_FILE_ID) with the code_interpreter
    tool via a sandbox container so the file is directly importable in
    the Python execution environment.  Falls back to the system-instructions
    file-copy protocol if container creation is unavailable.

    v7.2: Added code_interpreter tool for sandboxed Python execution.
    Enables the model to run Monte Carlo simulations, Bayesian inference,
    statistical analysis, and data visualisation directly.

    v7.1: Added file_search tool connected to vector store
    vs_693fe785b1a081918f82e9f903e008ed for config JSONs, PDFs, and
    proprietary reference files.

    v7.0: Uses OpenAI's built-in web_search_preview tool instead of custom
    med_affairs_data / Tavily backend.  The model manages all web search
    internally; no function-call routing is needed for search.
    Statistical analysis function tools are retained.
    """
    tools: List[dict] = []

    # 1. OpenAI built-in web search (v7.0)
    #    - Model decides when to search, what to query, and synthesizes results
    #    - search_context_size "high" maximises context fed back to the model
    #    - No function_call is emitted; search happens server-side
    tools.append({
        "type": "web_search_preview",
        "search_context_size": "high",
    })

    # 2. OpenAI built-in file_search (v7.1)
    #    - Searches the vector store for attached config JSONs, PDFs,
    #      system instructions, and any previously uploaded content
    #    - Vector store: vs_693fe785b1a081918f82e9f903e008ed
    tools.append({
        "type": "file_search",
        "vector_store_ids": [VECTOR_STORE_ID],
    })

    # 3. OpenAI built-in code_interpreter (v7.3)
    #    - Sandboxed Python execution environment
    #    - Used for Monte Carlo simulations, Bayesian inference,
    #      statistical modelling, and data visualisation
    #    - mc_rng.py (MC_RNG_FILE_ID) registered via container for sandbox access
    #    - No function_call routing needed; execution is server-side
    ci_tool: dict = {"type": "code_interpreter"}
    _container = _get_code_interpreter_container()
    if _container:
        ci_tool["container"] = _container
    tools.append(ci_tool)

    # 4. run_statistical_analysis – Monte Carlo & Bayesian (function tool)
    tools.append({
        "type": "function",
        "name": "run_statistical_analysis",
        "description": (
            "Run statistical analysis including Monte Carlo simulations and "
            "Bayesian inference for medical affairs scenarios."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to run.",
                    "enum": ["monte_carlo", "bayesian_inference", "sensitivity_analysis"],
                },
                "therapy_area": {
                    "type": "string",
                    "description": "Therapy area for the analysis.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Analysis-specific parameters.",
                },
            },
            "required": ["analysis_type"],
        },
    })

    # 5. monte_carlo_simulation – specific MC function
    tools.append({
        "type": "function",
        "name": "monte_carlo_simulation",
        "description": (
            "Run Monte Carlo simulation for medical affairs parameter sampling "
            "and scenario analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "therapy_area": {
                    "type": "string",
                    "description": "Therapy area (e.g., oncology, cardiology).",
                },
                "region": {
                    "type": "string",
                    "description": "Target region (e.g., US, EU, LATAM).",
                },
                "lifecycle_phase": {
                    "type": "string",
                    "description": "Product lifecycle phase.",
                    "enum": ["pre_launch", "launch", "growth", "mature", "loe"],
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of simulation iterations. Default: 1000.",
                    "default": 1000,
                },
                "scenarios": {
                    "type": "object",
                    "description": "Custom scenario parameters for simulation.",
                },
            },
            "required": ["therapy_area"],
        },
    })

    # 6. bayesian_analysis – specific Bayesian function
    tools.append({
        "type": "function",
        "name": "bayesian_analysis",
        "description": (
            "Run Bayesian inference analysis for evidence synthesis and "
            "probability estimation in medical affairs contexts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "therapy_area": {
                    "type": "string",
                    "description": "Therapy area for analysis.",
                },
                "evidence": {
                    "type": "object",
                    "description": "Evidence data for Bayesian updating.",
                },
                "priors": {
                    "type": "object",
                    "description": "Prior distribution parameters.",
                },
            },
            "required": ["therapy_area"],
        },
    })

    return tools


# Cache the tools list (loaded once)
_cached_tools: Optional[List[dict]] = None


def get_tools() -> List[dict]:
    """Return cached tools list."""
    global _cached_tools
    if _cached_tools is None:
        _cached_tools = build_tools_list()
    return _cached_tools


# ──────────────────────────────────────────────────────────────────────
#  System Instructions for Responses API
#  Loaded from system_instructions.txt (same file used by workflow_agents.py)
# ──────────────────────────────────────────────────────────────────────

_INSTRUCTIONS_PATH = os.path.join(os.path.dirname(__file__), "system_instructions.txt")

try:
    with open(_INSTRUCTIONS_PATH, encoding="utf-8") as _f:
        SYSTEM_INSTRUCTIONS = _f.read()
    _logger.info("Loaded system instructions from %s (%d chars)", _INSTRUCTIONS_PATH, len(SYSTEM_INSTRUCTIONS))
except FileNotFoundError:
    _logger.warning("system_instructions.txt not found at %s – using fallback", _INSTRUCTIONS_PATH)
    SYSTEM_INSTRUCTIONS = "You are sAImone, an expert Medical Affairs AI assistant powered by GPT-5.2."


# ──────────────────────────────────────────────────────────────────────
#  Lightweight token & context helpers
# ──────────────────────────────────────────────────────────────────────
def estimate_tokens_simple(content: Any) -> int:
    """Cheap 0.25-chars ~ 1 token estimate – fast enough for budgeting."""
    if isinstance(content, list):
        return sum(len(str(item)) // 4 for item in content)
    return len(str(content)) // 4


def adaptive_medcomms_context(
    history: List[Dict[str, str]], *, token_budget: int = 24_000
) -> List[Dict[str, str]]:
    """Return a subset of *history* that fits *token_budget*.

    Responses API context management:
    - Conversation history is managed via previous_response_id chain
    - Local context is supplementary for prompt enrichment
    - Last 8 exchanges included for good continuity
    """
    # Use optimized version if available
    try:
        from session_manager import get_optimized_context
        return get_optimized_context(history, token_budget)
    except ImportError:
        pass

    if not history:
        return []

    must_include = history[-8:] if len(history) >= 8 else history
    remaining = token_budget - estimate_tokens_simple(must_include)

    if len(history) <= 8:
        return must_include

    priority: Dict[str, List[Dict[str, str]]] = {
        "document_analysis": [], "regulatory_decisions": [], "compliance": [],
        "kol_research": [], "market_analysis": [], "general": [],
    }
    for msg in history[:-8]:
        t = msg["content"].lower()
        if any(k in t for k in ("upload", "document", "pdf", "file", "review")):
            priority["document_analysis"].append(msg)
        elif any(k in t for k in ("fda", "ema", "regulatory", "approval")):
            priority["regulatory_decisions"].append(msg)
        elif any(k in t for k in ("compliance", "gdpr", "clinical trial")):
            priority["compliance"].append(msg)
        elif any(k in t for k in ("kol", "investigator", "opinion leader")):
            priority["kol_research"].append(msg)
        elif any(k in t for k in ("market", "competitive", "launch", "strategy", "roi")):
            priority["market_analysis"].append(msg)
        else:
            priority["general"].append(msg)

    chosen: List[Dict[str, str]] = []
    for bucket in (
        "document_analysis", "regulatory_decisions", "compliance",
        "kol_research", "market_analysis", "general",
    ):
        for m in reversed(priority[bucket]):
            cost = estimate_tokens_simple([m])
            if remaining - cost < 100:
                break
            chosen.append(m)
            remaining -= cost

    chosen.sort(key=lambda x: history.index(x))
    chosen.extend(must_include)
    return chosen


def create_context_prompt_with_budget(
    user_input: str,
    output_type: str,
    response_tone: str,
    compliance_level: str,
    role: str,
    client: str,
    history: List[Dict[str, str]],
    token_budget: int,
    *,
    has_files: bool,
) -> str:
    """Compose the user-facing prompt while honouring *token_budget*."""
    context = adaptive_medcomms_context(history, token_budget=token_budget)
    now_iso = datetime.utcnow().isoformat()
    now_long = datetime.utcnow().strftime("%B %d, %Y")

    file_note = (
        "\n\nREFERENCE FILES AVAILABLE - use them when relevant."
        if has_files else ""
    )

    return (
        f"SYSTEM CONTEXT - {now_iso}\nToday's date: {now_long}\n\n"
        f"SESSION_CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        f"{file_note}\n\n"
        "INSTRUCTIONS:\n"
        f"- Output Type: {output_type}\n"
        f"- Response Tone: {response_tone}\n"
        f"- Compliance Level: {compliance_level}\n"
        f"- Token Budget: {token_budget:,}\n\n"
        f"USER_QUERY: {user_input}"
    )


# ──────────────────────────────────────────────────────────────────────
#  Result Truncation Helper
# ──────────────────────────────────────────────────────────────────────
def _truncate_search_results(
    res: dict,
    max_per_source: int = None,
    max_total: int = None,
    query: str = None,
    intent: QueryIntent = None,
) -> dict:
    """
    v4.0 - Token-efficient tiered result processing.

    Key changes from v3.x:
    1. DEDUP - cross-source duplicates removed by normalised title
    2. FLAT RANKED LIST - results_by_source flattened; source becomes a field
    3. TIERED CONTENT - top-N get full text, tail gets headline-only
    4. SUMMARY HEADER - one-paragraph overview for agent reasoning head-start
    5. URLS STRIPPED - unless the agent/user explicitly requests citations
    6. MESH / REFINEMENTS - not included inline (lazy-loadable via tools)
    """
    if not isinstance(res, dict):
        return res

    if query and not intent:
        intent = detect_query_intent(query)

    limits = get_truncation_limits(intent) if intent else get_truncation_limits()

    max_per_source = max_per_source or limits.get("max_per_source", DEFAULT_MAX_PER_SOURCE)
    max_total = max_total or limits.get("max_total", DEFAULT_MAX_TOTAL)

    title_max = RESULT_FIELD_LIMITS.get("title_max_chars", 150)
    extra_field_max = RESULT_FIELD_LIMITS.get("extra_field_max_chars", 200)
    extra_list_max = RESULT_FIELD_LIMITS.get("extra_list_max_items", 4)

    t1_count = RESULT_TIERS.get("tier_1_count", 5)
    t1_content = RESULT_TIERS.get("tier_1_content", 600)
    t2_count = RESULT_TIERS.get("tier_2_count", 10)
    t2_content = RESULT_TIERS.get("tier_2_content", 250)
    t3_content = RESULT_TIERS.get("tier_3_content", 0)

    preserve_fields = get_preserve_fields(intent)
    extract_outcomes = should_extract_clinical_outcomes(intent)

    # -- 1. Collect all results into flat list with source tag
    raw_items: list = []

    if "results_by_source" in res:
        for source, results in res.get("results_by_source", {}).items():
            if not isinstance(results, list):
                continue
            for r in results[:max_per_source]:
                raw_items.append((r, source))
    elif "results" in res and isinstance(res["results"], list):
        for r in res["results"]:
            src = r.get("source", r.get("_source", "unknown"))
            raw_items.append((r, src))

    # -- 2. Deduplicate by normalised title
    seen_titles: set = set()
    deduped: list = []
    duplicates_removed = 0

    for r, source in raw_items:
        title_raw = r.get("title", "")
        norm = title_raw.lower().strip()
        norm = "".join(c for c in norm if c.isalnum() or c == " ")
        norm = " ".join(norm.split())

        if norm and norm in seen_titles:
            duplicates_removed += 1
            continue
        if norm:
            seen_titles.add(norm)
        deduped.append((r, source))

    # -- 3. Rank: priority sources first, then by authority/date
    priority_sources = limits.get("priority_sources", [])

    def _rank_key(item_tuple):
        r, source = item_tuple
        try:
            src_rank = priority_sources.index(source)
        except ValueError:
            src_rank = len(priority_sources)
        authority = -(r.get("_authority", 0) or 0)
        return (src_rank, authority)

    deduped.sort(key=_rank_key)
    final_items = deduped[:max_total]
    total_available = len(deduped)

    # -- 4. Per-result slimming with tiered content depth
    def _process_result(r: dict, source: str, rank: int) -> dict:
        extra = r.get("extra", {})

        if rank < t1_count:
            content_budget = t1_content
        elif rank < t1_count + t2_count:
            content_budget = t2_content
        else:
            content_budget = t3_content

        content = (
            r.get("snippet")
            or extra.get("abstract")
            or extra.get("content")
            or ""
        )
        if isinstance(content, str):
            content = content.strip()
        if content and content_budget > 0 and len(content) > content_budget:
            content = content[:content_budget] + "..."
        elif content_budget == 0:
            content = ""

        slim: dict = {}

        title = r.get("title", "")
        if title:
            slim["title"] = title[:title_max]
        slim["source"] = source

        result_url = r.get("url") or r.get("extra", {}).get("url")
        if result_url and rank < t1_count + t2_count:
            slim["url"] = result_url

        if content:
            slim["content"] = content

        date = r.get("date") or extra.get("date")
        if date:
            slim["date"] = date

        if rank < t1_count + t2_count:
            for field_name in preserve_fields:
                val = extra.get(field_name) or r.get(field_name)
                if not val:
                    continue
                if field_name in ("content", "abstract", "snippet", "id",
                                  "url", "title", "date", "source"):
                    continue
                if isinstance(val, str) and len(val) > extra_field_max:
                    slim[field_name] = val[:extra_field_max] + "..."
                elif isinstance(val, list) and len(val) > extra_list_max:
                    slim[field_name] = val[:extra_list_max]
                else:
                    slim[field_name] = val

        if extract_outcomes and content and rank < t1_count:
            try:
                from med_affairs_data import extract_clinical_outcomes
                outcomes = extract_clinical_outcomes(content)
                if outcomes:
                    slim["outcomes"] = outcomes
            except ImportError:
                pass  # v7.0: med_affairs_data not in active path

        return slim

    processed = [
        _process_result(r, source, rank)
        for rank, (r, source) in enumerate(final_items)
    ]

    # -- 5. Build summary header
    source_counts: dict = {}
    for _, source in final_items:
        source_counts[source] = source_counts.get(source, 0) + 1
    sources_line = ", ".join(f"{s}:{n}" for s, n in source_counts.items())

    summary = {
        "query": query or "",
        "total_returned": len(processed),
    }
    if total_available > len(processed):
        summary["total_available"] = total_available
    if duplicates_removed > 0:
        summary["duplicates_removed"] = duplicates_removed
    summary["sources"] = sources_line

    urls_available = sum(1 for r in processed if r.get("url"))
    if urls_available > 0:
        summary["tip"] = (
            "Content above is truncated. If 1-2 results look particularly "
            "relevant, call read_webpage with the url to get the full page text."
        )

    # -- 6. MeSH: drug context one-liner (only for safety/reg)
    mesh_limits = dict(MESH_METADATA_LIMITS)
    if intent and intent.value in INTENT_MESH_LIMITS:
        mesh_limits.update(INTENT_MESH_LIMITS[intent.value])

    include_mesh = mesh_limits.get("include_inline", False)

    if include_mesh and "mesh_metadata" in res and isinstance(res["mesh_metadata"], dict):
        mesh = res["mesh_metadata"]
        dm = mesh.get("drug_mapping")
        if isinstance(dm, dict):
            drug_ctx = {}
            ind_limit = mesh_limits.get("drug_mapping_indications", 3)
            mech_limit = mesh_limits.get("drug_mapping_mechanism", 2)
            if dm.get("indications"):
                drug_ctx["indications"] = dm["indications"][:ind_limit]
            if dm.get("mechanism"):
                drug_ctx["mechanism"] = dm["mechanism"][:mech_limit]
            if drug_ctx:
                summary["drug_context"] = drug_ctx

    # -- 7. Assemble final output
    output: dict = {
        "summary": summary,
        "results": processed,
    }

    return output


def _enforce_output_size_limit(res: dict, max_bytes: int = MAX_TOOL_OUTPUT_BYTES, query: str = None) -> dict:
    """
    Enforce hard output size limit with progressive truncation.
    """
    if not isinstance(res, dict):
        return res

    try:
        serialized = json.dumps(res)
        current_size = len(serialized.encode('utf-8'))
    except (TypeError, ValueError) as e:
        _logger.error(f"JSON serialization failed: {e}")
        return {"error": "Serialization failed", "query": query}

    if current_size <= max_bytes:
        return res

    _logger.warning(f"Output size {current_size} bytes exceeds limit {max_bytes}, applying progressive truncation")

    for threshold, factor in PROGRESSIVE_TRUNCATION_THRESHOLDS:
        if current_size <= threshold:
            break

        if "results_by_source" in res:
            for source, results in res.get("results_by_source", {}).items():
                if isinstance(results, list):
                    new_count = max(3, int(len(results) * factor))
                    res["results_by_source"][source] = results[:new_count]

        if "results" in res and isinstance(res["results"], list):
            new_count = max(5, int(len(res["results"]) * factor))
            res["results"] = res["results"][:new_count]

        serialized = json.dumps(res)
        current_size = len(serialized.encode('utf-8'))

        if current_size <= max_bytes * 0.9:
            break

    if current_size > max_bytes:
        _logger.warning(f"Still over limit ({current_size} bytes), applying aggressive field stripping")
        res = _aggressive_truncation(res, max_bytes)
        serialized = json.dumps(res)
        current_size = len(serialized.encode('utf-8'))

    if current_size > max_bytes:
        _logger.error(f"Output still exceeds limit after truncation ({current_size} > {max_bytes})")
        res = _minimal_response(res, query)

    return res


def _aggressive_truncation(res: dict, max_bytes: int) -> dict:
    """Aggressive truncation when progressive reduction isn't enough."""
    keep_keys = {"summary", "results", "query", "source", "total_results", "drug_context"}
    for key in list(res.keys()):
        if key not in keep_keys:
            del res[key]

    def _slim(r: dict) -> dict:
        out: dict = {}
        if r.get("title"):
            out["title"] = str(r["title"])[:100]
        if r.get("source"):
            out["source"] = r["source"]
        content = r.get("content") or ""
        if content:
            out["content"] = str(content)[:200]
        return out

    if "results" in res and isinstance(res["results"], list):
        res["results"] = [_slim(r) for r in res["results"][:15]]

    return res


def _minimal_response(res: dict, query: str = None) -> dict:
    """Last resort - return count + message so the agent can refine."""
    result_count = len(res.get("results", []))
    return {
        "error": "Results truncated due to size limits",
        "query": query,
        "total_results_found": result_count,
        "message": "Too many results to return. Please refine your query or use more specific filters.",
    }


# ──────────────────────────────────────────────────────────────────────
#  Default (sync) tool router
# ──────────────────────────────────────────────────────────────────────
def _default_tool_router(name: str, args: Dict[str, Any]) -> str:
    """Routes function-call tools to backend handlers.

    v7.0: Web search is handled by OpenAI's built-in web_search_preview
    tool (server-side, no routing needed).  Only statistical analysis
    function tools are routed here.
    """

    # ── Statistical analysis tools ──
    if name == "run_statistical_analysis":
        try:
            from mc_bayesian_backend import handle_statistical_analysis_function_call
            result = handle_statistical_analysis_function_call(args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:
            _logger.error(f"Statistical analysis error: {exc}")
            return json.dumps({"error": str(exc)})

    if name == "monte_carlo_simulation":
        try:
            from mc_bayesian_backend import run_statistical_analysis
            parameters = {
                "therapy_area": args.get("therapy_area", "general"),
                "region": args.get("region", "US"),
                "lifecycle_phase": args.get("lifecycle_phase", "launch"),
                "iterations": args.get("iterations", 1000),
                "scenarios": args.get("scenarios"),
            }
            result = run_statistical_analysis("monte_carlo", parameters)
            return json.dumps(result)
        except Exception as exc:
            _logger.error(f"Monte Carlo error: {exc}")
            return json.dumps({"error": str(exc)})

    if name == "bayesian_analysis":
        try:
            from mc_bayesian_backend import run_statistical_analysis
            parameters = {
                "therapy_area": args.get("therapy_area", "general"),
                "evidence": args.get("evidence", {}),
                "priors": args.get("priors"),
            }
            result = run_statistical_analysis("bayesian_inference", parameters)
            return json.dumps(result)
        except Exception as exc:
            _logger.error(f"Bayesian analysis error: {exc}")
            return json.dumps({"error": str(exc)})

    return json.dumps({"error": f"Unknown function {name}"})


# ──────────────────────────────────────────────────────────────────────
#  Responses API Helpers
# ──────────────────────────────────────────────────────────────────────

def _extract_response_text(response) -> str:
    """Extract text content from a Responses API response object."""
    if not hasattr(response, "output") or not response.output:
        return "[No response generated]"

    text_parts: List[str] = []
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for content_block in getattr(item, "content", []):
                if getattr(content_block, "type", None) == "output_text":
                    text_parts.append(content_block.text)
    return "\n".join(text_parts) if text_parts else "[No response generated]"


def _get_tool_calls(response) -> list:
    """Extract function_call items from a Responses API response."""
    calls = []
    if not hasattr(response, "output") or not response.output:
        return calls
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            calls.append(item)
    return calls


# ──────────────────────────────────────────────────────────────────────
#  Blocking runner – Responses API (replaces run_assistant_sync)
# ──────────────────────────────────────────────────────────────────────
def run_responses_sync(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    input_messages: Optional[List[dict]] = None,
    input_text: Optional[str] = None,
    instructions: str = SYSTEM_INSTRUCTIONS,
    previous_response_id: Optional[str] = None,
    tool_router: Callable[[str, Dict[str, Any]], str] = _default_tool_router,
    timeout: int = 600,
    max_tool_rounds: int = 20,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
) -> Tuple[str, Optional[str], List[dict]]:
    """
    Synchronous runner using the OpenAI Responses API.

    Returns:
        (response_text, response_id, tool_call_log)
        - response_text: The assistant's final text output
        - response_id: The response ID for conversation continuity
        - tool_call_log: List of {name, args, output} dicts for audit
    """
    client = get_client()
    tools = get_tools()

    # Build input
    if input_text and not input_messages:
        api_input = input_text
    elif input_messages:
        api_input = input_messages
    else:
        raise ValueError("Either input_text or input_messages must be provided")

    tool_call_log: List[dict] = []
    start = time.time()

    # Create initial response
    try:
        kwargs = {
            "model": model,
            "temperature": temperature,
            "input": api_input,
            "tools": tools,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response = client.responses.create(**kwargs)
    except openai.BadRequestError as e:
        # previous_response_id may be stale/invalid — signal caller to clear it
        raise RuntimeError(f"API error (400): {str(e)}")
    except openai.NotFoundError as e:
        raise RuntimeError(f"Resource not found: {str(e)}")
    except openai.RateLimitError as e:
        # Retry once after backoff
        time.sleep(2)
        try:
            response = client.responses.create(**kwargs)
        except Exception:
            raise RuntimeError(f"Rate limited: {str(e)}")
    except openai.APIConnectionError as e:
        raise RuntimeError(f"Connection error: {str(e)}")
    except openai.APITimeoutError as e:
        raise RuntimeError(f"API timeout: {str(e)}")
    except openai.APIStatusError as e:
        raise RuntimeError(f"API status error ({e.status_code}): {str(e)}")

    # Tool call loop – keep going until we get a text response or hit limits
    for round_num in range(max_tool_rounds):
        if time.time() - start > timeout:
            raise TimeoutError(f"Response exceeded {timeout}s timeout")

        tool_calls = _get_tool_calls(response)

        if not tool_calls:
            # No tool calls – we have a final text response
            break

        # Process each tool call
        tool_outputs: List[dict] = []
        for call in tool_calls:
            fn_name = call.name
            call_id = call.call_id
            try:
                args = json.loads(call.arguments)
            except json.JSONDecodeError as exc:
                args = {}
                _logger.error(f"Failed to parse args for {fn_name}: {exc}")

            # Notify callback if provided (for UI spinners etc.)
            if on_tool_call:
                on_tool_call(fn_name, args)

            # Execute tool
            try:
                output = tool_router(fn_name, args)
            except Exception as exc:
                _logger.error(f"Tool execution error for {fn_name}: {exc}")
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
            })

        # Continue conversation with tool outputs
        try:
            response = client.responses.create(
                model=model,
                temperature=temperature,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=tools,
                instructions=instructions,
            )
        except openai.BadRequestError as e:
            raise RuntimeError(f"Tool output submission failed (400): {str(e)}")
        except openai.RateLimitError:
            time.sleep(2)
            try:
                response = client.responses.create(
                    model=model,
                    temperature=temperature,
                    previous_response_id=response.id,
                    input=tool_outputs,
                    tools=tools,
                    instructions=instructions,
                )
            except Exception as retry_exc:
                raise RuntimeError(f"Tool output submission failed after rate-limit retry: {retry_exc}")
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            # Retry once on transient network errors
            _logger.warning(f"Transient error during tool output submission: {e}, retrying...")
            time.sleep(2)
            try:
                response = client.responses.create(
                    model=model,
                    temperature=temperature,
                    previous_response_id=response.id,
                    input=tool_outputs,
                    tools=tools,
                    instructions=instructions,
                )
            except Exception as retry_exc:
                raise RuntimeError(f"Tool output submission failed after retry: {retry_exc}")

    # Extract final text
    text = _extract_response_text(response)
    response_id = response.id if hasattr(response, "id") else None

    return text, response_id, tool_call_log


# ──────────────────────────────────────────────────────────────────────
#  Async runner – Responses API (replaces run_assistant_async)
# ──────────────────────────────────────────────────────────────────────
async def run_responses_async(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    input_messages: Optional[List[dict]] = None,
    input_text: Optional[str] = None,
    instructions: str = SYSTEM_INSTRUCTIONS,
    previous_response_id: Optional[str] = None,
    tool_router_async: Optional[Callable[[str, Dict[str, Any]], Coroutine[Any, Any, str]]] = None,
    timeout: int = 600,
    max_tool_rounds: int = 20,
) -> Tuple[str, Optional[str], List[dict]]:
    """
    Async runner using the OpenAI Responses API.

    Returns:
        (response_text, response_id, tool_call_log)
    """
    import asyncio

    tool_router_async = tool_router_async or (
        lambda n, a: asyncio.get_running_loop().run_in_executor(None, _default_tool_router, n, a)
    )

    client = get_client()
    tools = get_tools()

    if input_text and not input_messages:
        api_input = input_text
    elif input_messages:
        api_input = input_messages
    else:
        raise ValueError("Either input_text or input_messages must be provided")

    tool_call_log: List[dict] = []
    start = time.time()

    # Initial response (run in thread pool since sync client)
    kwargs = {
        "model": model,
        "temperature": temperature,
        "input": api_input,
        "tools": tools,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    response = await asyncio.to_thread(client.responses.create, **kwargs)

    # Tool call loop
    for round_num in range(max_tool_rounds):
        if time.time() - start > timeout:
            raise TimeoutError(f"Response exceeded {timeout}s timeout")

        tool_calls = _get_tool_calls(response)

        if not tool_calls:
            break

        async def _one(call):
            fn_name = call.name
            call_id = call.call_id
            try:
                args = json.loads(call.arguments)
            except json.JSONDecodeError:
                args = {}
            output = await tool_router_async(fn_name, args)
            tool_call_log.append({
                "name": fn_name,
                "args": args,
                "output_size": len(output),
                "round": round_num,
            })
            return {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }

        tool_outputs = await asyncio.gather(*[_one(c) for c in tool_calls])

        response = await asyncio.to_thread(
            client.responses.create,
            model=model,
            temperature=temperature,
            previous_response_id=response.id,
            input=list(tool_outputs),
            tools=tools,
            instructions=instructions,
        )

    text = _extract_response_text(response)
    response_id = response.id if hasattr(response, "id") else None

    return text, response_id, tool_call_log
