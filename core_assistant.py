"""
core_assistant.py
Pure‑python engine shared by the Streamlit UI (assistant.py) and any CLI or
batch runner.  NO Streamlit or console I/O; it just raises exceptions.

Thread-based context management:
- OpenAI threads store full conversation history server-side via thread_id
- Local context is supplementary for checkpointing and emergency recovery
- Uses GPT-4.1 throughout for consistency

Improvements:
- Integration with session_manager for circuit breaker support
- Cached validation where possible
- Thread-centric context management
- Increased token budgets for richer responses

v3.3 Agent Data Pipeline:
- Structured data schemas for consistency
- Compression (gzip+base64) for large payloads
- Data manifests with searchable summaries
- Decompression tools for agent access
"""

from __future__ import annotations

import json, time, openai
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

# v3.0 truncation + intent metadata helpers
from med_affairs_data import extract_clinical_outcomes

# v3.3 Agent Data Pipeline (v1.1 Hybrid with Vector Store)
from agent_data_pipeline import (
    process_through_pipeline,
    process_with_hybrid_pipeline,
    decompress_pipeline_data,
    search_manifest,
    cache_pipeline_data,
    get_cached_pipeline_data,
    cache_vector_store_info,
    get_cached_vector_store_info,
    cleanup_vector_store,
    handle_pipeline_tool_call,
    select_pipeline_strategy,
    PipelineStrategy,
    MAX_UNCOMPRESSED_OUTPUT,
    VECTOR_STORE_THRESHOLD,
)

from tool_config import (
    DEFAULT_MAX_PER_SOURCE,
    DEFAULT_MAX_TOTAL,
    DEFAULT_MAX_REFINEMENT_SUGGESTIONS,
    MESH_METADATA_LIMITS,
    RESULT_FIELD_LIMITS,
    QueryIntent,
    detect_query_intent,
    get_truncation_limits,
    get_mesh_limits,
    get_preserve_fields,
    should_extract_clinical_outcomes,
)

# ──────────────────────────────────────────────────────────────────────
#  Output Size Limits - Prevents agent silent failures from truncation
#  OpenAI's submit_tool_outputs has undocumented limits (~256KB per output)
# ──────────────────────────────────────────────────────────────────────
MAX_TOOL_OUTPUT_BYTES = 200_000  # 200KB safety limit (OpenAI allows ~256KB)
PROGRESSIVE_TRUNCATION_THRESHOLDS = [
    (180_000, 0.9),   # At 180KB, reduce to 90% of results
    (160_000, 0.75),  # At 160KB, reduce to 75%
    (140_000, 0.6),   # At 140KB, reduce to 60%
    (120_000, 0.5),   # At 120KB, reduce to 50%
]

import logging
_logger = logging.getLogger(__name__)
# Import session management utilities (optional - graceful fallback)
try:
    from session_manager import (
        validate_thread_exists_cached,
        get_circuit_breaker,
        get_optimized_context,
    )
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────
#  Lightweight token & context helpers (lifted verbatim from assistant.py)
# ──────────────────────────────────────────────────────────────────────
def estimate_tokens_simple(content: Any) -> int:
    """Cheap 0·25‑chars ≃ 1 token estimate – fast enough for budgeting."""
    if isinstance(content, list):
        return sum(len(str(item)) // 4 for item in content)
    return len(str(content)) // 4


def adaptive_medcomms_context(
    history: List[Dict[str, str]], *, token_budget: int = 24_000
) -> List[Dict[str, str]]:
    """Return a subset of *history* that fits *token_budget*.

    Thread-based context management:
    - OpenAI threads store full conversation history server-side
    - This provides supplementary local context
    - thread_id is the primary context reference

    GPT-4.1 Optimized:
    - Default budget increased to 24k for richer responses
    - Last 8 exchanges included for good continuity (increased from 6)
    - Balances context quality with response speed

    If session_manager is available, uses optimized version with:
    - Deduplication
    - Caching
    - Better categorization
    """
    # Use optimized version if available
    if SESSION_MANAGER_AVAILABLE:
        return get_optimized_context(history, token_budget)

    # Fallback to original implementation
    if not history:
        return []

    # Include last 8 exchanges for good continuity (thread has full history)
    must_include = history[-8:] if len(history) >= 8 else history
    remaining = token_budget - estimate_tokens_simple(must_include)

    if len(history) <= 8:
        return must_include

    priority: Dict[str, List[Dict[str, str]]] = {
        "document_analysis": [], "regulatory_decisions": [], "compliance": [],
        "kol_research": [], "market_analysis": [], "general": [],
    }
    for msg in history[:-8]:  # Exclude last 8 (already in must_include)
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
        for m in reversed(priority[bucket]):          # oldest first
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
    """Compose full system prompt while honouring *token_budget*."""
    context = adaptive_medcomms_context(history, token_budget=token_budget)
    now_iso  = datetime.utcnow().isoformat()
    now_long = datetime.utcnow().strftime("%B %d, %Y")

    mem_note = (
        "You MAY call retrieve_memory_context if the user explicitly references past sessions."
        if any("past medical session" in m["content"].lower() for m in history[-3:])
        else "Call retrieve_memory_context **only** when the user clearly asks about past conversations."
    )
    file_note = (
        "\n\nREFERENCE FILES AVAILABLE – use them when relevant."
        if has_files else ""
    )

    return (
        f"SYSTEM CONTEXT – {now_iso}\nToday's date: {now_long}\n\n"
        f"SESSION_CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        f"{mem_note}{file_note}\n\n"
        "INSTRUCTIONS:\n"
        f"- Output Type: {output_type}\n"
        f"- Response Tone: {response_tone}\n"
        f"- Compliance Level: {compliance_level}\n"
        f"- Token Budget: {token_budget:,}\n\n"
        f"USER_QUERY: {user_input}"
    )

# ──────────────────────────────────────────────────────────────────────
#  Thread validation & guard – avoids "run already active" and stale thread errors
# ──────────────────────────────────────────────────────────────────────
def validate_thread_exists(thread_id: str) -> Tuple[bool, str]:
    """
    Check if a thread exists and is accessible on OpenAI servers.
    Returns (is_valid, error_message).

    If session_manager is available, uses cached validation to reduce API calls.
    """
    # Use cached validation if session manager is available
    if SESSION_MANAGER_AVAILABLE:
        return validate_thread_exists_cached(thread_id)

    # Fallback to direct validation
    try:
        openai.beta.threads.retrieve(thread_id)
        return True, ""
    except openai.NotFoundError:
        return False, f"Thread {thread_id} not found - it may have been deleted or expired"
    except openai.BadRequestError as e:
        return False, f"Thread {thread_id} is invalid: {str(e)}"
    except Exception as e:
        # For network errors etc., assume thread might be valid
        return True, f"Warning: Could not validate thread: {str(e)}"


def wait_for_idle_thread(thread_id: str, *, poll: float = 1.0, timeout: int = 120) -> None:
    """Block until *thread_id* has no active run (queued/in_progress/requires_action/cancelling)."""
    # Statuses that indicate an active run that blocks new operations
    ACTIVE_STATUSES = {"queued", "in_progress", "requires_action", "cancelling"}

    start = time.time()
    while time.time() - start < timeout:
        try:
            runs = openai.beta.threads.runs.list(thread_id=thread_id, limit=1)
            if not runs.data or runs.data[0].status not in ACTIVE_STATUSES:
                return
        except openai.NotFoundError:
            raise RuntimeError(f"Thread {thread_id} not found - please start a new session")
        except openai.BadRequestError as e:
            raise RuntimeError(f"Invalid thread {thread_id}: {str(e)}")
        time.sleep(poll)
    raise RuntimeError("Thread never became idle within allotted time")

# ──────────────────────────────────────────────────────────────────────
#  Result Truncation Helper - Prevents "output string too long" 400 errors
# ──────────────────────────────────────────────────────────────────────
def _truncate_search_results(
    res: dict,
    max_per_source: int = None,
    max_total: int = None,
    query: str = None,
    intent: QueryIntent = None,
) -> dict:
    """
    Truncate search results while preserving agent-critical context.

    v3.0 PHILOSOPHY:
    - Backend provides RICH, structured data
    - Agent handles interpretation and prioritization
    - Preserve intent-specific fields (phase, status, outcomes)
    - Extract clinical outcomes from abstracts/snippets
    - Include search transparency metadata for audit trail
    """
    if not isinstance(res, dict):
        return res

    if query and not intent:
        intent = detect_query_intent(query)

    if intent:
        limits = get_truncation_limits(intent)
        mesh_limits = get_mesh_limits(intent)
    else:
        limits = get_truncation_limits()
        mesh_limits = MESH_METADATA_LIMITS

    max_per_source = max_per_source or limits.get("max_per_source", DEFAULT_MAX_PER_SOURCE)
    max_total = max_total or limits.get("max_total", DEFAULT_MAX_TOTAL)
    max_refinement_suggestions = limits.get("refinement_suggestions", DEFAULT_MAX_REFINEMENT_SUGGESTIONS)

    title_max = RESULT_FIELD_LIMITS.get("title_max_chars", 200)
    snippet_max = RESULT_FIELD_LIMITS.get("snippet_max_chars", 400)

    preserve_fields = get_preserve_fields(intent)
    extract_outcomes = should_extract_clinical_outcomes(intent)

    def _process_result(r: dict, source: str = None) -> dict:
        extra = r.get("extra", {})
        slim = {
            "title": r.get("title", "")[:title_max],
            "id": r.get("id", ""),
            "url": r.get("url", ""),
            "date": r.get("date", extra.get("date", "")),
        }
        if source:
            slim["_source"] = source

        snippet = r.get("snippet", extra.get("abstract", ""))
        if snippet:
            slim["snippet"] = snippet[:snippet_max] + "..." if len(str(snippet)) > snippet_max else snippet

        for field in preserve_fields:
            if field in extra and extra[field]:
                value = extra[field]
                if isinstance(value, str) and len(value) > 200:
                    slim[field] = value[:200] + "..."
                elif isinstance(value, list) and len(value) > 5:
                    slim[field] = value[:5]
                else:
                    slim[field] = value
            elif field in r and r[field]:
                slim[field] = r[field]

        if extract_outcomes and snippet:
            outcomes = extract_clinical_outcomes(snippet)
            if outcomes:
                slim["_clinical_outcomes"] = outcomes

        if "_authority" in r:
            slim["_authority"] = r["_authority"]

        return slim

    if "results_by_source" in res:
        truncated_by_source = {}
        total_count = 0
        for source, results in res.get("results_by_source", {}).items():
            if isinstance(results, list):
                truncated = results[:max_per_source]
                slimmed = []
                for r in truncated:
                    if total_count >= max_total:
                        break
                    slim = _process_result(r, source)
                    slimmed.append(slim)
                    total_count += 1
                truncated_by_source[source] = slimmed

        res["results_by_source"] = truncated_by_source
        res["all_results"] = []
        res["total_hits"] = total_count
        res["_truncated"] = True
        res["_limits_applied"] = {
            "max_per_source": max_per_source,
            "max_total": max_total,
            "intent": intent.value if intent else "default",
            "preserved_fields": preserve_fields[:10],
            "clinical_outcomes_extracted": extract_outcomes,
        }

    elif "results" in res and isinstance(res["results"], list):
        source = res.get("source", "unknown")
        results = res["results"][:max_total]
        slimmed = [_process_result(r, source) for r in results]
        res["results"] = slimmed
        res["total_results"] = len(slimmed)
        res["_truncated"] = True
        res["_limits_applied"] = {
            "max_total": max_total,
            "intent": intent.value if intent else "default",
            "preserved_fields": preserve_fields[:10],
            "clinical_outcomes_extracted": extract_outcomes,
        }

    if "mesh_metadata" in res and isinstance(res["mesh_metadata"], dict):
        mesh = res["mesh_metadata"]
        truncated_mesh = {
            "intent": mesh.get("intent"),
            "expanded_terms_count": mesh.get("expanded_terms_count"),
            "qualifiers": mesh.get("qualifiers", [])[:mesh_limits.get("qualifiers", 10)],
            "major_only": mesh.get("major_only"),
            "pv_terms_count": mesh.get("pv_terms_count"),
        }
        if "mesh_records" in mesh and isinstance(mesh["mesh_records"], list):
            truncated_mesh["mesh_records"] = [
                {"name": r.get("name", "")[:100], "ui": r.get("ui", "")}
                for r in mesh["mesh_records"][:mesh_limits.get("mesh_records", 7)]
            ]
        if "tree_numbers" in mesh:
            truncated_mesh["tree_numbers"] = mesh["tree_numbers"][:mesh_limits.get("tree_numbers", 8)] if isinstance(mesh["tree_numbers"], list) else []
        if "pharmacological_actions" in mesh:
            truncated_mesh["pharmacological_actions"] = mesh["pharmacological_actions"][:mesh_limits.get("pharmacological_actions", 5)] if isinstance(mesh["pharmacological_actions"], list) else []
        if "drug_mapping" in mesh and isinstance(mesh["drug_mapping"], dict):
            dm = mesh["drug_mapping"]
            truncated_mesh["drug_mapping"] = {
                "indications": dm.get("indications", [])[:mesh_limits.get("drug_mapping_indications", 5)],
                "mechanism": dm.get("mechanism", [])[:mesh_limits.get("drug_mapping_mechanism", 4)],
                "mesh_scr": dm.get("mesh_scr"),
            }
        res["mesh_metadata"] = truncated_mesh

    if "refinement_suggestions" in res and isinstance(res["refinement_suggestions"], list):
        res["refinement_suggestions"] = res["refinement_suggestions"][:max_refinement_suggestions]

    if "pass_summary" in res and isinstance(res["pass_summary"], dict):
        res["pass_summary"] = {
            "total_sources": res["pass_summary"].get("total_sources"),
            "successful": res["pass_summary"].get("successful_sources", res["pass_summary"].get("successful")),
        }

    if "intent_context" not in res and intent:
        res["intent_context"] = {
            "detected_intent": intent.value,
            "preserved_fields": preserve_fields[:10],
        }

    return res


def _enforce_output_size_limit(res: dict, max_bytes: int = MAX_TOOL_OUTPUT_BYTES, query: str = None) -> dict:
    """
    Enforce hard output size limit with progressive truncation.

    This is the CRITICAL safeguard against agent silent failures.
    OpenAI's submit_tool_outputs has an undocumented limit (~256KB).

    Strategy:
    1. Check serialized JSON size
    2. If over limit, progressively reduce results count
    3. If still over, strip metadata and reduce field sizes
    4. Log warnings for debugging

    Returns:
        Truncated dict guaranteed to be under max_bytes when serialized
    """
    if not isinstance(res, dict):
        return res

    # First serialization check
    try:
        serialized = json.dumps(res)
        current_size = len(serialized.encode('utf-8'))
    except (TypeError, ValueError) as e:
        _logger.error(f"JSON serialization failed: {e}")
        return {"error": "Serialization failed", "query": query}

    if current_size <= max_bytes:
        # Add size metadata for debugging (only if not already over limit)
        res["_output_size_bytes"] = current_size
        return res

    _logger.warning(f"Output size {current_size} bytes exceeds limit {max_bytes}, applying progressive truncation")

    # Progressive reduction of results
    for threshold, factor in PROGRESSIVE_TRUNCATION_THRESHOLDS:
        if current_size <= threshold:
            break

        # Reduce results in results_by_source
        if "results_by_source" in res:
            for source, results in res.get("results_by_source", {}).items():
                if isinstance(results, list):
                    new_count = max(3, int(len(results) * factor))
                    res["results_by_source"][source] = results[:new_count]

        # Reduce results in flat results array
        if "results" in res and isinstance(res["results"], list):
            new_count = max(5, int(len(res["results"]) * factor))
            res["results"] = res["results"][:new_count]

        # Recalculate size
        serialized = json.dumps(res)
        current_size = len(serialized.encode('utf-8'))

        if current_size <= max_bytes * 0.9:  # Target 90% of max for safety
            break

    # If still over limit, apply aggressive field stripping
    if current_size > max_bytes:
        _logger.warning(f"Still over limit ({current_size} bytes), applying aggressive field stripping")
        res = _aggressive_truncation(res, max_bytes)
        serialized = json.dumps(res)
        current_size = len(serialized.encode('utf-8'))

    # Final safety check - if STILL over, strip to bare minimum
    if current_size > max_bytes:
        _logger.error(f"Output still exceeds limit after truncation ({current_size} > {max_bytes})")
        res = _minimal_response(res, query)
        serialized = json.dumps(res)
        current_size = len(serialized.encode('utf-8'))

    res["_output_size_bytes"] = current_size
    res["_truncated_for_size"] = True
    res["_original_estimated_size"] = "exceeded_limit"

    return res


def _aggressive_truncation(res: dict, max_bytes: int) -> dict:
    """
    Aggressive truncation when progressive reduction isn't enough.

    - Strips mesh_metadata to minimal
    - Shortens all snippets to 200 chars
    - Removes clinical_outcomes
    - Removes refinement_suggestions
    - Keeps only essential fields
    """
    # Strip non-essential top-level keys
    keys_to_remove = ["refinement_suggestions", "pass_summary", "intent_context", "search_metadata"]
    for key in keys_to_remove:
        res.pop(key, None)

    # Minimal mesh_metadata
    if "mesh_metadata" in res:
        mesh = res["mesh_metadata"]
        res["mesh_metadata"] = {
            "intent": mesh.get("intent"),
            "expanded_terms_count": mesh.get("expanded_terms_count"),
        }

    # Shorten all results
    def _slim_result(r: dict) -> dict:
        return {
            "title": str(r.get("title", ""))[:100],
            "id": r.get("id", ""),
            "url": r.get("url", ""),
            "date": r.get("date", ""),
            "_source": r.get("_source", ""),
            "snippet": str(r.get("snippet", ""))[:150] + "..." if r.get("snippet") else "",
        }

    if "results_by_source" in res:
        for source, results in res.get("results_by_source", {}).items():
            if isinstance(results, list):
                res["results_by_source"][source] = [_slim_result(r) for r in results[:10]]

    if "results" in res and isinstance(res["results"], list):
        res["results"] = [_slim_result(r) for r in res["results"][:20]]

    return res


def _minimal_response(res: dict, query: str = None) -> dict:
    """
    Last resort - return minimal response with error indication.
    """
    result_count = 0
    if "results_by_source" in res:
        for results in res.get("results_by_source", {}).values():
            if isinstance(results, list):
                result_count += len(results)
    elif "results" in res:
        result_count = len(res.get("results", []))

    return {
        "error": "Results truncated due to size limits",
        "query": query,
        "total_results_found": result_count,
        "message": "Too many results to return. Please refine your query or use more specific filters.",
        "_truncated_for_size": True,
    }


# ──────────────────────────────────────────────────────────────────────
#  Default (sync) tool router – override from UI/CLI if needed
# ──────────────────────────────────────────────────────────────────────
def _default_tool_router(name: str, args: Dict[str, Any]) -> str:
    """Routes get_med_affairs_data to the project's data layer.

    CRITICAL: Applies size enforcement to prevent agent silent failures.

    v3.3 Pipeline Integration:
    - Large results are compressed with gzip+base64
    - Manifest provides searchable summaries
    - Agent can decompress via pipeline tools
    """
    from med_affairs_data import (
        get_medaffairs_data,
    )  # local import to avoid cycles

    # ──────────────────────────────────────────────────────────────────
    # v3.3: Pipeline decompression tools
    # ──────────────────────────────────────────────────────────────────
    if name == "decompress_pipeline_data":
        try:
            manifest_id = args.get("manifest_id")
            # Try to get from cache first
            cached = get_cached_pipeline_data(manifest_id) if manifest_id else None
            if cached:
                result = decompress_pipeline_data(
                    cached,
                    chunk_index=args.get("chunk_index"),
                    item_indices=args.get("item_indices"),
                )
            else:
                # If not cached, data should be in args
                pipeline_data = args.get("pipeline_data", {})
                result = decompress_pipeline_data(
                    pipeline_data,
                    chunk_index=args.get("chunk_index"),
                    item_indices=args.get("item_indices"),
                )
            return json.dumps(result)
        except Exception as exc:
            _logger.error(f"Pipeline decompression error: {exc}")
            return json.dumps({"error": str(exc)})

    if name == "search_pipeline_manifest":
        try:
            manifest_id = args.get("manifest_id")
            # Try to get manifest from cache
            cached = get_cached_pipeline_data(manifest_id) if manifest_id else None
            manifest = cached.get("manifest", {}) if cached else args.get("manifest", {})
            result = search_manifest(
                manifest,
                search_term=args.get("search_term"),
                source_filter=args.get("source_filter"),
                date_filter=args.get("date_filter"),
                limit=args.get("limit", 20),
            )
            return json.dumps(result)
        except Exception as exc:
            _logger.error(f"Pipeline manifest search error: {exc}")
            return json.dumps({"error": str(exc)})

    # ──────────────────────────────────────────────────────────────────
    # Standard data tools with hybrid pipeline integration (v1.1)
    # ──────────────────────────────────────────────────────────────────
    if name in {
        "get_med_affairs_data",
        "get_pubmed_data",
        "get_fda_data",
        "get_ema_data",
        "get_core_data",
        "get_who_data",
        "tavily_tool",
        "aggregated_search",  # Multi-source search with refinement suggestions
    }:
        try:
            res = get_medaffairs_data(name, args)
            # v3.0: Truncate results with intent-aware context preservation
            query_text = args.get("query", "")
            res = _truncate_search_results(res, query=query_text)

            # v3.3 / v1.1: Hybrid pipeline with automatic strategy selection
            use_pipeline = args.get("use_pipeline", True)  # Enable by default
            serialized_check = json.dumps(res)
            size_bytes = len(serialized_check.encode('utf-8'))

            if use_pipeline and size_bytes > MAX_UNCOMPRESSED_OUTPUT:
                # Select strategy based on size
                strategy = select_pipeline_strategy(size_bytes)
                _logger.info(f"Hybrid pipeline for {name}: {size_bytes} bytes -> {strategy.value}")

                if strategy == PipelineStrategy.VECTOR_STORE:
                    # Use vector store for very large results
                    vector_store_id = args.get("vector_store_id")  # Optional existing VS
                    res = process_with_hybrid_pipeline(
                        res, query=query_text, vector_store_id=vector_store_id
                    )
                    # Cache vector store info for cleanup
                    manifest_id = res.get("manifest", {}).get("manifest_id")
                    if manifest_id and res.get("vector_store"):
                        cache_vector_store_info(manifest_id, res.get("vector_store"))
                else:
                    # Use compression for medium results
                    res = process_through_pipeline(res, query=query_text)

                # Cache for potential decompression requests
                manifest_id = res.get("manifest", {}).get("manifest_id")
                if manifest_id:
                    cache_pipeline_data(manifest_id, res)
            else:
                # v3.2: CRITICAL - Enforce output size limit to prevent agent silent failures
                res = _enforce_output_size_limit(res, query=query_text)

            return json.dumps(res)
        except Exception as exc:
            _logger.error(f"Tool router error for {name}: {exc}")
            return json.dumps({"error": str(exc)})

    return json.dumps({"error": f"Unknown function {name}"})


def _submit_tool_outputs(thread_id: str, run_id: str, outs: List[Dict[str, str]]) -> None:
    openai.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id, run_id=run_id, tool_outputs=outs
    )


def _process_tool_calls(tool_calls, *, thread_id: str, run_id: str, router):
    outs = []
    for call in tool_calls:
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError as exc:
            outs.append({"tool_call_id": call.id, "output": json.dumps({"error": str(exc)})})
            continue
        outs.append({"tool_call_id": call.id, "output": router(call.function.name, args)})
    _submit_tool_outputs(thread_id, run_id, outs)

# ──────────────────────────────────────────────────────────────────────
#  Blocking runner (for Streamlit wrapper)
# ──────────────────────────────────────────────────────────────────────
def run_assistant_sync(
    *,
    thread_id: str,
    assistant_id: str,
    prompt: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
    tool_router: Callable[[str, Dict[str, Any]], str] = _default_tool_router,
    poll: float = 0.5,  # Faster polling for GPT-4.1
    timeout: int = 600,
) -> str:
    """
    Synchronous assistant runner with improved race condition handling.
    GPT-4.1 optimized with faster polling and better message retrieval.
    """
    # Validate thread exists before proceeding
    is_valid, validation_error = validate_thread_exists(thread_id)
    if not is_valid:
        raise RuntimeError(validation_error)

    # CRITICAL: Wait for any active runs to complete before creating new message
    # This prevents the "run already active" 400 error
    wait_for_idle_thread(thread_id, timeout=120)

    # Record start time BEFORE creating message (for accurate message filtering)
    start_timestamp = time.time()

    # Create message with error handling for BadRequestError (400)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            msg = openai.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=prompt, attachments=attachments
            )
            user_message_id = msg.id  # Track the user message ID
            break
        except openai.BadRequestError as e:
            error_str = str(e).lower()
            # If run is active, wait and retry
            if "run" in error_str and "active" in error_str:
                time.sleep(2)
                wait_for_idle_thread(thread_id, timeout=60)
                continue
            # If file-related error and we have attachments, retry without them
            if ("file" in error_str or "attachment" in error_str) and attachments:
                attachments = None
                continue
            # If thread-related error, don't retry
            if "thread" in error_str:
                raise RuntimeError(f"Thread error (400): {str(e)} - please start a new session")
            # For other 400 errors, retry once more then fail
            if attempt == max_retries - 1:
                raise RuntimeError(f"Message creation failed after {max_retries} attempts: {str(e)}")
            time.sleep(0.5 * (attempt + 1))
        except openai.NotFoundError as e:
            raise RuntimeError(f"Thread {thread_id} not found: {str(e)} - please start a new session")
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    # Create run with error handling and retry for active run conflicts
    run = None
    for attempt in range(max_retries):
        try:
            run = openai.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
            break
        except openai.BadRequestError as e:
            error_str = str(e).lower()
            if "run" in error_str and "active" in error_str:
                # Another run is still active - wait for it
                time.sleep(2)
                wait_for_idle_thread(thread_id, timeout=60)
                continue
            raise RuntimeError(f"Failed to create run (400): {str(e)}")
        except openai.NotFoundError as e:
            raise RuntimeError(f"Thread or assistant not found: {str(e)}")

    if run is None:
        raise RuntimeError("Failed to create run after retries")

    # Poll for completion
    start = time.time()
    while True:
        try:
            status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        except openai.BadRequestError as e:
            # Handle transient errors during polling
            if time.time() - start > timeout:
                raise TimeoutError(f"Assistant run exceeded {timeout}s")
            time.sleep(poll)
            continue

        if status.status == "requires_action":
            _process_tool_calls(
                status.required_action.submit_tool_outputs.tool_calls,
                thread_id=thread_id,
                run_id=run.id,
                router=tool_router,
            )
        elif status.status == "completed":
            break
        elif status.status in {"failed", "cancelled", "expired"}:
            error_msg = f"Run ended with status {status.status}"
            if hasattr(status, 'last_error') and status.last_error:
                error_msg += f": {status.last_error.message}"
            raise RuntimeError(error_msg)
        elif time.time() - start > timeout:
            raise TimeoutError(f"Assistant run exceeded {timeout}s")
        time.sleep(poll)

    # IMPROVED: Get messages in descending order (newest first) and find the response
    msgs = openai.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=10)
    for m in msgs.data:
        # Find assistant message that was created after our user message
        if m.role == "assistant":
            # Check if this message's run_id matches our run
            if hasattr(m, 'run_id') and m.run_id == run.id:
                if m.content and len(m.content) > 0:
                    return m.content[0].text.value
            # Fallback: use timestamp comparison (convert created_at to comparable format)
            elif m.created_at >= int(start_timestamp):
                if m.content and len(m.content) > 0:
                    return m.content[0].text.value

    return "[No response generated]"

# ──────────────────────────────────────────────────────────────────────
#  Async runner (CLI / batch)
# ──────────────────────────────────────────────────────────────────────
async def run_assistant_async(
    *,
    thread_id: str,
    assistant_id: str,
    prompt: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
    tool_router_async: Callable[[str, Dict[str, Any]], Coroutine[Any, Any, str]] | None = None,
    poll: float = 1.0,
    timeout: int = 600,
) -> str:
    import asyncio
    tool_router_async = tool_router_async or (
        lambda n, a: asyncio.get_running_loop().run_in_executor(None, _default_tool_router, n, a)
    )

    # Validate thread exists before proceeding
    is_valid, validation_error = await asyncio.to_thread(validate_thread_exists, thread_id)
    if not is_valid:
        raise RuntimeError(validation_error)

    await asyncio.to_thread(wait_for_idle_thread, thread_id)

    # Create message with error handling for BadRequestError (400)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await asyncio.to_thread(
                openai.beta.threads.messages.create,
                thread_id=thread_id, role="user", content=prompt, attachments=attachments,
            )
            break
        except openai.BadRequestError as e:
            error_str = str(e).lower()
            if ("file" in error_str or "attachment" in error_str) and attachments:
                attachments = None
                continue
            if "thread" in error_str:
                raise RuntimeError(f"Thread error (400): {str(e)} - please start a new session")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Message creation failed after {max_retries} attempts: {str(e)}")
            await asyncio.sleep(0.5 * (attempt + 1))
        except openai.NotFoundError as e:
            raise RuntimeError(f"Thread {thread_id} not found: {str(e)} - please start a new session")
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

    # Create run with error handling
    try:
        run = await asyncio.to_thread(
            openai.beta.threads.runs.create, thread_id=thread_id, assistant_id=assistant_id
        )
    except openai.BadRequestError as e:
        raise RuntimeError(f"Failed to create run (400): {str(e)}")
    except openai.NotFoundError as e:
        raise RuntimeError(f"Thread or assistant not found: {str(e)}")

    start = time.time()
    while True:
        status = await asyncio.to_thread(
            openai.beta.threads.runs.retrieve, thread_id=thread_id, run_id=run.id
        )
        if status.status == "requires_action":
            calls = status.required_action.submit_tool_outputs.tool_calls

            async def _one(call):
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as exc:
                    return {"tool_call_id": call.id, "output": json.dumps({"error": str(exc)})}
                out = await tool_router_async(call.function.name, args)
                return {"tool_call_id": call.id, "output": out}

            outs = await asyncio.gather(*[_one(c) for c in calls])
            await asyncio.to_thread(_submit_tool_outputs, thread_id, run.id, outs)

        elif status.status == "completed":
            break
        elif status.status in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"Run ended with status {status.status}")
        elif time.time() - start > timeout:
            raise TimeoutError(f"Assistant run exceeded {timeout}s")
        await asyncio.sleep(poll)

    msgs = await asyncio.to_thread(openai.beta.threads.messages.list, thread_id=thread_id)
    for m in msgs.data:
        if m.role == "assistant" and m.created_at > start:
            return m.content[0].text.value
    return "[No response generated]"
