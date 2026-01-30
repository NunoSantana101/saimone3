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
    v4.0 – Token-efficient tiered result processing.

    Key changes from v3.x:
    1. DEDUP – cross-source duplicates removed by normalised title
    2. FLAT RANKED LIST – results_by_source flattened; source becomes a field
    3. TIERED CONTENT – top-N get full text, tail gets headline-only
    4. SUMMARY HEADER – one-paragraph overview for agent reasoning head-start
    5. URLS STRIPPED – unless the agent/user explicitly requests citations
    6. MESH / REFINEMENTS – not included inline (lazy-loadable via tools)

    Net effect: ~40-60% fewer tokens, same or better agent reasoning quality.
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

    # Tier thresholds
    t1_count = RESULT_TIERS.get("tier_1_count", 5)
    t1_content = RESULT_TIERS.get("tier_1_content", 600)
    t2_count = RESULT_TIERS.get("tier_2_count", 10)
    t2_content = RESULT_TIERS.get("tier_2_content", 250)
    t3_content = RESULT_TIERS.get("tier_3_content", 0)

    preserve_fields = get_preserve_fields(intent)
    extract_outcomes = should_extract_clinical_outcomes(intent)

    # ── 1. Collect all results into flat list with source tag ────
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

    # ── 2. Deduplicate by normalised title ───────────────────────
    seen_titles: set = set()
    deduped: list = []
    duplicates_removed = 0

    for r, source in raw_items:
        title_raw = r.get("title", "")
        norm = title_raw.lower().strip()
        # Normalise: drop punctuation, collapse whitespace
        norm = "".join(c for c in norm if c.isalnum() or c == " ")
        norm = " ".join(norm.split())

        if norm and norm in seen_titles:
            duplicates_removed += 1
            continue
        if norm:
            seen_titles.add(norm)
        deduped.append((r, source))

    # ── 3. Rank: priority sources first, then by authority/date ──
    priority_sources = limits.get("priority_sources", [])

    def _rank_key(item_tuple):
        r, source = item_tuple
        # Priority sources get lower rank number (sorted ascending)
        try:
            src_rank = priority_sources.index(source)
        except ValueError:
            src_rank = len(priority_sources)
        # Authority score (higher is better → negate for ascending sort)
        authority = -(r.get("_authority", 0) or 0)
        return (src_rank, authority)

    deduped.sort(key=_rank_key)

    # Cap to max_total
    final_items = deduped[:max_total]
    total_available = len(deduped)

    # ── 4. Per-result slimming with tiered content depth ─────────
    def _process_result(r: dict, source: str, rank: int) -> dict:
        extra = r.get("extra", {})

        # Determine content budget based on rank tier
        if rank < t1_count:
            content_budget = t1_content
        elif rank < t1_count + t2_count:
            content_budget = t2_content
        else:
            content_budget = t3_content

        # Extract raw content
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
            content = ""  # Tier-3: no body text

        slim: dict = {}

        # Title (always present)
        title = r.get("title", "")
        if title:
            slim["title"] = title[:title_max]

        # Source tag (replaces results_by_source grouping)
        slim["source"] = source

        # URL (tier-1 and tier-2 only – needed for read_webpage tool)
        result_url = r.get("url") or r.get("extra", {}).get("url")
        if result_url and rank < t1_count + t2_count:
            slim["url"] = result_url

        # Content (tiered)
        if content:
            slim["content"] = content

        # Date
        date = r.get("date") or extra.get("date")
        if date:
            slim["date"] = date

        # Intent-specific fields (only for tier-1 and tier-2)
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

        # Clinical outcome extraction (tier-1 only, safety/trial intents)
        if extract_outcomes and content and rank < t1_count:
            outcomes = extract_clinical_outcomes(content)
            if outcomes:
                slim["outcomes"] = outcomes

        return slim

    processed = [
        _process_result(r, source, rank)
        for rank, (r, source) in enumerate(final_items)
    ]

    # ── 5. Build summary header ──────────────────────────────────
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

    # v4.1: Hint for the agent about deep-read capability
    # Count how many results have URLs (tier-1/tier-2)
    urls_available = sum(1 for r in processed if r.get("url"))
    if urls_available > 0:
        summary["tip"] = (
            "Content above is truncated. If 1-2 results look particularly "
            "relevant, call read_webpage with the url to get the full page text."
        )

    # ── 6. MeSH: drug context one-liner (only for safety/reg) ───
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

    # ── 7. Assemble final output ─────────────────────────────────
    output: dict = {
        "summary": summary,
        "results": processed,
    }

    return output


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

    return res


def _aggressive_truncation(res: dict, max_bytes: int) -> dict:
    """
    Aggressive truncation when progressive reduction isn't enough.
    Content-first: shorten content per result but keep it present.
    v4.0: Works with flat result list only.
    """
    # Strip any leftover non-essential top-level keys
    keep_keys = {"summary", "results", "query", "source", "total_results", "drug_context"}
    for key in list(res.keys()):
        if key not in keep_keys:
            del res[key]

    # Shorten results – keep title + truncated content
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
    """Last resort – return count + message so the agent can request a refined query."""
    result_count = len(res.get("results", []))
    return {
        "error": "Results truncated due to size limits",
        "query": query,
        "total_results_found": result_count,
        "message": "Too many results to return. Please refine your query or use more specific filters.",
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
    # v4.1: Standalone read_webpage tool (Search-Decide-Read pattern)
    # ──────────────────────────────────────────────────────────────────
    if name == "read_webpage":
        try:
            from med_affairs_data import read_webpage
            result = read_webpage(
                url=args.get("url", ""),
                context_query=args.get("context_query"),
            )
            return json.dumps(result)
        except Exception as exc:
            _logger.error(f"read_webpage error: {exc}")
            return json.dumps({"status": "error", "error": str(exc)})

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
