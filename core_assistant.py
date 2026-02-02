"""
core_assistant.py
Pure-python engine shared by the Streamlit UI (assistant.py) and any CLI or
batch runner.  NO Streamlit or console I/O; it just raises exceptions.

v8.0 – Prompt Caching & Agent Resilience:
- Integrated STATIC_CONTEXT_BLOCK from prompt_cache.py as default instructions
  for OpenAI automatic prompt caching (50-90% input token cost reduction)
- Extended 24h cache retention replaces heartbeat daemon
- prompt_cache_key for consistent server routing (~95% cache hit rate)
- cached_tokens monitoring on every API response
- Tool call loop detection (prevents 20x identical calls)
- Response chain recovery with graceful fallback + chain_reset signal
- Empty response retry on first pass
- Exponential backoff on rate limits (4 retries: 2s/4s/8s/16s)
- Async runner error handling (parity with sync runner)
- Pending tool call state recovery on BadRequest mid-loop
- Proactive container age tracking and keep-alive

v7.5.1 – Reasoning & Verbosity Controls:
- Added reasoning.effort to all API calls (default: medium, auto-high for MC/stats)
- Added text.verbosity to all API calls (default: medium)
- Removed dead max_output_tokens cap (was never passed to API)
- Both sync and async runners now accept reasoning_effort and verbosity params

v7.4 – Hard Logic (pandas DataFrames):
- JSON configs loaded into pandas DataFrames at session start via hard_logic.py
- New query_hard_logic function tool for deterministic, instant structured queries
- Hard Logic (DataFrames) stays in memory; Soft Logic (GPT-5.2) handles strategy
- file_search reserved for PDFs and free-text docs only
- All 7 tools: web_search_preview, file_search, code_interpreter,
  run_statistical_analysis, monte_carlo_simulation, bayesian_analysis,
  query_hard_logic

v7.3 – mc_rng.py Code Interpreter Registration:
- Registers mc_rng.py (file-8ofv14JxUdJ2pSTjDxuFHT) as a code interpreter
  file via a sandbox container for direct import in Python execution
- Container is lazily created and cached; falls back to system instructions
  file loading if container API is unavailable

v7.2 – Code Interpreter:
- Activates OpenAI's built-in code_interpreter tool so the model can
  execute Python code in a sandboxed environment for quantitative analysis,
  Monte Carlo simulations, Bayesian inference, and data visualisation

v7.1 – Vector Store file_search:
- Connects vector store vs_693fe785b1a081918f82e9f903e008ed via built-in
  file_search tool so the Responses API can retrieve config JSONs, PDFs,
  and proprietary reference files at runtime

v7.0 – OpenAI Built-in Web Search (test):
- Replaces custom med_affairs_data / Tavily backend with OpenAI's
  built-in web_search_preview tool (search_context_size="high")

v6.0 – GPT-5.2 Upgrade:
- Model upgraded from gpt-4.1 to gpt-5.2
- 400K token context window, 128K max output tokens

v5.0 – OpenAI Responses API Migration:
- Replaces thread-based Assistants API with stateless Responses API
- Single client.responses.create() call per interaction
- Conversation continuity via previous_response_id

Key architecture:
- No threads, no runs, no polling
- response_id replaces thread_id for continuity
- Tool definitions passed inline with each request
- Instructions via STATIC_CONTEXT_BLOCK for prompt caching
- 24h extended cache retention, prompt_cache_key routing
"""

from __future__ import annotations

import atexit, json, threading, time, os, logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import httpx
import openai
from openai import OpenAI

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

from assistant_config import (
    DEFAULT_REASONING_EFFORT,
    HIGH_REASONING_EFFORT,
    LOW_REASONING_EFFORT,
    DEFAULT_VERBOSITY,
    needs_high_reasoning,
    get_reasoning_effort,
)

from hard_logic import (
    get_store as _get_hard_logic_store,
    handle_query_hard_logic as _handle_query_hard_logic,
    QUERY_HARD_LOGIC_TOOL,
)

from prompt_cache import STATIC_CONTEXT_BLOCK, get_cache_stats

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Prompt Cache Configuration
# ──────────────────────────────────────────────────────────────────────
# prompt_cache_key groups requests for consistent server routing,
# maximising cache hit rate (~95% vs ~50% without).
PROMPT_CACHE_KEY = "saimone-v8-static"

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

# Reasoning models that reject temperature / top_p / sampling params.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_UNSUPPORTED_SAMPLING_PARAMS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")


def _strip_sampling_params(request: httpx.Request) -> None:
    """httpx event-hook: strip sampling params the SDK may inject.

    The openai Python SDK can include temperature/top_p in the request
    body even when the caller never passes them (openai-python#2072).
    GPT-5.x and o-series reasoning models reject these with 400.
    This hook inspects every outgoing request and removes the offending
    fields at the HTTP layer — the last line of defence.
    """
    if not request.content:
        return
    try:
        body = json.loads(request.content)
        model = body.get("model", "")
        if not isinstance(model, str) or not model.startswith(_REASONING_MODEL_PREFIXES):
            return
        modified = False
        for param in _UNSUPPORTED_SAMPLING_PARAMS:
            if param in body:
                del body[param]
                modified = True
        if modified:
            raw = json.dumps(body).encode("utf-8")
            request._content = raw
            request.headers["content-length"] = str(len(raw))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        pass


def get_client() -> OpenAI:
    """Return a shared OpenAI client (created lazily).

    The client uses a custom httpx transport with a request hook that
    strips unsupported sampling parameters (temperature, top_p, etc.)
    for reasoning models.  This prevents 400 errors regardless of
    which SDK version is installed.
    """
    global _client
    if _client is None:
        # Prefer explicit key from openai module (set by main.py),
        # then fall back to OPENAI_API_KEY env var.
        api_key = getattr(openai, "api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key found. Set openai.api_key or the "
                "OPENAI_API_KEY environment variable before calling get_client()."
            )
        _client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(
                event_hooks={"request": [_strip_sampling_params]},
            ),
        )
    return _client


def reset_client():
    """Force re-creation of the client (e.g. after API key change)."""
    global _client
    _client = None


# ──────────────────────────────────────────────────────────────────────
#  Default model
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gpt-5.2"

# ──────────────────────────────────────────────────────────────────────
#  Vector Store – OpenAI file_search
# ──────────────────────────────────────────────────────────────────────
VECTOR_STORE_ID = "vs_693fe785b1a081918f82e9f903e008ed"

# ──────────────────────────────────────────────────────────────────────
#  Code Interpreter – file registration (mc_rng.py + JSON configs)
# ──────────────────────────────────────────────────────────────────────
MC_RNG_FILE_ID = "file-P2EgMJJmLDWSJqZnKBoiBJ"

_ci_container_id: Optional[str] = None
_ci_container_attempts: int = 0
_CI_MAX_RETRIES: int = 3  # stop retrying after this many failures
_ci_container_created_at: Optional[float] = None  # v8.0: age tracking for proactive keep-alive
_CI_MAX_AGE_SECONDS: int = 240  # Reset container proactively after 4 min
_container_lock = threading.Lock()  # Protects all _ci_container_* globals

# Cache for JSON files uploaded to OpenAI during this session
# (files without pre-existing file_ids in the registry)
_MAX_UPLOADED_FILE_IDS = 50  # Prevent unbounded growth in long sessions
_uploaded_json_file_ids: Dict[str, str] = {}


def _ensure_json_file_ids() -> Dict[str, str]:
    """Upload local JSON files that lack OpenAI file IDs.

    Returns a mapping of filename → file_id for all JSON configs
    that have valid IDs (either from the Hard Logic registry or
    freshly uploaded during this session).

    Files already registered with a file_id are used directly.
    Local-only files (empty file_id) are uploaded to OpenAI so they
    can be loaded into the code interpreter container.
    """
    from hard_logic import FILE_REGISTRY, LOCAL_DATA_DIR

    all_ids: Dict[str, str] = {}
    client = get_client()

    for key, meta in FILE_REGISTRY.items():
        filename = meta["filename"]

        # Already has a file_id in registry
        existing_id = meta.get("file_id", "")
        if existing_id:
            all_ids[filename] = existing_id
            continue

        # Already uploaded this session
        if filename in _uploaded_json_file_ids:
            all_ids[filename] = _uploaded_json_file_ids[filename]
            continue

        # Try to upload from local file
        local_path = LOCAL_DATA_DIR / filename
        if not local_path.exists():
            _logger.warning("No local file for %s at %s", key, local_path)
            continue

        try:
            with open(local_path, "rb") as f:
                fobj = client.files.create(
                    file=(filename, f),
                    purpose="assistants",
                )
            fid = getattr(fobj, "id", None)
            if not fid:
                _logger.warning("Upload of %s returned empty file ID — skipping", filename)
                continue
            # Evict oldest entry if cache is full
            if len(_uploaded_json_file_ids) >= _MAX_UPLOADED_FILE_IDS:
                oldest_key = next(iter(_uploaded_json_file_ids))
                del _uploaded_json_file_ids[oldest_key]
            _uploaded_json_file_ids[filename] = fid
            all_ids[filename] = fid
            _logger.info("Uploaded %s → %s for CI container", filename, fid)
        except Exception as exc:
            _logger.warning("Could not upload %s to OpenAI: %s", filename, exc)

    return all_ids


def _get_all_container_file_ids() -> List[str]:
    """Collect all file IDs to load into the code interpreter container.

    Includes mc_rng.py plus every JSON config file that has a valid
    OpenAI file ID (pre-existing or freshly uploaded).
    """
    file_ids = [MC_RNG_FILE_ID] if MC_RNG_FILE_ID else []

    try:
        json_ids = _ensure_json_file_ids()
        file_ids.extend(json_ids.values())
    except Exception as exc:
        _logger.warning("Could not resolve JSON file IDs for container: %s", exc)

    # Deduplicate and filter empty/None IDs
    return [fid for fid in set(file_ids) if fid]


def get_container_file_map() -> Dict[str, str]:
    """Return a mapping of OpenAI file_id → friendly filename for all
    files loaded (or to be loaded) into the code interpreter container.

    Used to generate the file-copy initialisation script in system
    instructions so the agent can reference files by name.
    """
    from hard_logic import FILE_REGISTRY

    fmap: Dict[str, str] = {MC_RNG_FILE_ID: "mc_rng.py"}

    for _key, meta in FILE_REGISTRY.items():
        filename = meta["filename"]
        fid = meta.get("file_id", "")
        if fid:
            fmap[fid] = filename
        elif filename in _uploaded_json_file_ids:
            fmap[_uploaded_json_file_ids[filename]] = filename

    return fmap


def _get_code_interpreter_container() -> Optional[str]:
    """Lazy-create a sandbox container with mc_rng.py and all JSON config
    files for the code interpreter.

    Returns the container ID, or None if creation fails (falls back to
    system-instructions-based file loading).  After *_CI_MAX_RETRIES*
    consecutive failures, sets _ci_container_id to the sentinel ``""``
    so get_tools() can cache the (container-less) tools list instead of
    retrying on every request.

    v8.0: Tracks container creation time for proactive age-based reset.
    Thread-safe: all global state mutations protected by _container_lock.
    """
    global _ci_container_id, _ci_container_attempts, _ci_container_created_at
    with _container_lock:
        if _ci_container_id is not None:
            return _ci_container_id or None  # "" sentinel → None
        try:
            client = get_client()
            all_file_ids = _get_all_container_file_ids()
            container = client.containers.create(
                name="saimone-ci",
                file_ids=all_file_ids,
            )
            cid = getattr(container, "id", None)
            if not cid:
                raise ValueError("Container creation returned empty ID")
            _ci_container_id = cid
            _ci_container_created_at = time.time()
            _ci_container_attempts = 0
            _logger.info(
                "Created CI container %s with %d files (mc_rng.py + %d JSON configs)",
                _ci_container_id,
                len(all_file_ids),
                len(all_file_ids) - 1,
            )
            return _ci_container_id
        except Exception as exc:
            _ci_container_attempts += 1
            if _ci_container_attempts >= _CI_MAX_RETRIES:
                _logger.warning(
                    "CI container creation failed %d times; accepting "
                    "fallback permanently: %s",
                    _ci_container_attempts,
                    exc,
                )
                _ci_container_id = ""  # sentinel: stops further retries
            else:
                _logger.warning(
                    "Could not create CI container (attempt %d/%d): %s – "
                    "will retry on next call",
                    _ci_container_attempts,
                    _CI_MAX_RETRIES,
                    exc,
                )
            return None


def _check_container_age() -> None:
    """Proactively reset the container if it's older than _CI_MAX_AGE_SECONDS.

    v8.0: Prevents mid-loop 400 errors from expired containers by resetting
    before the server-side TTL expires.
    Thread-safe: reads under _container_lock then delegates to reset_container.
    """
    global _ci_container_created_at
    with _container_lock:
        if (
            _ci_container_created_at is not None
            and _ci_container_id
            and _ci_container_id != ""
            and (time.time() - _ci_container_created_at) > _CI_MAX_AGE_SECONDS
        ):
            age = int(time.time() - _ci_container_created_at)
            _logger.info(
                "Proactively resetting stale CI container %s (age=%ds, max=%ds)",
                _ci_container_id, age, _CI_MAX_AGE_SECONDS,
            )
            _reset_container_unlocked()
            _ci_container_created_at = None


def _cleanup_container() -> None:
    """Delete the sandbox container on process exit to prevent orphans."""
    if _ci_container_id and _ci_container_id != "":
        try:
            get_client().containers.delete(_ci_container_id)
            _logger.info("Deleted CI container %s on shutdown", _ci_container_id)
        except Exception as exc:
            _logger.debug("Container cleanup failed (non-fatal): %s", exc)


atexit.register(_cleanup_container)


def _reset_container_unlocked() -> None:
    """Inner reset — caller must hold _container_lock."""
    global _ci_container_id, _ci_container_attempts, _cached_tools, _ci_container_created_at
    old = _ci_container_id
    _ci_container_id = None
    _ci_container_attempts = 0
    _ci_container_created_at = None
    _cached_tools = None
    _logger.info("Reset stale CI container (was %s); will recreate on next call", old)


def reset_container() -> None:
    """Reset the cached container ID and tools list.

    Called when a 400 error suggests the server-side container has expired,
    or proactively by _check_container_age() before the TTL expires.
    The next call to get_tools() / build_tools_list() will lazily create a
    fresh container.
    Thread-safe: acquires _container_lock.
    """
    with _container_lock:
        _reset_container_unlocked()


# ──────────────────────────────────────────────────────────────────────
#  Tool Definitions for Responses API
# ──────────────────────────────────────────────────────────────────────

# v7.0: _load_tool_schema_v3 removed – no longer loading custom tool schema
# Web search is handled by OpenAI's built-in web_search_preview tool.


def build_tools_list() -> List[dict]:
    """
    Build the complete tools list for Responses API.

    v7.6: Added search_medical_links (soft integration) for Link-First
    access to PubMed, FDA, and EMA.  Returns compact Markdown links
    (~50 tokens/result) that the model can deep-dive via web_search_preview.

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

    # 3. OpenAI built-in code_interpreter (v7.5)
    #    - Sandboxed Python execution environment
    #    - Used for Monte Carlo simulations, Bayesian inference,
    #      statistical modelling, data visualisation, and similarity scoring
    #    - mc_rng.py + all JSON config files registered via container
    #    - JSON files available at /mnt/data/ for direct import/loading
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

    # 7. query_hard_logic – in-memory pandas DataFrame queries (v7.4)
    #    Provides deterministic, instant access to all JSON config data
    #    (pillars, metrics, tactics, stakeholders, roles, KOLs, etc.)
    #    loaded into pandas DataFrames at session start.
    tools.append(QUERY_HARD_LOGIC_TOOL)

    # 8. search_medical_links – Soft Integration Link-First layer (v7.6)
    #    Returns Markdown-formatted links (~50 tokens/result) from PubMed,
    #    FDA (openFDA → DailyMed/Drugs@FDA), and EMA public datasets.
    #    The model then uses web_search_preview to deep-dive into the
    #    returned links for full content when needed.
    tools.append({
        "type": "function",
        "name": "search_medical_links",
        "description": (
            "Search PubMed, FDA, and/or EMA for medical/regulatory data. "
            "Returns compact Markdown links with minimal metadata (~50 tokens each). "
            "Use this FIRST to gather relevant links, then use web_search_preview "
            "to deep-dive into specific results. "
            "Source can be 'pubmed', 'fda', 'ema', or 'all'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Data source to search.",
                    "enum": ["pubmed", "fda", "ema", "all"],
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Search query — drug name, condition, topic, "
                        "active substance, or natural language query."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum links to return per source. Default 20, max 100.",
                    "default": 20,
                },
                "date_range": {
                    "type": "string",
                    "description": (
                        "Optional date filter, e.g. '2020/01/01-2025/12/31'. "
                        "Applies to PubMed publication dates and FDA effective dates."
                    ),
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order for PubMed results.",
                    "enum": ["relevance", "date"],
                    "default": "relevance",
                },
                "collection": {
                    "type": "string",
                    "description": (
                        "FDA openFDA collection to search. Default: 'drug/label'. "
                        "Other options: 'drug/event', 'drug/enforcement'."
                    ),
                },
            },
            "required": ["source", "query"],
        },
    })

    return tools


# Required built-in tools that must be present in every API request.
# web_search_preview is essential: the agent must always be able to
# validate claims, check regulatory status, and retrieve up-to-date
# medical information during any turn of the session.
_REQUIRED_TOOL_TYPES = frozenset({"web_search_preview", "file_search", "code_interpreter"})


# Cache the tools list.  Only cached once the code-interpreter
# container has been resolved (success OR permanent fallback).
# If the container creation fails transiently, the list is rebuilt
# on the next call so the container can be retried.
_cached_tools: Optional[List[dict]] = None


def get_tools() -> List[dict]:
    """Return cached tools list, rebuilding if container is unresolved.

    Guarantees that web_search_preview, file_search, and code_interpreter
    are always present — raises immediately if the invariant is violated
    so the bug is caught during development, not silently in production.
    """
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools
    tools = build_tools_list()

    # ── Invariant: required built-in tools must always be present ──
    present = {t.get("type") for t in tools}
    missing = _REQUIRED_TOOL_TYPES - present
    if missing:
        raise RuntimeError(
            f"Tool list is missing required built-in tools: {missing}. "
            "web_search_preview must be available for every session turn."
        )

    # Only cache if the container was resolved (present or permanently
    # unavailable after the grace window).  _ci_container_id is set to
    # a string on success; it stays None on transient failure.
    if _ci_container_id is not None:
        _cached_tools = tools
    return tools


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
    has_response_chain: bool = False,
) -> str:
    """Compose the user-facing prompt while honouring *token_budget*.

    When *has_response_chain* is True the OpenAI Responses API already
    holds the full conversation server-side (via previous_response_id).
    In that case we omit the local conversation history from the prompt
    to avoid triple-sending context (prompt text + response chain +
    system instructions).
    """
    now_iso = datetime.utcnow().isoformat()
    now_long = datetime.utcnow().strftime("%B %d, %Y")

    file_note = (
        "\n\nREFERENCE FILES AVAILABLE - use them when relevant."
        if has_files else ""
    )

    # Only serialize local history when there is no active response
    # chain.  When the chain is active the API already has full context.
    if has_response_chain:
        context_section = (
            f"(Conversation history available via response chain — "
            f"{len(history)} exchanges in session)"
        )
    else:
        context = adaptive_medcomms_context(history, token_budget=token_budget)
        context_section = json.dumps(context, indent=2)

    # Inject Hard Logic schema summary if DataFrames are loaded
    hard_logic_section = ""
    try:
        store = _get_hard_logic_store()
        if store.is_loaded and store.available_datasets:
            hard_logic_section = (
                "\n\nHARD LOGIC (in-memory DataFrames — use query_hard_logic tool):\n"
                + store.get_schema_summary()
                + "\n"
            )
    except Exception:
        pass  # graceful degradation — file_search still available

    return (
        f"SYSTEM CONTEXT - {now_iso}\nToday's date: {now_long}\n\n"
        f"SESSION_CONTEXT:\n{context_section}\n\n"
        f"{file_note}"
        f"{hard_logic_section}\n\n"
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

    v7.6: Added search_medical_links for Link-First medical API access
    (PubMed, FDA, EMA) via soft_integration.py.
    v7.4: Added query_hard_logic for in-memory pandas DataFrame access.
    v7.0: Web search is handled by OpenAI's built-in web_search_preview
    tool (server-side, no routing needed).  Statistical analysis
    function tools and hard logic queries are routed here.
    """

    # ── Hard Logic DataFrame queries (v7.4) ──
    if name == "query_hard_logic":
        try:
            return _handle_query_hard_logic(args)
        except Exception as exc:
            _logger.error(f"Hard logic query error: {exc}")
            return json.dumps({"error": str(exc)})

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

    # ── Soft Integration – Link-First medical API search (v7.6) ──
    if name == "search_medical_links":
        try:
            from soft_integration import search_medical_links
            result = search_medical_links(
                source=args.get("source", "all"),
                query=args.get("query", ""),
                max_results=args.get("max_results", 20),
                date_range=args.get("date_range"),
                sort=args.get("sort", "relevance"),
                collection=args.get("collection"),
            )
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:
            _logger.error(f"Soft integration error: {exc}")
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


def _log_cache_metrics(response, label: str = "") -> Dict[str, Any]:
    """Log and return prompt cache hit metrics from an API response.

    v8.0: Tracks cached_tokens from response.usage.input_tokens_details
    to monitor the effectiveness of prompt caching.
    """
    metrics: Dict[str, Any] = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "cache_hit_rate": 0.0,
    }
    if not hasattr(response, "usage") or not response.usage:
        return metrics

    usage = response.usage
    metrics["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
    metrics["output_tokens"] = getattr(usage, "output_tokens", 0) or 0

    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        if hasattr(details, "cached_tokens"):
            metrics["cached_tokens"] = details.cached_tokens or 0
        elif isinstance(details, dict):
            metrics["cached_tokens"] = details.get("cached_tokens", 0) or 0

    if metrics["input_tokens"] > 0:
        metrics["cache_hit_rate"] = round(
            metrics["cached_tokens"] / metrics["input_tokens"] * 100, 1
        )

    _logger.info(
        "Cache metrics [%s]: input=%d, cached=%d (%.1f%% hit), output=%d",
        label or "response",
        metrics["input_tokens"],
        metrics["cached_tokens"],
        metrics["cache_hit_rate"],
        metrics["output_tokens"],
    )
    return metrics


def _detect_tool_loop(tool_call_log: List[dict], current_calls: list) -> bool:
    """Detect if the agent is stuck calling the same tools repeatedly.

    v8.0: Returns True if the last 3 rounds contain identical tool call
    signatures (same function name + same arguments).  This prevents the
    agent from wasting 20 rounds on a loop that will never resolve.
    """
    if len(tool_call_log) < 3:
        return False

    # Build signature for current calls
    try:
        current_sigs = frozenset(
            (c.name, c.arguments) for c in current_calls
        )
    except (AttributeError, TypeError):
        return False

    # Check the last 3 logged rounds
    recent_rounds = set()
    for entry in tool_call_log:
        recent_rounds.add(entry.get("round", -1))

    last_3_rounds = sorted(recent_rounds)[-3:]
    if len(last_3_rounds) < 3:
        return False

    # Get signatures from each of the last 3 rounds
    for rnd in last_3_rounds:
        round_entries = [e for e in tool_call_log if e.get("round") == rnd]
        round_sigs = frozenset(
            (e["name"], json.dumps(e["args"], sort_keys=True))
            for e in round_entries
        )
        # Compare by name+args (arguments from log are dicts, current are raw strings)
        try:
            current_compare = frozenset(
                (c.name, json.dumps(json.loads(c.arguments), sort_keys=True))
                for c in current_calls
            )
        except (json.JSONDecodeError, TypeError):
            return False
        if round_sigs != current_compare:
            return False

    _logger.warning(
        "Tool loop detected: same calls repeated for 3+ rounds: %s",
        [(c.name, c.arguments[:80]) for c in current_calls],
    )
    return True


def _retry_with_backoff(
    fn: Callable,
    kwargs: dict,
    max_retries: int = 4,
    base_delay: float = 2.0,
    label: str = "",
) -> Any:
    """Retry an API call with exponential backoff on RateLimitError.

    v8.0: Replaces single 2s retry with up to 4 attempts (2s/4s/8s/16s).
    Returns the successful response or raises RuntimeError.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        delay = base_delay * (2 ** (attempt - 1))
        _logger.warning(
            "Rate limited [%s] (attempt %d/%d), backing off %.0fs",
            label, attempt, max_retries, delay,
        )
        time.sleep(delay)
        try:
            return fn(**kwargs)
        except openai.RateLimitError as e:
            last_exc = e
            continue
        except Exception as e:
            raise RuntimeError(f"[{label}] Failed after rate-limit backoff: {e}")
    raise RuntimeError(f"[{label}] Rate limited after {max_retries} retries: {last_exc}")


# ──────────────────────────────────────────────────────────────────────
#  Blocking runner – Responses API (replaces run_assistant_sync)
# ──────────────────────────────────────────────────────────────────────
def run_responses_sync(
    *,
    model: str = DEFAULT_MODEL,
    input_messages: Optional[List[dict]] = None,
    input_text: Optional[str] = None,
    instructions: str = STATIC_CONTEXT_BLOCK,
    previous_response_id: Optional[str] = None,
    tool_router: Callable[[str, Dict[str, Any]], str] = _default_tool_router,
    timeout: int = 600,
    max_tool_rounds: int = 20,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> Tuple[str, Optional[str], List[dict]]:
    """
    Synchronous runner using the OpenAI Responses API.

    v8.0 – Prompt Caching & Agent Resilience:
    - instructions defaults to STATIC_CONTEXT_BLOCK (immutable prefix >1024
      tokens) for OpenAI automatic prompt caching with 24h extended retention
    - prompt_cache_key ensures consistent server routing (~95% cache hit rate)
    - cached_tokens tracked on every response for cost monitoring
    - Tool call loop detection breaks after 3 identical rounds
    - Response chain recovery: drops stale previous_response_id on 400,
      signals chain_reset via tool_call_log metadata
    - Empty response retry on first pass
    - Exponential backoff on rate limits (4 retries: 2s/4s/8s/16s)
    - Pending tool call state recovery on BadRequest mid-loop
    - Proactive container age check before each tool round

    Args:
        instructions: System instructions.  Defaults to STATIC_CONTEXT_BLOCK
            (the immutable prefix from prompt_cache.py) to maximise prompt
            cache hits.  MUST remain identical across requests — no dynamic
            content (timestamps, user_ids, session state).
        reasoning_effort: "none"|"low"|"medium"|"high"|"xhigh".
            Defaults to DEFAULT_REASONING_EFFORT ("medium").
            Auto-escalated to "high" for MC/stats queries when not
            explicitly overridden.
        verbosity: "low"|"medium"|"high".  Maps to text.verbosity.
            Defaults to DEFAULT_VERBOSITY ("medium").

    Returns:
        (response_text, response_id, tool_call_log)
        - response_text: The assistant's final text output
        - response_id: The response ID for conversation continuity
        - tool_call_log: List of {name, args, output} dicts for audit.
          Last entry may contain _cache_metrics and/or _chain_reset metadata.
    """
    client = get_client()
    tools = get_tools()
    chain_was_reset = False

    # Build input
    if input_text and not input_messages:
        api_input = input_text
    elif input_messages:
        api_input = input_messages
    else:
        raise ValueError("Either input_text or input_messages must be provided")

    # Resolve reasoning effort — three-tier: low / medium / high
    if reasoning_effort:
        _effort = reasoning_effort
    else:
        query_text = input_text or ""
        if input_messages:
            query_text = " ".join(
                m.get("content", "") for m in input_messages
                if isinstance(m, dict)
            )
        _effort = get_reasoning_effort(query_text)
        if _effort != DEFAULT_REASONING_EFFORT:
            _logger.info("Auto-selected reasoning effort '%s' for query", _effort)

    _verbosity = verbosity or DEFAULT_VERBOSITY

    tool_call_log: List[dict] = []
    cumulative_cache_metrics: Dict[str, Any] = {
        "total_input_tokens": 0,
        "total_cached_tokens": 0,
        "total_output_tokens": 0,
        "api_calls": 0,
    }
    start = time.time()

    # ── Proactive container age check before the API call ──
    _check_container_age()
    tools = get_tools()  # refresh if container was reset

    # ── Build initial request kwargs ──
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": api_input,
        "tools": tools,
        "reasoning": {"effort": _effort},
        "text": {"verbosity": _verbosity},
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "truncation": "auto",          # v9.0: auto-truncate to avoid context_length_exceeded
    }
    if instructions:
        kwargs["instructions"] = instructions
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    # ── Create initial response ──
    try:
        response = client.responses.create(**kwargs)
    except openai.BadRequestError as e:
        error_str = str(e).lower()
        # ── Context overflow: attempt compaction before giving up ──
        if "context_length_exceeded" in error_str or "context window" in error_str:
            _logger.warning(
                "context_length_exceeded on initial call — attempting compaction"
            )
            try:
                from session_manager import compact_context as _compact
                _summary, _ok = _compact(
                    on_tool_call.__self__.session_state.get("history", [])
                    if hasattr(on_tool_call, "__self__") else [],
                    input_text or "",
                )
            except Exception:
                _summary, _ok = "", False
            if _ok and _summary:
                kwargs.pop("previous_response_id", None)
                kwargs["input"] = (
                    f"[Session context summary]\n{_summary}\n\n"
                    f"[Current query]\n{input_text or ''}"
                )
                chain_was_reset = True
                try:
                    response = client.responses.create(**kwargs)
                except openai.BadRequestError:
                    raise RuntimeError(f"API error (400) after compaction: {str(e)}")
            else:
                # Compaction unavailable — fall through to normal retry
                stale_id = kwargs.pop("previous_response_id", None)
                if stale_id:
                    chain_was_reset = True
                try:
                    response = client.responses.create(**kwargs)
                except openai.BadRequestError:
                    raise RuntimeError(f"API error (400): {str(e)}")
        else:
            # 400 can be caused by a stale previous_response_id OR an expired
            # code-interpreter container.  Reset both and retry once without
            # the response chain.
            stale_id = kwargs.pop("previous_response_id", None)
            _logger.warning(
                "BadRequestError on initial call — resetting container "
                "and dropping response chain (%s): %s", stale_id, e,
            )
            reset_container()
            tools = get_tools()
            kwargs["tools"] = tools
            if stale_id:
                chain_was_reset = True
            try:
                response = client.responses.create(**kwargs)
            except openai.BadRequestError:
                raise RuntimeError(f"API error (400): {str(e)}")
    except openai.NotFoundError as e:
        raise RuntimeError(f"Resource not found: {str(e)}")
    except openai.RateLimitError:
        response = _retry_with_backoff(
            client.responses.create, kwargs, label="initial",
        )
    except openai.APIConnectionError as e:
        raise RuntimeError(f"Connection error: {str(e)}")
    except openai.APITimeoutError as e:
        raise RuntimeError(f"API timeout: {str(e)}")
    except openai.APIStatusError as e:
        raise RuntimeError(f"API status error ({e.status_code}): {str(e)}")

    # ── Log cache metrics for initial response ──
    metrics = _log_cache_metrics(response, "initial")
    cumulative_cache_metrics["total_input_tokens"] += metrics["input_tokens"]
    cumulative_cache_metrics["total_cached_tokens"] += metrics["cached_tokens"]
    cumulative_cache_metrics["total_output_tokens"] += metrics["output_tokens"]
    cumulative_cache_metrics["api_calls"] += 1

    # ── Tool call loop – keep going until text response or limits ──
    for round_num in range(max_tool_rounds):
        if time.time() - start > timeout:
            raise TimeoutError(f"Response exceeded {timeout}s timeout")

        tool_calls = _get_tool_calls(response)

        if not tool_calls:
            # No tool calls — check for empty response on first pass
            text = _extract_response_text(response)
            if text == "[No response generated]" and round_num == 0:
                _logger.warning("Empty response on first pass — retrying without response chain")
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("previous_response_id", None)
                try:
                    retry_response = client.responses.create(**retry_kwargs)
                    retry_text = _extract_response_text(retry_response)
                    if retry_text != "[No response generated]":
                        response = retry_response
                        metrics = _log_cache_metrics(response, "empty-retry")
                        cumulative_cache_metrics["total_input_tokens"] += metrics["input_tokens"]
                        cumulative_cache_metrics["total_cached_tokens"] += metrics["cached_tokens"]
                        cumulative_cache_metrics["total_output_tokens"] += metrics["output_tokens"]
                        cumulative_cache_metrics["api_calls"] += 1
                        if previous_response_id:
                            chain_was_reset = True
                except Exception as retry_exc:
                    _logger.warning("Empty response retry failed: %s", retry_exc)
            break

        # ── Tool call loop detection ──
        if _detect_tool_loop(tool_call_log, tool_calls):
            _logger.warning(
                "Breaking tool loop at round %d — same calls repeated 3+ times",
                round_num,
            )
            break

        # ── Proactive container age check each round ──
        _check_container_age()
        tools = get_tools()

        # ── Process each tool call ──
        tool_outputs: List[dict] = []
        for call in tool_calls:
            fn_name = call.name
            call_id = call.call_id
            try:
                args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                _logger.error("Failed to parse args for %s: %s", fn_name, exc)
                # Tell the model its arguments were malformed so it can retry correctly
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({
                        "error": f"Malformed JSON arguments: {exc}",
                        "hint": "Please re-send with valid JSON arguments.",
                    }),
                })
                tool_call_log.append({
                    "name": fn_name,
                    "args": {},
                    "output_size": 0,
                    "round": round_num,
                    "_parse_error": str(exc),
                })
                continue

            # Notify callback if provided (for UI spinners etc.)
            if on_tool_call:
                on_tool_call(fn_name, args)

            # Execute tool
            try:
                output = tool_router(fn_name, args)
            except Exception as exc:
                _logger.error("Tool execution error for %s: %s", fn_name, exc)
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

        # ── Proactive compaction: if cumulative tokens approach the limit,
        #    compact context and restart the chain to avoid 400 errors. ──
        try:
            from session_manager import needs_compaction as _needs_compact
            if _needs_compact(cumulative_cache_metrics["total_input_tokens"]):
                _logger.warning(
                    "Proactive compaction triggered at %d cumulative input tokens "
                    "(round %d)",
                    cumulative_cache_metrics["total_input_tokens"], round_num,
                )
                from session_manager import compact_context as _compact_ctx
                _summary, _ok = _compact_ctx(
                    on_tool_call.__self__.session_state.get("history", [])
                    if hasattr(on_tool_call, "__self__") else [],
                    input_text or "",
                )
                if _ok and _summary:
                    kwargs.pop("previous_response_id", None)
                    kwargs["input"] = (
                        f"[Compacted session context]\n{_summary}\n\n"
                        f"[Current query]\n{input_text or ''}"
                    )
                    chain_was_reset = True
                    response = client.responses.create(**kwargs)
                    metrics = _log_cache_metrics(response, "post-compaction")
                    cumulative_cache_metrics["total_input_tokens"] = metrics["input_tokens"]
                    cumulative_cache_metrics["total_cached_tokens"] = metrics["cached_tokens"]
                    cumulative_cache_metrics["total_output_tokens"] += metrics["output_tokens"]
                    cumulative_cache_metrics["api_calls"] += 1
                    continue  # re-enter the while loop with the fresh response
        except ImportError:
            pass

        # ── Continue conversation with tool outputs ──
        continuation_kwargs: Dict[str, Any] = {
            "model": model,
            "previous_response_id": response.id,
            "input": tool_outputs,
            "tools": tools,
            "instructions": instructions,
            "reasoning": {"effort": _effort},
            "text": {"verbosity": _verbosity},
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "truncation": "auto",          # v9.0: auto-truncate safety net
        }
        try:
            response = client.responses.create(**continuation_kwargs)
        except openai.BadRequestError as e:
            error_str = str(e).lower()
            # ── Context overflow during tool loop ──
            if "context_length_exceeded" in error_str or "context window" in error_str:
                _logger.warning(
                    "context_length_exceeded in tool loop round %d — "
                    "dropping chain and retrying", round_num,
                )
                restart_kwargs = dict(kwargs)
                restart_kwargs.pop("previous_response_id", None)
                restart_kwargs["tools"] = tools
                chain_was_reset = True
                try:
                    response = client.responses.create(**restart_kwargs)
                except Exception as restart_exc:
                    raise RuntimeError(
                        f"Tool loop restart failed after context overflow: {restart_exc}"
                    )
            elif "previous_response_id" in error_str or "response" in error_str:
                # Response chain broken — restart the entire request fresh
                _logger.warning(
                    "BadRequestError (chain broken) during tool loop round %d — "
                    "restarting request from scratch: %s", round_num, e,
                )
                reset_container()
                tools = get_tools()
                restart_kwargs = dict(kwargs)
                restart_kwargs.pop("previous_response_id", None)
                restart_kwargs["tools"] = tools
                chain_was_reset = True
                try:
                    response = client.responses.create(**restart_kwargs)
                except Exception as restart_exc:
                    raise RuntimeError(
                        f"Tool loop restart failed after chain break: {restart_exc}"
                    )
            else:
                # Container may have expired; reset and retry with same tool outputs
                _logger.warning(
                    "BadRequestError (container?) during tool loop round %d — "
                    "resetting container: %s", round_num, e,
                )
                reset_container()
                tools = get_tools()
                continuation_kwargs["tools"] = tools
                try:
                    response = client.responses.create(**continuation_kwargs)
                except openai.BadRequestError:
                    raise RuntimeError(f"Tool output submission failed (400): {str(e)}")
        except openai.RateLimitError:
            response = _retry_with_backoff(
                client.responses.create, continuation_kwargs,
                label=f"tool-round-{round_num}",
            )
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            _logger.warning(
                "Transient error during tool output submission (round %d): %s — retrying",
                round_num, e,
            )
            time.sleep(2)
            try:
                response = client.responses.create(**continuation_kwargs)
            except Exception as retry_exc:
                raise RuntimeError(f"Tool output submission failed after retry: {retry_exc}")

        # ── Log cache metrics for continuation ──
        metrics = _log_cache_metrics(response, f"tool-round-{round_num}")
        cumulative_cache_metrics["total_input_tokens"] += metrics["input_tokens"]
        cumulative_cache_metrics["total_cached_tokens"] += metrics["cached_tokens"]
        cumulative_cache_metrics["total_output_tokens"] += metrics["output_tokens"]
        cumulative_cache_metrics["api_calls"] += 1

    # ── Extract final text ──
    text = _extract_response_text(response)
    response_id = response.id if hasattr(response, "id") else None

    # ── Append metadata to tool_call_log ──
    overall_hit_rate = 0.0
    if cumulative_cache_metrics["total_input_tokens"] > 0:
        overall_hit_rate = round(
            cumulative_cache_metrics["total_cached_tokens"]
            / cumulative_cache_metrics["total_input_tokens"] * 100, 1,
        )
    tool_call_log.append({
        "_cache_metrics": {
            **cumulative_cache_metrics,
            "overall_cache_hit_rate": overall_hit_rate,
        },
        "_chain_reset": chain_was_reset,
    })

    _logger.info(
        "run_responses_sync complete: %d tool rounds, %d API calls, "
        "%.1f%% cache hit rate, chain_reset=%s",
        len([e for e in tool_call_log if "round" in e]),
        cumulative_cache_metrics["api_calls"],
        overall_hit_rate,
        chain_was_reset,
    )

    return text, response_id, tool_call_log


# ──────────────────────────────────────────────────────────────────────
#  Async runner – Responses API (replaces run_assistant_async)
# ──────────────────────────────────────────────────────────────────────
async def run_responses_async(
    *,
    model: str = DEFAULT_MODEL,
    input_messages: Optional[List[dict]] = None,
    input_text: Optional[str] = None,
    instructions: str = STATIC_CONTEXT_BLOCK,
    previous_response_id: Optional[str] = None,
    tool_router_async: Optional[Callable[[str, Dict[str, Any]], Coroutine[Any, Any, str]]] = None,
    timeout: int = 600,
    max_tool_rounds: int = 20,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> Tuple[str, Optional[str], List[dict]]:
    """
    Async runner using the OpenAI Responses API.

    v8.0: Full parity with sync runner:
    - STATIC_CONTEXT_BLOCK as default instructions for prompt caching
    - prompt_cache_key for consistent server routing
    - Error handling (BadRequest, RateLimit, connection, timeout)
    - Response chain recovery with chain_reset signal
    - Tool call loop detection
    - Proactive container age check
    - cached_tokens monitoring

    Returns:
        (response_text, response_id, tool_call_log)
    """
    import asyncio

    tool_router_async = tool_router_async or (
        lambda n, a: asyncio.get_running_loop().run_in_executor(None, _default_tool_router, n, a)
    )

    client = get_client()
    chain_was_reset = False

    # ── Proactive container age check ──
    _check_container_age()
    tools = get_tools()

    if input_text and not input_messages:
        api_input = input_text
    elif input_messages:
        api_input = input_messages
    else:
        raise ValueError("Either input_text or input_messages must be provided")

    # Resolve reasoning effort — three-tier: low / medium / high
    if reasoning_effort:
        _effort = reasoning_effort
    else:
        query_text = input_text or ""
        if input_messages:
            query_text = " ".join(
                m.get("content", "") for m in input_messages
                if isinstance(m, dict)
            )
        _effort = get_reasoning_effort(query_text)
        if _effort != DEFAULT_REASONING_EFFORT:
            _logger.info("Auto-selected reasoning effort '%s' for async query", _effort)

    _verbosity = verbosity or DEFAULT_VERBOSITY

    tool_call_log: List[dict] = []
    cumulative_cache_metrics: Dict[str, Any] = {
        "total_input_tokens": 0,
        "total_cached_tokens": 0,
        "total_output_tokens": 0,
        "api_calls": 0,
    }
    start = time.time()

    # ── Build initial request kwargs ──
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": api_input,
        "tools": tools,
        "reasoning": {"effort": _effort},
        "text": {"verbosity": _verbosity},
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "truncation": "auto",          # v9.0: auto-truncate to avoid context_length_exceeded
    }
    if instructions:
        kwargs["instructions"] = instructions
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    # ── Initial response (run in thread pool since sync client) ──
    try:
        response = await asyncio.to_thread(client.responses.create, **kwargs)
    except openai.BadRequestError as e:
        error_str = str(e).lower()
        if "context_length_exceeded" in error_str or "context window" in error_str:
            _logger.warning("context_length_exceeded on async initial call — dropping chain")
            kwargs.pop("previous_response_id", None)
            chain_was_reset = True
            try:
                response = await asyncio.to_thread(client.responses.create, **kwargs)
            except openai.BadRequestError:
                raise RuntimeError(f"API error (400) after chain drop: {str(e)}")
        else:
            stale_id = kwargs.pop("previous_response_id", None)
            _logger.warning(
                "BadRequestError on async initial call — resetting container "
                "and dropping response chain (%s): %s", stale_id, e,
            )
            reset_container()
            tools = get_tools()
            kwargs["tools"] = tools
            if stale_id:
                chain_was_reset = True
            try:
                response = await asyncio.to_thread(client.responses.create, **kwargs)
            except openai.BadRequestError:
                raise RuntimeError(f"API error (400): {str(e)}")
    except openai.RateLimitError:
        response = await asyncio.to_thread(
            _retry_with_backoff,
            client.responses.create, kwargs, 4, 2.0, "async-initial",
        )
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        raise RuntimeError(f"Connection/timeout error: {str(e)}")
    except openai.APIStatusError as e:
        raise RuntimeError(f"API status error ({e.status_code}): {str(e)}")

    # ── Log cache metrics for initial response ──
    metrics = _log_cache_metrics(response, "async-initial")
    cumulative_cache_metrics["total_input_tokens"] += metrics["input_tokens"]
    cumulative_cache_metrics["total_cached_tokens"] += metrics["cached_tokens"]
    cumulative_cache_metrics["total_output_tokens"] += metrics["output_tokens"]
    cumulative_cache_metrics["api_calls"] += 1

    # ── Tool call loop ──
    for round_num in range(max_tool_rounds):
        if time.time() - start > timeout:
            raise TimeoutError(f"Response exceeded {timeout}s timeout")

        tool_calls = _get_tool_calls(response)

        if not tool_calls:
            # Check for empty response on first pass
            text = _extract_response_text(response)
            if text == "[No response generated]" and round_num == 0:
                _logger.warning("Empty async response on first pass — retrying")
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("previous_response_id", None)
                try:
                    retry_response = await asyncio.to_thread(
                        client.responses.create, **retry_kwargs,
                    )
                    retry_text = _extract_response_text(retry_response)
                    if retry_text != "[No response generated]":
                        response = retry_response
                        if previous_response_id:
                            chain_was_reset = True
                except Exception as retry_exc:
                    _logger.warning("Empty async response retry failed: %s", retry_exc)
            break

        # ── Tool call loop detection ──
        if _detect_tool_loop(tool_call_log, tool_calls):
            _logger.warning(
                "Breaking async tool loop at round %d", round_num,
            )
            break

        # ── Proactive container age check ──
        _check_container_age()
        tools = get_tools()

        async def _one(call):
            fn_name = call.name
            call_id = call.call_id
            try:
                args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                _logger.error("Failed to parse args for async %s: %s", fn_name, exc)
                tool_call_log.append({
                    "name": fn_name,
                    "args": {},
                    "output_size": 0,
                    "round": round_num,
                    "_parse_error": str(exc),
                })
                return {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({
                        "error": f"Malformed JSON arguments: {exc}",
                        "hint": "Please re-send with valid JSON arguments.",
                    }),
                }
            try:
                output = await tool_router_async(fn_name, args)
            except Exception as exc:
                _logger.error("Async tool execution error for %s: %s", fn_name, exc)
                output = json.dumps({"error": str(exc)})
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

        # ── Proactive compaction check (async) ──
        try:
            from session_manager import needs_compaction as _needs_compact
            if _needs_compact(cumulative_cache_metrics["total_input_tokens"]):
                _logger.warning(
                    "Proactive compaction triggered (async) at %d input tokens (round %d)",
                    cumulative_cache_metrics["total_input_tokens"], round_num,
                )
                from session_manager import compact_context as _compact_ctx
                _summary, _ok = _compact_ctx(
                    on_tool_call.__self__.session_state.get("history", [])
                    if hasattr(on_tool_call, "__self__") else [],
                    input_text or "",
                )
                if _ok and _summary:
                    kwargs.pop("previous_response_id", None)
                    kwargs["input"] = (
                        f"[Compacted session context]\n{_summary}\n\n"
                        f"[Current query]\n{input_text or ''}"
                    )
                    chain_was_reset = True
                    response = await asyncio.to_thread(client.responses.create, **kwargs)
                    metrics = _log_cache_metrics(response, "async-post-compaction")
                    cumulative_cache_metrics["total_input_tokens"] = metrics["input_tokens"]
                    cumulative_cache_metrics["total_cached_tokens"] = metrics["cached_tokens"]
                    cumulative_cache_metrics["total_output_tokens"] += metrics["output_tokens"]
                    cumulative_cache_metrics["api_calls"] += 1
                    continue
        except ImportError:
            pass

        continuation_kwargs: Dict[str, Any] = {
            "model": model,
            "previous_response_id": response.id,
            "input": list(tool_outputs),
            "tools": tools,
            "instructions": instructions,
            "reasoning": {"effort": _effort},
            "text": {"verbosity": _verbosity},
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "truncation": "auto",          # v9.0: auto-truncate safety net
        }
        try:
            response = await asyncio.to_thread(
                client.responses.create, **continuation_kwargs,
            )
        except openai.BadRequestError as e:
            error_str = str(e).lower()
            # ── Context overflow during async tool loop ──
            if "context_length_exceeded" in error_str or "context window" in error_str:
                _logger.warning(
                    "context_length_exceeded in async tool loop round %d — "
                    "dropping chain and retrying", round_num,
                )
                restart_kwargs = dict(kwargs)
                restart_kwargs.pop("previous_response_id", None)
                restart_kwargs["tools"] = tools
                chain_was_reset = True
                try:
                    response = await asyncio.to_thread(
                        client.responses.create, **restart_kwargs,
                    )
                except Exception as restart_exc:
                    raise RuntimeError(
                        f"Async tool loop restart failed after context overflow: {restart_exc}"
                    )
            elif "previous_response_id" in error_str or "response" in error_str:
                _logger.warning(
                    "BadRequestError (chain broken) in async tool loop round %d: %s",
                    round_num, e,
                )
                reset_container()
                tools = get_tools()
                restart_kwargs = dict(kwargs)
                restart_kwargs.pop("previous_response_id", None)
                restart_kwargs["tools"] = tools
                chain_was_reset = True
                response = await asyncio.to_thread(
                    client.responses.create, **restart_kwargs,
                )
            else:
                _logger.warning(
                    "BadRequestError (container?) in async tool loop round %d: %s",
                    round_num, e,
                )
                reset_container()
                tools = get_tools()
                continuation_kwargs["tools"] = tools
                try:
                    response = await asyncio.to_thread(
                        client.responses.create, **continuation_kwargs,
                    )
                except openai.BadRequestError:
                    raise RuntimeError(f"Async tool output submission failed (400): {str(e)}")
        except openai.RateLimitError:
            response = await asyncio.to_thread(
                _retry_with_backoff,
                client.responses.create, continuation_kwargs,
                4, 2.0, f"async-tool-round-{round_num}",
            )
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            _logger.warning(
                "Transient error in async tool output submission (round %d): %s",
                round_num, e,
            )
            await asyncio.sleep(2)
            try:
                response = await asyncio.to_thread(
                    client.responses.create, **continuation_kwargs,
                )
            except Exception as retry_exc:
                raise RuntimeError(f"Async tool output retry failed: {retry_exc}")

        # ── Log cache metrics for continuation ──
        metrics = _log_cache_metrics(response, f"async-tool-round-{round_num}")
        cumulative_cache_metrics["total_input_tokens"] += metrics["input_tokens"]
        cumulative_cache_metrics["total_cached_tokens"] += metrics["cached_tokens"]
        cumulative_cache_metrics["total_output_tokens"] += metrics["output_tokens"]
        cumulative_cache_metrics["api_calls"] += 1

    # ── Extract final text ──
    text = _extract_response_text(response)
    response_id = response.id if hasattr(response, "id") else None

    # ── Append metadata to tool_call_log ──
    overall_hit_rate = 0.0
    if cumulative_cache_metrics["total_input_tokens"] > 0:
        overall_hit_rate = round(
            cumulative_cache_metrics["total_cached_tokens"]
            / cumulative_cache_metrics["total_input_tokens"] * 100, 1,
        )
    tool_call_log.append({
        "_cache_metrics": {
            **cumulative_cache_metrics,
            "overall_cache_hit_rate": overall_hit_rate,
        },
        "_chain_reset": chain_was_reset,
    })

    return text, response_id, tool_call_log
