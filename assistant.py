"""
assistant.py
Streamlit wrapper around core_assistant.py.
Exports:
    - run_assistant             - Streamlit UI entry-point
    - handle_file_upload        - file-upload helper
    - validate_file_exists      - tiny utility

v8.0 – Prompt Caching & Agent Resilience:
- Chain reset detection: clears session state last_response_id when the
  core runner signals a broken response chain (_chain_reset in tool_call_log)
- Prompt caching active via STATIC_CONTEXT_BLOCK (from prompt_cache.py)
- Cache metrics forwarded to session state for sidebar monitoring

v5.0 – OpenAI Responses API Migration:
- Replaces thread-based Assistants API with stateless Responses API
- No threads, no runs, no polling - single responses.create() call
- Conversation continuity via previous_response_id (stored in session state)
- Tool calls handled inside core_assistant.run_responses_sync()
- UI feedback via on_tool_call callback
"""

from __future__ import annotations

import json, time, openai, streamlit as st
from typing import Any, Dict, List, Optional, Tuple

from core_assistant import (
    create_context_prompt_with_budget as _make_prompt,
    run_responses_sync as _core_run,
    DEFAULT_MODEL,
)


# ──────────────────────────────────────────────────────────────────────
#  Streamlit tool-call callback (UI spinners / feedback)
# ──────────────────────────────────────────────────────────────────────
def _streamlit_tool_callback(fn_name: str, args: dict) -> None:
    """Called by the core runner for each function-call tool invocation.

    v7.4: Added query_hard_logic callback (silent — instant in-memory lookup).
    v7.0: Web search is handled by OpenAI's built-in web_search_preview
    (server-side, no callback).  Only statistical analysis tools trigger here.
    """
    if fn_name == "query_hard_logic":
        # Silent — in-memory DataFrames are instant, no spinner needed
        return
    if fn_name in {"run_statistical_analysis", "monte_carlo_simulation", "bayesian_analysis"}:
        st.info(f"🔬 Running {fn_name.replace('_', ' ').title()}...")
    else:
        st.info(f"🔧 Calling {fn_name}...")


# ──────────────────────────────────────────────────────────────────────
#  Public UI entry-point
# ──────────────────────────────────────────────────────────────────────
def _handle_post_run_metadata(tool_log: List[dict]) -> None:
    """Process metadata entries appended to tool_call_log by the core runner.

    v8.0: Handles two metadata signals:
    - _chain_reset: If True, the response chain was broken during the run.
      Clears last_response_id in session state so the next turn starts fresh
      instead of trying to continue a dead chain.
    - _cache_metrics: Stores cumulative cache hit metrics in session state
      for sidebar monitoring.
    """
    if not tool_log:
        return

    # Metadata is always the last entry in the log
    last = tool_log[-1]
    if not isinstance(last, dict):
        return

    # ── Chain reset handling ──
    if last.get("_chain_reset"):
        st.session_state["last_response_id"] = None
        _logger_msg = (
            "Response chain was reset during this run — "
            "next turn will start a fresh chain"
        )
        # Use logging instead of st.toast to avoid UI clutter
        import logging
        logging.getLogger(__name__).info(_logger_msg)

    # ── Cache metrics ──
    cache_metrics = last.get("_cache_metrics")
    if cache_metrics:
        st.session_state["_last_cache_metrics"] = cache_metrics


def run_assistant(
    *,
    user_input: str,
    output_type: str,
    response_tone: str,
    compliance_level: str,
    previous_response_id: Optional[str] = None,
    uploaded_file_ids: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
) -> Tuple[str, Optional[str], List[dict]]:
    """Streamlit wrapper for the Responses API runner.

    v8.0: Post-run metadata processing:
    - Detects _chain_reset in tool_call_log and clears session state
    - Stores _cache_metrics for sidebar monitoring

    Responses API context:
    - previous_response_id chains conversation turns
    - No thread_id or assistant_id needed
    - Instructions via STATIC_CONTEXT_BLOCK for prompt caching

    Returns:
        (response_text, response_id, tool_call_log)
    """
    prompt = _make_prompt(
        user_input,
        output_type,
        response_tone,
        compliance_level,
        st.session_state.get("user_role", ""),
        st.session_state.get("user_client", ""),
        st.session_state.get("history", []),
        st.session_state.get("token_budget", 24_000),
        has_files=bool(uploaded_file_ids),
        has_response_chain=bool(previous_response_id),
    )

    with st.spinner("🤖 Thinking..."):
        try:
            text, response_id, tool_log = _core_run(
                model=model,
                input_text=prompt,
                user_query=user_input,     # raw query for profile classification
                previous_response_id=previous_response_id,
                on_tool_call=_streamlit_tool_callback,
            )
            # v8.0: Handle chain reset + cache metrics metadata
            _handle_post_run_metadata(tool_log)
            return text, response_id, tool_log
        except RuntimeError as exc:
            error_msg = str(exc)
            st.error(f"❌ {error_msg}")
            return f"❌ {error_msg}", None, []
        except openai.BadRequestError as exc:
            error_msg = str(exc)
            st.error(f"❌ API Error (400): {error_msg}")
            return f"❌ API Error: {error_msg}", None, []
        except openai.RateLimitError as exc:
            st.error(f"❌ Rate limited: {exc}")
            return f"❌ Rate limited: {exc}", None, []
        except TimeoutError as exc:
            st.error(f"❌ Request timed out: {exc}")
            return f"❌ {exc}", None, []
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            return f"❌ {exc}", None, []


# ──────────────────────────────────────────────────────────────────────
#  Simple run (no UI context) – for quick actions, welcome, etc.
# ──────────────────────────────────────────────────────────────────────
def run_simple(
    message: str,
    *,
    previous_response_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    on_tool_call=None,
) -> Tuple[str, Optional[str], List[dict]]:
    """Run a simple message through the Responses API without full context assembly.

    Used for welcome messages, quick actions, and internal prompts.
    v8.0: Also processes post-run metadata (chain reset, cache metrics).
    """
    try:
        text, response_id, tool_log = _core_run(
            model=model,
            input_text=message,
            user_query=message,            # raw query for profile classification
            previous_response_id=previous_response_id,
            on_tool_call=on_tool_call or _streamlit_tool_callback,
        )
        _handle_post_run_metadata(tool_log)
        return text, response_id, tool_log
    except Exception as exc:
        return f"❌ {exc}", None, []


# ──────────────────────────────────────────────────────────────────────
#  File utilities (unchanged – files API is independent of Responses API)
# ──────────────────────────────────────────────────────────────────────
def handle_file_upload(uploaded_file) -> Dict[str, Any]:
    """Upload *uploaded_file* to OpenAI, wait until processed, return meta."""
    try:
        file_bytes = uploaded_file.getvalue()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > 100:
            return {"success": False, "error": f"File too large: {size_mb:.1f} MB (max 100 MB)"}

        allowed = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "application/json",
            "text/csv",
        }
        if uploaded_file.type not in allowed and not uploaded_file.name.endswith(
            (".pdf", ".docx", ".txt", ".json", ".csv")
        ):
            return {"success": False, "error": f"Unsupported type: {uploaded_file.type}"}

        with st.spinner(f"📤 Uploading {uploaded_file.name}..."):
            fobj = openai.files.create(
                file=(uploaded_file.name, file_bytes), purpose="assistants"
            )

        # Poll until processed (max ~120s with exponential backoff)
        poll_delay = 1.0
        max_poll_time = 120
        poll_start = time.time()
        while time.time() - poll_start < max_poll_time:
            info = openai.files.retrieve(fobj.id)
            if info.status == "processed":
                return {
                    "success": True,
                    "file_id": fobj.id,
                    "filename": uploaded_file.name,
                    "size_mb": size_mb,
                }
            if info.status == "error":
                return {"success": False, "error": "File processing failed"}
            time.sleep(poll_delay)
            poll_delay = min(poll_delay * 1.5, 8)  # backoff: 1→1.5→2.25→...→8s
        return {"success": False, "error": "File processing timeout"}
    except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as exc:
        return {"success": False, "error": f"Upload failed: {type(exc).__name__}"}
    except Exception as exc:
        return {"success": False, "error": f"Upload failed: {type(exc).__name__}"}


def validate_file_exists(file_id: str) -> bool:
    try:
        return openai.files.retrieve(file_id).status in {"uploaded", "processed"}
    except Exception:
        return False
