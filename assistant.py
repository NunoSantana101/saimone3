"""
assistant.py
Streamlit wrapper around core_assistant.py.
Exports:
    - run_assistant             - Streamlit UI entry-point
    - run_assistant_pipeline    - 3-stage pipeline UI entry-point (v9.0)
    - handle_file_upload        - file-upload helper
    - validate_file_exists      - tiny utility

v9.0 – 3-Stage Pipeline Integration:
- Added run_assistant_pipeline() for Architect → Researcher → Synthesizer
- run_assistant() auto-routes to pipeline when PIPELINE_MODE="auto"
- Stage progress feedback via Streamlit status indicators
- Pipeline audit trail exposed in tool_call_log

v5.0 – OpenAI Responses API Migration:
- Replaces thread-based Assistants API with stateless Responses API
- No threads, no runs, no polling - single responses.create() call
- Conversation continuity via previous_response_id (stored in session state)
- Tool calls handled inside core_assistant.run_responses_sync()
- UI feedback via on_tool_call callback

Key changes from v4.x:
- Removed _handle_function_call() - tool calls now handled in core loop
- Removed verify_assistant_functions() - no assistant objects
- run_assistant() no longer takes thread_id/assistant_id
- Returns (response_text, response_id, tool_call_log) tuple
"""

from __future__ import annotations

import json, time, openai, streamlit as st
from typing import Any, Dict, List, Optional, Tuple

from core_assistant import (
    create_context_prompt_with_budget as _make_prompt,
    run_responses_sync as _core_run,
    run_pipeline_sync as _pipeline_run,
    should_use_pipeline as _should_pipeline,
    SYSTEM_INSTRUCTIONS,
    STATIC_CONTEXT_BLOCK,
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

    v9.0: Auto-routes complex queries through the 3-stage pipeline
    (Architect → Researcher → Synthesizer) when PIPELINE_MODE is "auto"
    or "always".  Simple queries use the monolithic path.

    Returns:
        (response_text, response_id, tool_call_log)
    """
    # ── Pipeline auto-routing ──
    if _should_pipeline(user_input):
        return run_assistant_pipeline(
            user_input=user_input,
            output_type=output_type,
            response_tone=response_tone,
            compliance_level=compliance_level,
            previous_response_id=previous_response_id,
            model=model,
        )

    # ── Monolithic path (simple queries, greetings, etc.) ──
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

    with st.spinner("Thinking..."):
        try:
            text, response_id, tool_log = _core_run(
                model=model,
                input_text=prompt,
                previous_response_id=previous_response_id,
                on_tool_call=_streamlit_tool_callback,
            )
            return text, response_id, tool_log
        except RuntimeError as exc:
            error_msg = str(exc)
            st.error(f"Error: {error_msg}")
            return f"Error: {error_msg}", None, []
        except openai.BadRequestError as exc:
            error_msg = str(exc)
            st.error(f"API Error (400): {error_msg}")
            return f"API Error: {error_msg}", None, []
        except openai.RateLimitError as exc:
            st.error(f"Rate limited: {exc}")
            return f"Rate limited: {exc}", None, []
        except TimeoutError as exc:
            st.error(f"Request timed out: {exc}")
            return f"Timeout: {exc}", None, []
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            return f"Error: {exc}", None, []


# ──────────────────────────────────────────────────────────────────────
#  Pipeline UI entry-point  (v9.0)
# ──────────────────────────────────────────────────────────────────────

def _streamlit_stage_callback(stage_name: str, stage_output: Any) -> None:
    """Update the Streamlit UI after each pipeline stage completes."""
    labels = {
        "architect": "Stage 1/3: Planning complete",
        "researcher": "Stage 2/3: Research complete",
        "synthesizer": "Stage 3/3: Response ready",
    }
    label = labels.get(stage_name, f"Stage '{stage_name}' complete")
    st.info(label)


def run_assistant_pipeline(
    *,
    user_input: str,
    output_type: str,
    response_tone: str,
    compliance_level: str,
    previous_response_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Tuple[str, Optional[str], List[dict]]:
    """Run the 3-stage MedAffairs pipeline with Streamlit UI feedback.

    Architect → Researcher → Synthesizer

    Returns:
        (response_text, response_id, tool_call_log)
    """
    system_state = {
        "user_role": st.session_state.get("user_role", ""),
        "current_date": time.strftime("%Y-%m-%d"),
    }

    with st.status("Running 3-stage analysis pipeline...", expanded=True) as status:
        try:
            st.write("**Stage 1/3:** Planning research strategy...")
            text, response_id, tool_log = _pipeline_run(
                user_query=user_input,
                model=model,
                system_state=system_state,
                output_type=output_type,
                response_tone=response_tone,
                compliance_level=compliance_level,
                previous_response_id=previous_response_id,
                on_tool_call=_streamlit_tool_callback,
                on_stage_complete=_streamlit_stage_callback,
            )
            status.update(label="Pipeline complete", state="complete")
            return text, response_id, tool_log

        except RuntimeError as exc:
            status.update(label="Pipeline failed", state="error")
            error_msg = str(exc)
            st.error(f"Pipeline error: {error_msg}")
            return f"Pipeline error: {error_msg}", None, []
        except TimeoutError as exc:
            status.update(label="Pipeline timed out", state="error")
            st.error(f"Pipeline timed out: {exc}")
            return f"Timeout: {exc}", None, []
        except Exception as exc:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Unexpected pipeline error: {exc}")
            return f"Error: {exc}", None, []


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
    """
    try:
        return _core_run(
            model=model,
            input_text=message,
            previous_response_id=previous_response_id,
            on_tool_call=on_tool_call or _streamlit_tool_callback,
        )
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

        # Poll until processed (max 120s)
        for _ in range(60):
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
            time.sleep(2)
        return {"success": False, "error": "File processing timeout"}
    except Exception as exc:
        return {"success": False, "error": f"Upload failed: {exc}"}


def validate_file_exists(file_id: str) -> bool:
    try:
        return openai.files.retrieve(file_id).status in {"uploaded", "processed"}
    except Exception:
        return False
