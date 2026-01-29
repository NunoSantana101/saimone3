"""
assistant.py
Streamlit wrapper around core_assistant.py.
Exports:
    • run_assistant             – Streamlit UI entry‑point (unchanged signature)
    • _handle_function_call     – used elsewhere in the dashboard
    • handle_file_upload        – file‑upload helper
    • validate_file_exists      – tiny utility

OPTIMIZATION UPDATE (v3.0):
Design Philosophy:
  Backend = "Dumb" transparent data provider (fetch, structure, audit trail)
  Agent = Reasoning engine (interpret, prioritize, synthesize, recommend)

Key Changes:
- Integrated with tool_config.py v3.0 for centralized limit management
- Added adaptive truncation based on query intent
- PRESERVES intent-specific extra fields (phase, status, sponsor, etc.)
- EXTRACTS clinical outcomes from abstracts (PFS, OS, HR, p-values)
- INCLUDES search transparency metadata for audit trail
- Increased default limits for richer agent context:
  - max_per_source: 15 → 25 (default), up to 30 for complex queries
  - max_total: 40 → 80 (default), up to 100 for complex queries
  - refinement_suggestions: 3 → 5 (default), up to 6 for complex queries
  - MeSH metadata limits increased for better drug/indication context

v3.3 Agent Data Pipeline Integration:
- Structured data compression for large results
- Searchable manifests without full decompression
- Chunk-based access for very large datasets
- Pipeline tools: decompress_pipeline_data, search_pipeline_manifest

v1.1 Hybrid Pipeline:
- Automatic strategy selection based on data size:
  - Small (<150KB): Return directly
  - Medium (150-500KB): Compress with manifest
  - Large (>500KB): Upload to vector store for file_search
- Semantic search capability for very large result sets
"""

from __future__ import annotations

import json, time, openai, streamlit as st
from typing import Any, Dict, List, Optional

from core_assistant import (
    create_context_prompt_with_budget as _make_prompt,
    run_assistant_sync as _core_run,
    wait_for_idle_thread,
    validate_thread_exists,
    # v3.2: Use consolidated truncation functions from core_assistant
    _truncate_search_results,
    _enforce_output_size_limit,
)
from med_affairs_data import get_medaffairs_data  # domain data layer

# v3.3: Agent Data Pipeline (v1.1 Hybrid with Vector Store)
from agent_data_pipeline import (
    process_through_pipeline,
    process_with_hybrid_pipeline,
    decompress_pipeline_data,
    search_manifest,
    cache_pipeline_data,
    get_cached_pipeline_data,
    cache_vector_store_info,
    select_pipeline_strategy,
    PipelineStrategy,
    MAX_UNCOMPRESSED_OUTPUT,
    VECTOR_STORE_THRESHOLD,
)

# Import centralized configuration (only what's needed for assistant.py)
from tool_config import QueryIntent


# ──────────────────────────────────────────────────────────────────────
#  Streamlit‑flavoured tool‑call handler (UI spinners, session state)
#  NOTE: _truncate_search_results and _enforce_output_size_limit are now
#  imported from core_assistant.py to avoid code duplication (v3.2)
# ──────────────────────────────────────────────────────────────────────
def _handle_function_call(run_status, thread_id: str, run_id: str) -> None:
    """Processes tool calls with Streamlit feedback banners."""
    calls = run_status.required_action.submit_tool_outputs.tool_calls
    outs: List[Dict[str, str]] = []

    with st.spinner(f"🔧 Processing {len(calls)} function call(s)…"):
        for call in calls:
            fn = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                outs.append({"tool_call_id": call.id, "output": json.dumps({"error": str(exc)})})
                continue

            # ----------------------------------------------------------------
            # v3.3: Pipeline decompression tools
            # ----------------------------------------------------------------
            if fn == "decompress_pipeline_data":
                try:
                    manifest_id = args.get("manifest_id")
                    cached = get_cached_pipeline_data(manifest_id) if manifest_id else None
                    if cached:
                        result = decompress_pipeline_data(
                            cached,
                            chunk_index=args.get("chunk_index"),
                            item_indices=args.get("item_indices"),
                        )
                    else:
                        pipeline_data = args.get("pipeline_data", {})
                        result = decompress_pipeline_data(
                            pipeline_data,
                            chunk_index=args.get("chunk_index"),
                            item_indices=args.get("item_indices"),
                        )
                    st.info("📦 Decompressed pipeline data")
                    outs.append({"tool_call_id": call.id, "output": json.dumps(result)})
                except Exception as exc:
                    outs.append({"tool_call_id": call.id, "output": json.dumps({"error": str(exc)})})
                continue

            if fn == "search_pipeline_manifest":
                try:
                    manifest_id = args.get("manifest_id")
                    cached = get_cached_pipeline_data(manifest_id) if manifest_id else None
                    manifest = cached.get("manifest", {}) if cached else args.get("manifest", {})
                    result = search_manifest(
                        manifest,
                        search_term=args.get("search_term"),
                        source_filter=args.get("source_filter"),
                        date_filter=args.get("date_filter"),
                        limit=args.get("limit", 20),
                    )
                    st.info(f"🔍 Searched manifest: {result.get('total_matches', 0)} matches")
                    outs.append({"tool_call_id": call.id, "output": json.dumps(result)})
                except Exception as exc:
                    outs.append({"tool_call_id": call.id, "output": json.dumps({"error": str(exc)})})
                continue

            # ----------------------------------------------------------------
            # Unified Med Affairs data wrappers (v1.1 hybrid pipeline integration)
            # ----------------------------------------------------------------
            if fn in {
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
                    res = get_medaffairs_data(fn, args)
                    # v3.2: Adaptive truncation + size enforcement to prevent silent failures
                    query_text = args.get("query", "")
                    res = _truncate_search_results(res, query=query_text)

                    # v1.1: Hybrid pipeline with automatic strategy selection
                    use_pipeline = args.get("use_pipeline", True)
                    serialized_check = json.dumps(res)
                    size_bytes = len(serialized_check.encode('utf-8'))

                    if use_pipeline and size_bytes > MAX_UNCOMPRESSED_OUTPUT:
                        # Select strategy based on size
                        strategy = select_pipeline_strategy(size_bytes)

                        if strategy == PipelineStrategy.VECTOR_STORE:
                            st.info(f"📤 Uploading large result ({size_bytes // 1024}KB) to vector store...")
                            vector_store_id = args.get("vector_store_id")
                            res = process_with_hybrid_pipeline(
                                res, query=query_text, vector_store_id=vector_store_id
                            )
                            manifest_id = res.get("manifest", {}).get("manifest_id")
                            vs_info = res.get("vector_store", {})
                            if vs_info.get("vector_store_id"):
                                cache_vector_store_info(manifest_id, vs_info)
                                st.success(f"✅ Data indexed in vector store (use file_search)")
                        else:
                            st.info(f"📦 Compressing large result ({size_bytes // 1024}KB)...")
                            res = process_through_pipeline(res, query=query_text)
                            manifest_id = res.get("manifest", {}).get("manifest_id")
                            if manifest_id:
                                cache_pipeline_data(manifest_id, res)
                                st.success(f"✅ Data compressed with manifest ID: {manifest_id[:8]}...")
                    else:
                        # CRITICAL: Enforce output size limit to prevent agent truncation
                        res = _enforce_output_size_limit(res, query=query_text)

                    outs.append({"tool_call_id": call.id, "output": json.dumps(res)})
                except Exception as exc:
                    outs.append({"tool_call_id": call.id, "output": json.dumps({"error": str(exc)})})

            # ----------------------------------------------------------------
            # NEW: run_statistical_analysis - Monte Carlo & Bayesian
            # ----------------------------------------------------------------
            elif fn == "run_statistical_analysis":
                try:
                    # Import the statistical analysis module
                    from mc_bayesian_backend import handle_statistical_analysis_function_call
                    
                    # Add session context to the analysis
                    args["session_id"] = thread_id
                    args["user_context"] = {
                        "user_name": st.session_state.get("user_name", ""),
                        "user_role": st.session_state.get("user_role", ""),
                        "user_client": st.session_state.get("user_client", ""),
                        "access_level": st.session_state.get("user_access_level", "")
                    }
                    
                    # Show progress for statistical analysis
                    st.info("🔬 Running statistical analysis... This may take a moment.")
                    
                    # Call the statistical analysis function
                    result = handle_statistical_analysis_function_call(args)
                    
                    # Parse result to show user-friendly progress
                    try:
                        result_dict = json.loads(result)
                        if result_dict.get("status") == "success":
                            analysis_type = args.get("analysis_type", "statistical analysis")
                            st.success(f"✅ {analysis_type.title()} completed successfully!")
                        else:
                            st.warning(f"⚠️ Analysis completed with issues: {result_dict.get('error_message', 'Unknown error')}")
                    except:
                        pass  # Continue with original result
                    
                    outs.append({"tool_call_id": call.id, "output": result})
                    
                except Exception as exc:
                    error_msg = f"Statistical analysis failed: {str(exc)}"
                    st.error(f"❌ {error_msg}")
                    outs.append({"tool_call_id": call.id, "output": json.dumps({"error": error_msg})})

            # ----------------------------------------------------------------
            # NEW: monte_carlo_simulation (specific function)
            # ----------------------------------------------------------------
            elif fn == "monte_carlo_simulation":
                try:
                    from mc_bayesian_backend import run_statistical_analysis
                    
                    # Convert monte_carlo_simulation args to statistical_analysis format
                    parameters = {
                        "therapy_area": args.get("therapy_area", "general"),
                        "region": args.get("region", "US"),
                        "lifecycle_phase": args.get("lifecycle_phase", "launch"),
                        "iterations": args.get("iterations", 1000),
                        "scenarios": args.get("scenarios"),
                        "session_id": thread_id
                    }
                    
                    st.info("🎲 Running Monte Carlo simulation...")
                    
                    result = run_statistical_analysis("monte_carlo", parameters)
                    outs.append({"tool_call_id": call.id, "output": json.dumps(result)})
                    
                    if result.get("status") == "success":
                        st.success("✅ Monte Carlo simulation completed!")
                    
                except Exception as exc:
                    error_msg = f"Monte Carlo simulation failed: {str(exc)}"
                    st.error(f"❌ {error_msg}")
                    outs.append({"tool_call_id": call.id, "output": json.dumps({"error": error_msg})})

            # ----------------------------------------------------------------
            # NEW: bayesian_analysis (specific function)
            # ----------------------------------------------------------------
            elif fn == "bayesian_analysis":
                try:
                    from mc_bayesian_backend import run_statistical_analysis
                    
                    # Convert bayesian_analysis args to statistical_analysis format
                    parameters = {
                        "therapy_area": args.get("therapy_area", "general"),
                        "evidence": args.get("evidence", {}),
                        "priors": args.get("priors"),
                        "session_id": thread_id
                    }
                    
                    st.info("📊 Running Bayesian inference analysis...")
                    
                    result = run_statistical_analysis("bayesian_inference", parameters)
                    outs.append({"tool_call_id": call.id, "output": json.dumps(result)})
                    
                    if result.get("status") == "success":
                        st.success("✅ Bayesian analysis completed!")
                    
                except Exception as exc:
                    error_msg = f"Bayesian analysis failed: {str(exc)}"
                    st.error(f"❌ {error_msg}")
                    outs.append({"tool_call_id": call.id, "output": json.dumps({"error": error_msg})})

            else:  # unknown function
                outs.append({"tool_call_id": call.id, "output": json.dumps({"error": f'Unknown function: {fn}'})})

    # Submit all tool outputs
    openai.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id, run_id=run_id, tool_outputs=outs
    )
    st.success("✅ Tool outputs submitted")

# ──────────────────────────────────────────────────────────────────────
#  Public UI entry‑point  (signature preserved)
# ──────────────────────────────────────────────────────────────────────
def run_assistant(
    *,
    user_input: str,
    output_type: str,
    response_tone: str,
    compliance_level: str,
    thread_id: str,
    assistant_id: str,
    uploaded_file_ids: Optional[List[str]] = None,
) -> str:
    """Thin wrapper – all heavy lifting lives in core_assistant.py.

    Thread-based context: OpenAI thread (via thread_id) stores full history.
    Local context is supplementary for checkpointing.
    """
    prompt = _make_prompt(
        user_input,
        output_type,
        response_tone,
        compliance_level,
        st.session_state.get("user_role", ""),
        st.session_state.get("user_client", ""),
        st.session_state.get("history", []),
        st.session_state.get("token_budget", 24_000),  # Increased from 32k default
        has_files=bool(uploaded_file_ids),
    )

    # Build attachments for file_search / code_interpreter
    attachments = None
    if uploaded_file_ids:
        valid = [fid for fid in uploaded_file_ids if validate_file_exists(fid)]
        if valid:
            attachments = [
                {"file_id": fid, "tools": [{"type": "file_search"}, {"type": "code_interpreter"}]}
                for fid in valid
            ]

    with st.spinner("🤖 Thinking…"):
        try:
            return _core_run(
                thread_id=thread_id,
                assistant_id=assistant_id,
                prompt=prompt,
                attachments=attachments,
            )
        except RuntimeError as exc:
            error_msg = str(exc)
            # Check if this is a thread-related error that requires session reset
            if "thread" in error_msg.lower() and ("not found" in error_msg.lower() or "invalid" in error_msg.lower()):
                st.error(f"❌ Session error: {error_msg}")
                st.warning("💡 Please click 'Reset Session' in the sidebar to start fresh.")
                # Clear the invalid thread from registry
                if "user_thread_registry" in st.session_state:
                    user_id = st.session_state.get("user_email", "anonymous")
                    if user_id in st.session_state["user_thread_registry"]:
                        del st.session_state["user_thread_registry"][user_id]
            else:
                st.error(f"❌ {error_msg}")
            return f"❌ {error_msg}"
        except openai.BadRequestError as exc:
            error_msg = str(exc)
            st.error(f"❌ API Error (400): {error_msg}")
            # If attachment-related, suggest without files
            if "file" in error_msg.lower():
                st.info("💡 Tip: Try your question without file attachments.")
            return f"❌ API Error: {error_msg}"
        except openai.NotFoundError as exc:
            st.error(f"❌ Resource not found: {exc}")
            st.warning("💡 Please click 'Reset Session' in the sidebar to start fresh.")
            return f"❌ Resource not found: {exc}"
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            return f"❌ {exc}"

# ──────────────────────────────────────────────────────────────────────
#  File‑utilities (unchanged from existing code) :contentReference[oaicite:1]{index=1}
# ──────────────────────────────────────────────────────────────────────
def handle_file_upload(uploaded_file) -> Dict[str, Any]:
    """Upload *uploaded_file* to OpenAI, wait until processed, return meta."""
    try:
        file_bytes = uploaded_file.getvalue()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > 100:
            return {"success": False, "error": f"File too large: {size_mb:.1f} MB (max 100 MB)"}

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

        with st.spinner(f"📤 Uploading {uploaded_file.name}…"):
            fobj = openai.files.create(
                file=(uploaded_file.name, file_bytes), purpose="assistants"
            )

        # Poll until processed (≤120 s)
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


def verify_assistant_functions(assistant_id: str) -> Dict[str, Any]:
    """Utility for the dashboard’s debug pane."""
    try:
        assistant = openai.beta.assistants.retrieve(assistant_id)
        fn_names = [t.function.name for t in assistant.tools if t.type == "function"]
        return {
            "success": True,
            "assistant_name": assistant.name,
            "function_names": fn_names,
            "has_memory": "retrieve_memory_context" in fn_names,
            "has_data_search": "get_med_affairs_data" in fn_names,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
