"""
workflow_agents.py (v2.1)

Correct Python implementation of your exported JS workflow using the OpenAI Agents SDK.

Key fixes vs v1:
- Uses Runner.run_sync as a @classmethod (per SDK docs)
- Uses correct tool import paths: WebSearchTool / FileSearchTool from `agents`
- Wraps execution in `trace()` so your workflow_id metadata is attached
- Returns visible error text if anything fails, so Streamlit won't "do nothing"

v2.1 changes:
- System instructions loaded from vector store via file_search (system_instructions.txt)
- Agent now retrieves its own instructions from the config vector store at runtime
- Inline instructions reduced to a bootstrap prompt that directs the agent to file_search
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import traceback

from agents import Agent, Runner, WebSearchTool, FileSearchTool, function_tool, trace


WF_ID = "wf_694143ba69ac81908d6378babdaed7f20eeb3fa4d72095a6"
VECTOR_STORE_ID = "vs_693fe785b1a081918f82e9f903e008ed"

# Instructions bootstrap: the full system instructions live in system_instructions.txt
# inside the vector store. The agent uses file_search to retrieve them at runtime.
SAIMONE_V2_INSTRUCTIONS = (
    "You are sAImone v2, a MAPS-compliant Medical Affairs AI assistant.\n\n"
    "IMPORTANT: Your full system instructions and configuration are stored in the "
    "attached vector store as 'system_instructions.txt'. Use file_search to retrieve "
    "and follow those instructions for every interaction.\n\n"
    "Always validate via tools and attached files. Provide structured outputs and footnotes."
)


# Hosted tools (Agents SDK)
web_search = WebSearchTool(search_context_size="high")
file_search = FileSearchTool(vector_store_ids=[VECTOR_STORE_ID])


@function_tool
def getMedAffairsData(
    source: str,
    query: str,
    max_results: int = 10,
    cursor: Optional[int] = None,
    fallback_sources: Optional[List[str]] = None,
    mesh: bool = False,
    date_range: Optional[str] = None,
    fda_decision_type: Optional[str] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stub. Replace with your real backend integration.
    """
    return {
        "status": "stub",
        "message": "getMedAffairsData is not wired. Implement it in workflow_agents.py.",
        "input": {
            "source": source,
            "query": query,
            "max_results": max_results,
            "cursor": cursor,
            "fallback_sources": fallback_sources or [],
            "mesh": mesh,
            "date_range": date_range,
            "fda_decision_type": fda_decision_type,
            "collection": collection,
        },
    }


def build_agent(model: str) -> Agent:
    return Agent(
        name="saimone v2",
        instructions=SAIMONE_V2_INSTRUCTIONS,
        model=model,
        tools=[getMedAffairsData, file_search, web_search],
    )


def run_workflow_sync(text: str, *, model: str = "gpt-4.1") -> str:
    """
    Runs the workflow and returns the final output text.
    Uses GPT-4.1 for consistency across the system.
    If anything fails, returns a visible error string (so Streamlit doesn't appear stuck).
    """
    try:
        agent = build_agent(model=model)
        with trace(workflow_name="Saimonev2", metadata={"__trace_source__": "agent-builder", "workflow_id": WF_ID}):
            result = Runner.run_sync(agent, input=text)

        out = getattr(result, "final_output", None)
        if isinstance(out, str) and out.strip():
            return out.strip()

        # Fallback: show something rather than returning empty
        return f"[WF ran but produced no final_output] result={out!r}"
    except Exception as e:
        tb = traceback.format_exc(limit=8)
        return f"[WF execution error: {type(e).__name__}: {e}]\n{tb}"
