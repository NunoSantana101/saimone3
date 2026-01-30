# assistant_config.py - GPT-5.2 Responses API Configuration
# ────────────────────────────────────────────────────────────────
# v6.0 – GPT-5.2 Upgrade
# GPT-5.2 features: 400K token context, 128K max output, adaptive
# reasoning, improved function calling, and enhanced coding/science.
#
# Responses API context management:
# - Conversation continuity via previous_response_id
# - Instructions passed directly to each API call
# - No server-side thread storage (stateless API)
# - Token budgets define how much local context to include in prompts
# ────────────────────────────────────────────────────────────────

# Sidebar Context Management Settings (for main.py integration)
"""
st.sidebar.markdown("### Context Management")
context_mode = st.sidebar.selectbox("Context Mode:", [
    "Fast (24k tokens) - GPT-5.2 Standard",
    "Extended (48k tokens) - GPT-5.2 Extended",
    "Maximum (96k tokens) - GPT-5.2 Maximum",
], index=0)

if "24k" in context_mode:
    token_budget = 24000   # GPT-5.2 optimized default - balanced responses
elif "48k" in context_mode:
    token_budget = 48000   # Extended for complex queries
elif "96k" in context_mode:
    token_budget = 96000   # Maximum for deep analysis
else:
    token_budget = 24000   # Default to fast mode

st.session_state["token_budget"] = token_budget

# GPT-5.2 Status Indicator
st.sidebar.markdown("### API Status")
st.sidebar.success("GPT-5.2 Active - Responses API")
st.sidebar.caption("Conversation continuity via response chaining")

if st.session_state["history"]:
    total_exchanges = len(st.session_state["history"])
    estimated_tokens = sum(len(msg["content"]) // 4 for msg in st.session_state["history"])
    st.sidebar.caption(f"Total: {total_exchanges} exchanges (~{estimated_tokens:,} tokens)")
    st.sidebar.caption(f"Budget: {token_budget:,} tokens per request")

    # Context utilization
    utilization = (estimated_tokens / 400_000) * 100
    st.sidebar.caption(f"Context Used: {utilization:.2f}% of 400K")

# Initialize session start time for tracking
if "session_start" not in st.session_state:
    st.session_state["session_start"] = time.time()
"""

# ────────────────────────────────────────────────────────────────
# GPT-5.2 Model Configuration
# ────────────────────────────────────────────────────────────────

GPT52_CONFIG = {
    "model": "gpt-5.2",
    "model_checkpoint": "gpt-5.2",
    "max_context_tokens": 400_000,
    "max_output_tokens": 128_000,
    "default_token_budget": 24_000,
    "extended_token_budget": 48_000,
    "maximum_token_budget": 96_000,
    # Responses API: stateless, no server-side history
    "api": "responses",
}

# ────────────────────────────────────────────────────────────────
# Responses API Configuration
# ────────────────────────────────────────────────────────────────

def get_responses_config(token_budget: int = 24000) -> dict:
    """Get optimized configuration for GPT-5.2 Responses API calls.

    Responses API is stateless:
    - Instructions passed per-request
    - Conversation continuity via previous_response_id
    - No thread/run management overhead
    """
    return {
        "model": "gpt-5.2",
        "max_output_tokens": 12_000,
        "tool_choice": "auto",
        "token_budget": token_budget,
    }


# NOTE: adaptive_medcomms_context, estimate_tokens_simple, and
# create_context_prompt_with_budget previously lived here as dead
# duplicates.  The authoritative implementations are in
# core_assistant.py (used by assistant.py via import).  The copies
# were removed in v6.1 to eliminate confusion.
