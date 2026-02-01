# assistant_config.py - GPT-5.2 Responses API Configuration
# ────────────────────────────────────────────────────────────────
# v6.2 – Reasoning & Verbosity Controls
# - Added reasoning.effort: medium (default), high (MC/stats queries)
# - Added text.verbosity: low | medium | high (native API control)
# - Removed dead get_responses_config() / max_output_tokens cap
#
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
    "default_token_budget": 24_000,
    "extended_token_budget": 48_000,
    "maximum_token_budget": 96_000,
    # Responses API: stateless, no server-side history
    "api": "responses",
}

# ────────────────────────────────────────────────────────────────
# GPT-5.2 Reasoning & Verbosity Defaults
# ────────────────────────────────────────────────────────────────
# GPT-5.2 reasoning.effort values: none | low | medium | high | xhigh
# GPT-5.2 text.verbosity values:   low | medium | high
#
# Default reasoning is "none" if omitted, so we pin to "medium".
# MC / Bayesian / statistical analysis queries get "high".
DEFAULT_REASONING_EFFORT = "medium"
HIGH_REASONING_EFFORT = "high"

DEFAULT_VERBOSITY = "medium"

# Keywords that trigger high reasoning effort (MC sims, stats, Bayesian)
HIGH_REASONING_KEYWORDS = [
    "monte carlo", "simulation", "bayesian", "statistical analysis",
    "sensitivity analysis", "probability", "scenario analysis",
    "confidence interval", "hypothesis", "p-value", "regression",
]


def needs_high_reasoning(user_input: str) -> bool:
    """Return True if the query warrants high reasoning effort."""
    q = user_input.lower()
    return any(kw in q for kw in HIGH_REASONING_KEYWORDS)
