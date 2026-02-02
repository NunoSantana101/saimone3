# assistant_config.py - GPT-5.2 Responses API Configuration
# ────────────────────────────────────────────────────────────────
# v6.3 – Query Complexity Profiles (Lightweight Pipeline)
# - Added QueryProfile: controls reasoning, search depth, tool rounds,
#   timeout, and verbosity from a single query classification
# - get_query_profile() returns a full pipeline config per query
# - Three tiers: lightweight / standard / deep
# - Prevents timeout on simple queries by capping tool rounds & search
#
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
# xhigh — brand-new in GPT-5.2: deepest reasoning for multi-step analytical queries
# high   — MC / Bayesian / statistical analysis
# medium — default (strategy, tactical, multi-phase workflows)
# low    — simple factual lookups (dates, names, definitions)
DEFAULT_REASONING_EFFORT = "medium"
XHIGH_REASONING_EFFORT = "xhigh"
HIGH_REASONING_EFFORT = "high"
LOW_REASONING_EFFORT = "low"

DEFAULT_VERBOSITY = "medium"

# Keywords that trigger xhigh reasoning effort (multi-step complex analysis)
XHIGH_REASONING_KEYWORDS = [
    "comprehensive analysis", "full landscape", "deep dive",
    "multi-step", "end-to-end strategy", "complete competitive",
    "scenario modelling", "portfolio optimization",
    "cascade analysis", "similarity scoring",
]

# Keywords that trigger high reasoning effort (MC sims, stats, Bayesian)
HIGH_REASONING_KEYWORDS = [
    "monte carlo", "simulation", "bayesian", "statistical analysis",
    "sensitivity analysis", "probability", "scenario analysis",
    "confidence interval", "hypothesis", "p-value", "regression",
]

# Keywords that allow low reasoning effort (simple factual lookups)
LOW_REASONING_KEYWORDS = [
    "what is", "when was", "who is", "define", "list",
    "approval date", "status of", "price of", "what date",
    "tell me the", "what are the", "name of",
]


def needs_high_reasoning(user_input: str) -> bool:
    """Return True if the query warrants high reasoning effort."""
    q = user_input.lower()
    return any(kw in q for kw in HIGH_REASONING_KEYWORDS)


def get_reasoning_effort(user_input: str) -> str:
    """Return the appropriate reasoning effort for the query.

    Four-tier approach (GPT-5.2):
      xhigh  — multi-step complex analyses, full landscape, cascade scoring
      high   — MC simulations, Bayesian, statistical analysis
      medium — default (strategy, tactical, multi-phase workflows)
      low    — simple factual lookups (dates, names, definitions)
    """
    q = user_input.lower()
    if any(kw in q for kw in XHIGH_REASONING_KEYWORDS):
        return XHIGH_REASONING_EFFORT
    if any(kw in q for kw in HIGH_REASONING_KEYWORDS):
        return HIGH_REASONING_EFFORT
    if any(kw in q for kw in LOW_REASONING_KEYWORDS):
        return LOW_REASONING_EFFORT
    return DEFAULT_REASONING_EFFORT


# ────────────────────────────────────────────────────────────────
# Query Complexity Profiles (v6.3)
# ────────────────────────────────────────────────────────────────
# Controls the ENTIRE pipeline weight — not just reasoning effort.
# Each profile sets: reasoning, search depth, tool rounds, timeout,
# verbosity, and result limits as a single coherent config.
#
# Three tiers:
#   lightweight — simple lookups, definitions, single-fact questions
#                 Fast path: low search, few tool rounds, short timeout
#   standard    — typical strategy/tactical queries, drug status checks,
#                 regulatory questions. Balanced search and reasoning.
#   deep        — full landscape analysis, multi-step, MC simulations,
#                 cascade analysis, portfolio optimization.
#                 Full pipeline: high search, many rounds, long timeout.
# ────────────────────────────────────────────────────────────────

# Keywords that signal lightweight queries (fast path)
# These should be specific patterns that clearly indicate a single-fact lookup,
# NOT broad prefixes like "what are the" which can start complex questions.
LIGHTWEIGHT_KEYWORDS = [
    "what is", "when was", "who is", "define",
    "approval date", "status of", "price of", "what date",
    "tell me the", "name of", "how many",
    "which company", "what company", "when did",
    "yes or no", "quick question", "brief", "summarize this",
    "summarise this", "recap", "remind me", "what was",
]

# If any of these words appear alongside a lightweight keyword, upgrade to standard.
# Prevents "what is the full competitive landscape..." from being lightweight.
LIGHTWEIGHT_UPGRADE_WORDS = [
    "landscape", "competitive", "strategy", "analysis", "compare",
    "assess", "evaluate", "recommendation", "framework", "tactical",
    "threats", "opportunities", "swot", "stakeholder", "implications",
    "comprehensive", "detailed", "complete", "full", "thorough",
    "all", "every", "across", "portfolio",
]

# Keywords that signal deep queries (full pipeline)
DEEP_KEYWORDS = [
    # Multi-step analysis
    "comprehensive analysis", "full landscape", "deep dive",
    "multi-step", "end-to-end strategy", "complete competitive",
    "scenario modelling", "portfolio optimization",
    "cascade analysis", "similarity scoring",
    # Quantitative
    "monte carlo", "simulation", "bayesian", "statistical analysis",
    "sensitivity analysis", "probability", "scenario analysis",
    "confidence interval", "hypothesis", "p-value", "regression",
    # Broad research
    "full competitive", "landscape analysis", "market assessment",
    "complete swot", "tows matrix", "war gaming",
    "all assets", "all competitors", "every molecule",
    "complete pipeline", "full pipeline review",
]

# Query profiles — each controls the full pipeline
QUERY_PROFILES = {
    "lightweight": {
        "reasoning_effort": "low",
        "verbosity": "low",
        "search_context_size": "low",
        "max_tool_rounds": 4,
        "timeout": 120,
        "max_results_per_source": 5,
        "max_results_total": 10,
        "tier_1_count": 3,
        "tier_1_content": 800,
        "tier_2_count": 5,
        "tier_2_content": 400,
        "tier_3_content": 100,
    },
    "standard": {
        "reasoning_effort": "medium",
        "verbosity": "medium",
        "search_context_size": "medium",
        "max_tool_rounds": 8,
        "timeout": 300,
        "max_results_per_source": 8,
        "max_results_total": 20,
        "tier_1_count": 5,
        "tier_1_content": 1200,
        "tier_2_count": 8,
        "tier_2_content": 600,
        "tier_3_content": 150,
    },
    "deep": {
        "reasoning_effort": "high",
        "verbosity": "medium",
        "search_context_size": "high",
        "max_tool_rounds": 20,
        "timeout": 600,
        "max_results_per_source": 12,
        "max_results_total": 30,
        "tier_1_count": 8,
        "tier_1_content": 1500,
        "tier_2_count": 12,
        "tier_2_content": 800,
        "tier_3_content": 200,
    },
}


def classify_query_complexity(user_input: str) -> str:
    """Classify query into lightweight / standard / deep.

    Checks deep keywords first (highest priority), then lightweight.
    Lightweight is guarded: if the query contains complexity-indicating
    words (landscape, strategy, analysis, etc.) or is long, it upgrades
    to standard.  Anything else defaults to standard.
    """
    q = user_input.lower()

    # Deep triggers — multi-step, quantitative, broad landscape
    if any(kw in q for kw in DEEP_KEYWORDS):
        return "deep"

    # Lightweight triggers — simple factual lookups
    if any(kw in q for kw in LIGHTWEIGHT_KEYWORDS):
        # Guard 1: if complexity-indicating words are present, upgrade
        if any(w in q for w in LIGHTWEIGHT_UPGRADE_WORDS):
            return "standard"
        # Guard 2: long queries are usually not simple lookups
        if len(user_input.strip()) > 150:
            return "standard"
        return "lightweight"

    # Default: standard (balanced)
    return "standard"


def get_query_profile(user_input: str) -> dict:
    """Return a full pipeline profile for the query.

    The profile controls: reasoning effort, search depth, tool rounds,
    timeout, verbosity, and result limits.  Consumed by core_assistant.py
    to configure the entire request pipeline from a single classification.

    Also preserves the existing four-tier reasoning escalation within
    the deep profile (high → xhigh for specific keywords).
    """
    complexity = classify_query_complexity(user_input)
    profile = dict(QUERY_PROFILES[complexity])
    profile["complexity"] = complexity

    # Within deep tier: escalate reasoning for xhigh keywords
    if complexity == "deep":
        q = user_input.lower()
        if any(kw in q for kw in XHIGH_REASONING_KEYWORDS):
            profile["reasoning_effort"] = "xhigh"

    return profile
