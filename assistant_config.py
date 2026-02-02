# assistant_config.py - GPT-5.2 Responses API Configuration
# ────────────────────────────────────────────────────────────────
# v6.5 – Simplification
# - Removed dead needs_high_reasoning() / get_reasoning_effort() stubs
# - Capped deep profile tool rounds 20→12 to prevent runaway loops
# - Removed 2-step MC pipeline (handled in main.py removal)
# - All reasoning stays flat at "medium" — GPT-5.2 handles depth
#
# v6.3 – Query Complexity Profiles
# - Three tiers: lightweight / standard / deep
# - Controls timeout, tool rounds, search depth, verbosity, result limits
#
# v6.0 – GPT-5.2 Upgrade
# - 400K context, 128K max output, adaptive reasoning
# - Stateless Responses API, previous_response_id for continuity
# ────────────────────────────────────────────────────────────────

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
    "api": "responses",
}

# ────────────────────────────────────────────────────────────────
# Reasoning & Verbosity Defaults
# ────────────────────────────────────────────────────────────────
# All tiers use "medium" reasoning — GPT-5.2 adaptive reasoning
# handles depth internally. No keyword-based escalation.
DEFAULT_REASONING_EFFORT = "medium"

DEFAULT_VERBOSITY = "medium"


# ────────────────────────────────────────────────────────────────
# Query Complexity Profiles
# ────────────────────────────────────────────────────────────────
# Controls the full pipeline weight per query.
# Three tiers: lightweight / standard / deep
# ────────────────────────────────────────────────────────────────

# Keywords that signal lightweight queries (fast path)
LIGHTWEIGHT_KEYWORDS = [
    "what is", "when was", "who is", "define",
    "approval date", "status of", "price of", "what date",
    "tell me the", "name of", "how many",
    "which company", "what company", "when did",
    "yes or no", "quick question", "brief", "summarize this",
    "summarise this", "recap", "remind me", "what was",
]

# If any of these appear alongside a lightweight keyword, upgrade to standard.
LIGHTWEIGHT_UPGRADE_WORDS = [
    "landscape", "competitive", "strategy", "analysis", "compare",
    "assess", "evaluate", "recommendation", "framework", "tactical",
    "threats", "opportunities", "swot", "stakeholder", "implications",
    "comprehensive", "detailed", "complete", "full", "thorough",
    "all", "every", "across", "portfolio",
]

# Keywords that signal deep queries (full pipeline)
DEEP_KEYWORDS = [
    "comprehensive analysis", "full landscape", "deep dive",
    "multi-step", "end-to-end strategy", "complete competitive",
    "scenario modelling", "portfolio optimization",
    "cascade analysis", "similarity scoring",
    "monte carlo", "simulation", "bayesian", "statistical analysis",
    "sensitivity analysis", "probability", "scenario analysis",
    "confidence interval", "hypothesis", "p-value", "regression",
    "full competitive", "landscape analysis", "market assessment",
    "complete swot", "tows matrix", "war gaming",
    "all assets", "all competitors", "every molecule",
    "complete pipeline", "full pipeline review",
]

QUERY_PROFILES = {
    "lightweight": {
        "reasoning_effort": "medium",
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
        "reasoning_effort": "medium",
        "verbosity": "medium",
        "search_context_size": "high",
        "max_tool_rounds": 12,
        "timeout": 480,
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
    """Classify query into lightweight / standard / deep."""
    q = user_input.lower()

    if any(kw in q for kw in DEEP_KEYWORDS):
        return "deep"

    if any(kw in q for kw in LIGHTWEIGHT_KEYWORDS):
        if any(w in q for w in LIGHTWEIGHT_UPGRADE_WORDS):
            return "standard"
        if len(user_input.strip()) > 150:
            return "standard"
        return "lightweight"

    return "standard"


def get_query_profile(user_input: str) -> dict:
    """Return a full pipeline profile for the query."""
    complexity = classify_query_complexity(user_input)
    profile = dict(QUERY_PROFILES[complexity])
    profile["complexity"] = complexity
    return profile
