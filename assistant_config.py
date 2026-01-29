# assistant_config.py - GPT-4.1 Thread-based Context Configuration
# ────────────────────────────────────────────────────────────────
# This file contains configuration snippets for GPT-4.1 integration.
# GPT-4.1 features: 1M token context, improved instruction following,
# better function calling, and enhanced reasoning capabilities.
#
# Thread-based context management:
# - OpenAI threads store full conversation history server-side
# - thread_id is the primary context reference
# - Token budgets define how much local context to include in prompts
# ────────────────────────────────────────────────────────────────

# Sidebar Context Management Settings (for main.py integration)
"""
st.sidebar.markdown("### 🧠 Context Management")
context_mode = st.sidebar.selectbox("Context Mode:", [
    "Fast (24k tokens) - GPT-4.1 Standard",
    "Extended (48k tokens) - GPT-4.1 Extended",
    "Maximum (96k tokens) - GPT-4.1 Maximum",
], index=0)

if "24k" in context_mode:
    token_budget = 24000   # GPT-4.1 optimized default - balanced responses
elif "48k" in context_mode:
    token_budget = 48000   # Extended for complex queries
elif "96k" in context_mode:
    token_budget = 96000   # Maximum for deep analysis
else:
    token_budget = 24000   # Default to fast mode

st.session_state["token_budget"] = token_budget

# GPT-4.1 Status Indicator
st.sidebar.markdown("### ⚡ API Status")
st.sidebar.success("🚀 GPT-4.1 Active - Thread-based Context")
st.sidebar.caption("Thread manages full history, local context is supplementary")

if st.session_state["history"]:
    total_exchanges = len(st.session_state["history"])
    estimated_tokens = sum(len(msg["content"]) // 4 for msg in st.session_state["history"])
    st.sidebar.caption(f"📊 Total: {total_exchanges} exchanges (~{estimated_tokens:,} tokens)")
    st.sidebar.caption(f"🎯 Budget: {token_budget:,} tokens per request")

    # Context utilization
    utilization = (estimated_tokens / 1_000_000) * 100
    st.sidebar.caption(f"📈 Context Used: {utilization:.2f}% of 1M")

# Initialize session start time for tracking
if "session_start" not in st.session_state:
    st.session_state["session_start"] = time.time()
"""

# ────────────────────────────────────────────────────────────────
# GPT-4.1 Model Configuration
# ────────────────────────────────────────────────────────────────

GPT41_CONFIG = {
    "model": "gpt-4.1",
    "model_checkpoint": "gpt-4.1",  # Use full model for checkpoints too
    "max_context_tokens": 1_000_000,
    "max_output_tokens": 32_768,
    "default_token_budget": 24_000,   # Balanced responses (increased from 16k)
    "extended_token_budget": 48_000,  # Complex queries (increased from 32k)
    "maximum_token_budget": 96_000,   # Deep analysis (increased from 64k)
    # Thread-based context: OpenAI threads manage full history server-side
    "thread_based_context": True,
}

# ────────────────────────────────────────────────────────────────
# Run Configuration for GPT-4.1
# ────────────────────────────────────────────────────────────────

def get_run_config(token_budget: int = 24000) -> dict:
    """Get optimized run configuration for GPT-4.1 with thread-based context."""
    return {
        "tool_choice": "auto",
        "max_prompt_tokens": min(token_budget, 48_000),  # Increased for richer context
        "max_completion_tokens": 12_000,  # Increased for GPT-4.1 capacity
        "truncation_strategy": {"type": "auto"},
        # Thread-based: OpenAI handles context management via thread_id
    }


# ────────────────────────────────────────────────────────────────
# Adaptive Context Function (GPT-4.1 Optimized)
# ────────────────────────────────────────────────────────────────

def adaptive_medcomms_context(history, token_budget=24000):
    """
    Smart context that adapts to token budget - optimized for GPT-4.1.

    Thread-based Context:
    - OpenAI threads store full conversation history server-side
    - This function provides supplementary local context
    - Prioritizes: Recent + Document analysis + Key decisions

    GPT-4.1 Optimization:
    - Default budget increased to 24k for richer responses
    - Last 8 exchanges included for good continuity (increased from 6)
    - Balances context quality with response speed
    """
    if not history:
        return []

    # Include last 8 exchanges for good continuity (thread has full history)
    must_include = history[-8:] if len(history) >= 8 else history
    remaining_budget = token_budget - estimate_tokens_simple(must_include)

    if len(history) <= 8:
        return must_include

    # Categorization logic continues...
    return must_include  # Placeholder


def estimate_tokens_simple(messages):
    """Estimate tokens in message list (1 token ~ 4 chars)."""
    if not messages:
        return 0
    return sum(len(msg.get("content", "")) // 4 for msg in messages)


# ────────────────────────────────────────────────────────────────
# Context Prompt Builder (GPT-4.1 Optimized)
# ────────────────────────────────────────────────────────────────

def create_context_prompt_with_budget(
    user_input, output_type, response_tone, compliance_level,
    user_role, user_client, history, token_budget
):
    """Create context-aware prompt optimized for GPT-4.1 with thread-based context.

    Thread-based context management:
    - OpenAI threads store full conversation history server-side via thread_id
    - This prompt includes supplementary local context for immediate reference
    - Model: GPT-4.1 throughout
    """
    from datetime import datetime
    import json

    current_date = datetime.now().strftime("%Y-%m-%d")
    date_today = datetime.now().strftime("%B %d, %Y")

    # Use adaptive context management - thread has full history
    context_history = adaptive_medcomms_context(history, token_budget=token_budget)

    # Create summary if we trimmed content
    summary = None
    if len(context_history) < len(history):
        trimmed_count = len(history) - len(context_history)
        summary = f"Prior {trimmed_count} exchanges stored in thread (full history preserved server-side)."

    context = {
        "current_date": current_date,
        "session_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_exchanges": len(history),
            "context_exchanges": len(context_history),
            "context_mode": f"gpt41_thread_based_{token_budget}tokens",
            "token_budget": token_budget,
            "model": "gpt-4.1",
            "max_context": "1M tokens",
            "thread_based": True  # OpenAI thread manages full history
        },
        "user_profile": {
            "role": user_role,
            "client": user_client
        },
        "settings": {
            "output_type": output_type,
            "response_tone": response_tone,
            "compliance_level": compliance_level
        },
        "conversation_history": context_history
    }

    if summary:
        context["conversation_summary"] = summary

    prompt = f"""SYSTEM CONTEXT - Current Date: {current_date}
Today's date: {date_today}

SESSION_CONTEXT:
{json.dumps(context, indent=2)}

INSTRUCTIONS:
- Output Type: {output_type}
- Response Tone: {response_tone}
- Compliance Level: {compliance_level}
- Context: GPT-4.1 Thread-based MedComms ({token_budget:,} tokens)
- Model: GPT-4.1
- Note: Full conversation history is maintained in the OpenAI thread

Focus on the most relevant recent context for high-quality responses.

USER_QUERY: {user_input}"""

    return prompt
