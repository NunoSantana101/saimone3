"""
session_manager.py
Session context management with cost optimization and resilience patterns.

v5.0 – OpenAI Responses API Migration:
- Removed thread-based validation (no threads in Responses API)
- Removed ThreadValidationCache (no longer needed)
- Conversation continuity now via previous_response_id
- Circuit breaker retained for API resilience
- Context building retained (more important now – no server-side history)
- Checkpoint summaries retained for session continuity

v6.0 – GPT-5.2 Upgrade:
- Model upgraded from gpt-4.1 to gpt-5.2
- 400K context window, 128K max output tokens

Key features:
1. Circuit breaker pattern for API failures
2. Context building with deduplication and caching
3. GPT-5.2 checkpoint summaries for session continuity
4. Token budget management
5. Silent instructions optimization
"""

from __future__ import annotations

import time
import json
import hashlib
import openai
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass, field
from enum import Enum


# ────────────────────────────────────────────────────────────────
# Configuration Constants
# ────────────────────────────────────────────────────────────────

try:
    from tool_config import (
        CONTEXT_CONFIG,
        TOKEN_BUDGETS,
        API_LIMITS,
    )
    DEFAULT_TOKEN_BUDGET = CONTEXT_CONFIG.get("default_token_budget", 24_000)
    EXTENDED_TOKEN_BUDGET = CONTEXT_CONFIG.get("extended_token_budget", 48_000)
    MAXIMUM_TOKEN_BUDGET = CONTEXT_CONFIG.get("maximum_token_budget", 96_000)
    MAX_CONTEXT_TOKENS = CONTEXT_CONFIG.get("max_context_tokens", 400_000)
    MAX_HISTORY_FOR_CONTEXT = CONTEXT_CONFIG.get("max_history_for_context", 40)
    CONTEXT_CACHE_TTL = CONTEXT_CONFIG.get("context_cache_ttl", 300)
    CHECKPOINT_MAX_TOKENS = CONTEXT_CONFIG.get("checkpoint_max_tokens", 600)
    COMPACTION_THRESHOLD = CONTEXT_CONFIG.get("compaction_threshold", 300_000)
    # DEPRECATED: compaction_model / compaction_max_tokens no longer used.
    # Compaction is handled by client.responses.compact() in core_assistant.py.
    COMPACTION_MODEL = CONTEXT_CONFIG.get("compaction_model", "gpt-5.2")
    COMPACTION_MAX_TOKENS = CONTEXT_CONFIG.get("compaction_max_tokens", 0)
    _CONFIG_LOADED = True
except ImportError:
    _CONFIG_LOADED = False
    DEFAULT_TOKEN_BUDGET = 24_000
    EXTENDED_TOKEN_BUDGET = 48_000
    MAXIMUM_TOKEN_BUDGET = 96_000
    MAX_CONTEXT_TOKENS = 400_000
    CONTEXT_CACHE_TTL = 300
    MAX_HISTORY_FOR_CONTEXT = 40
    CHECKPOINT_MAX_TOKENS = 600
    COMPACTION_THRESHOLD = 300_000
    COMPACTION_MODEL = "gpt-5.2"       # Deprecated — kept for compat
    COMPACTION_MAX_TOKENS = 0          # Deprecated — unused

# Model Selection
MODEL_GPT52 = "gpt-5.2"
# Checkpoints use gpt-4.1-mini: fast, cheap, supports temperature,
# and adequate for 150-word session summaries.  GPT-5.2 (reasoning model)
# rejects temperature → 400 errors via Chat Completions API.
CHECKPOINT_MODEL = "gpt-4.1-mini"

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_TIMEOUT = 60
CIRCUIT_BREAKER_HALF_OPEN_REQUESTS = 1


# ────────────────────────────────────────────────────────────────
# Circuit Breaker Pattern
# ────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for API resilience.
    Prevents cascading failures by failing fast when services are down.
    """
    name: str
    failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD
    timeout: float = CIRCUIT_BREAKER_TIMEOUT

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0, init=False)
    half_open_requests: int = field(default=0, init=False)

    def can_execute(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                return True
            return False

        if self.half_open_requests < CIRCUIT_BREAKER_HALF_OPEN_REQUESTS:
            self.half_open_requests += 1
            return True
        return False

    def record_success(self):
        """Record successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN and not self.can_execute()


# Global circuit breakers for different services
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name)
    return _circuit_breakers[name]


# ────────────────────────────────────────────────────────────────
# Context Assembly with Deduplication
# ────────────────────────────────────────────────────────────────

def _hash_message(msg: Dict[str, str]) -> str:
    """Create a hash of a message for deduplication.

    Uses SHA-256 instead of MD5 for pharma-audit-grade integrity
    (collision resistance matters when hashes appear in audit trails).
    """
    content = f"{msg.get('role', '')}:{msg.get('content', '')[:500]}"
    return hashlib.sha256(content.encode()).hexdigest()


def deduplicate_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate messages from history."""
    seen = set()
    deduplicated = []

    for msg in history:
        msg_hash = _hash_message(msg)
        if msg_hash not in seen:
            seen.add(msg_hash)
            deduplicated.append(msg)

    return deduplicated


def estimate_tokens(content: Any) -> int:
    """Fast token estimation (1 token ~ 4 chars)."""
    if isinstance(content, list):
        return sum(len(str(item)) // 4 for item in content)
    return len(str(content)) // 4


def adaptive_context_builder(
    history: List[Dict[str, str]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    priority_keywords: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Build context from local history for prompt enrichment.

    Responses API context management:
    - Conversation continuity is via previous_response_id chain
    - This function provides supplementary local context for prompt enrichment
    - Focus on recent exchanges for immediate continuity
    """
    if not history:
        return []

    clean_history = deduplicate_history(history[-MAX_HISTORY_FOR_CONTEXT:])

    if len(clean_history) <= 8:
        return clean_history

    must_include = clean_history[-8:]
    remaining_budget = token_budget - estimate_tokens(must_include)

    if remaining_budget <= 0:
        return must_include

    priority_buckets: Dict[str, List[Dict[str, str]]] = {
        "document_analysis": [],
        "regulatory": [],
        "compliance": [],
        "kol": [],
        "strategic": [],
        "general": [],
    }

    category_keywords = {
        "document_analysis": ["upload", "document", "pdf", "file", "review", "analysis"],
        "regulatory": ["fda", "ema", "regulatory", "approval", "submission", "nda", "bla"],
        "compliance": ["compliance", "gdpr", "clinical trial", "audit", "mlr"],
        "kol": ["kol", "investigator", "opinion leader", "expert", "author"],
        "strategic": ["market", "competitive", "launch", "strategy", "roi", "stakeholder"],
    }

    if priority_keywords:
        category_keywords["strategic"].extend(priority_keywords)

    for msg in clean_history[:-8]:
        content_lower = msg.get("content", "").lower()
        categorized = False

        for category, keywords in category_keywords.items():
            if any(kw in content_lower for kw in keywords):
                priority_buckets[category].append(msg)
                categorized = True
                break

        if not categorized:
            priority_buckets["general"].append(msg)

    chosen: List[Dict[str, str]] = []
    priority_order = [
        "document_analysis", "regulatory", "compliance",
        "kol", "strategic", "general"
    ]

    for bucket in priority_order:
        for msg in reversed(priority_buckets[bucket]):
            cost = estimate_tokens([msg])
            if remaining_budget - cost < 100:
                break
            chosen.append(msg)
            remaining_budget -= cost

    chosen.sort(key=lambda x: clean_history.index(x))
    chosen.extend(must_include)

    return chosen


# ────────────────────────────────────────────────────────────────
# Cost-Optimized Checkpointing
# ────────────────────────────────────────────────────────────────

def create_checkpoint_summary(
    history_slice: List[Dict[str, str]],
    user_email: str = "unknown"
) -> Dict[str, Any]:
    """
    Create a checkpoint summary using GPT-5.2.

    Checkpoints are useful for:
    - Session resumption when previous_response_id chain is broken
    - Cross-session context (previous_response_id only chains within a session)
    - Local backup of conversation state
    """
    cb = get_circuit_breaker("openai_chat")
    if not cb.can_execute():
        return {
            "turn": len(history_slice),
            "summary": "Checkpoint skipped - service temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
            "user": user_email,
            "error": "circuit_breaker_open"
        }

    compact_history = []
    for msg in history_slice[-16:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:250] + "...[truncated]..." + content[-200:]
        compact_history.append(f"{role}: {content}")

    convo_text = "\n".join(compact_history)

    prompt = f"""Summarize this medical affairs session concisely (max 150 words):
- Key decisions/insights
- Regulatory/compliance points
- Action items

Conversation:
{convo_text}

Summary:"""

    try:
        response = openai.chat.completions.create(
            model=CHECKPOINT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=CHECKPOINT_MAX_TOKENS,
            temperature=0.0,
        )
        summary = response.choices[0].message.content.strip()
        cb.record_success()

        return {
            "turn": len(history_slice),
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "user": user_email,
            "model": CHECKPOINT_MODEL,
            "tokens_used": response.usage.total_tokens if response.usage else 0
        }
    except openai.RateLimitError as e:
        cb.record_failure()
        return {
            "turn": len(history_slice),
            "summary": f"Checkpoint delayed - rate limit: {str(e)[:100]}",
            "timestamp": datetime.now().isoformat(),
            "user": user_email,
            "error": "rate_limit"
        }
    except Exception as e:
        cb.record_failure()
        return {
            "turn": len(history_slice),
            "summary": f"Summary failed: {str(e)[:100]}. Session covers {len(history_slice)} medical affairs exchanges.",
            "timestamp": datetime.now().isoformat(),
            "user": user_email,
            "error": str(type(e).__name__)
        }


# ────────────────────────────────────────────────────────────────
# Lazy Context Assembly
# ────────────────────────────────────────────────────────────────

class LazyContextManager:
    """
    Manages context assembly with lazy evaluation and caching.
    Local context is supplementary for prompt enrichment.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def _compute_history_hash(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "empty"
        last_content = history[-1].get("content", "")[:100] if history else ""
        return f"{len(history)}:{hashlib.sha256(last_content.encode()).hexdigest()}"

    def get_context(
        self,
        history: List[Dict[str, str]],
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        force_refresh: bool = False
    ) -> List[Dict[str, str]]:
        history_hash = self._compute_history_hash(history)
        cache_key = f"{history_hash}:{token_budget}"

        if not force_refresh and cache_key in self._cache:
            cached_context, timestamp = self._cache[cache_key]
            if time.time() - timestamp < CONTEXT_CACHE_TTL:
                return cached_context

        context = adaptive_context_builder(history, token_budget)
        self._cache[cache_key] = (context, time.time())

        self._cleanup_cache()

        return context

    def _cleanup_cache(self):
        now = time.time()
        expired = [
            key for key, (_, timestamp) in self._cache.items()
            if now - timestamp > CONTEXT_CACHE_TTL
        ]
        for key in expired:
            del self._cache[key]

    def invalidate(self):
        self._cache.clear()


# Global context manager instance
_context_manager = LazyContextManager()


def get_optimized_context(
    history: List[Dict[str, str]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    force_refresh: bool = False
) -> List[Dict[str, str]]:
    """Public interface for optimized context retrieval."""
    return _context_manager.get_context(history, token_budget, force_refresh)


# ────────────────────────────────────────────────────────────────
# Context Configuration
# ────────────────────────────────────────────────────────────────

def get_context_config() -> Dict[str, Any]:
    """Get configuration for context management."""
    return {
        "model": MODEL_GPT52,
        "default_token_budget": DEFAULT_TOKEN_BUDGET,
        "extended_token_budget": EXTENDED_TOKEN_BUDGET,
        "maximum_token_budget": MAXIMUM_TOKEN_BUDGET,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "checkpoint_model": CHECKPOINT_MODEL,
        "checkpoint_max_tokens": CHECKPOINT_MAX_TOKENS,
        "context_cache_ttl": CONTEXT_CACHE_TTL,
        "max_history_for_local": MAX_HISTORY_FOR_CONTEXT,
        "api": "responses",
    }


def get_token_budget_for_mode(mode: str) -> int:
    """Get token budget based on context mode selection."""
    mode_lower = mode.lower()
    if "fast" in mode_lower or "24k" in mode_lower:
        return DEFAULT_TOKEN_BUDGET
    elif "extended" in mode_lower or "48k" in mode_lower:
        return EXTENDED_TOKEN_BUDGET
    elif "maximum" in mode_lower or "96k" in mode_lower:
        return MAXIMUM_TOKEN_BUDGET
    return DEFAULT_TOKEN_BUDGET


# ────────────────────────────────────────────────────────────────
# Session State Utilities
# ────────────────────────────────────────────────────────────────

def get_session_metrics(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Get session metrics for monitoring and debugging."""
    if not history:
        return {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "estimated_tokens": 0,
            "unique_messages": 0,
            "duplicate_rate": 0.0
        }

    deduplicated = deduplicate_history(history)

    return {
        "total_messages": len(history),
        "user_messages": sum(1 for m in history if m.get("role") == "user"),
        "assistant_messages": sum(1 for m in history if m.get("role") == "assistant"),
        "estimated_tokens": estimate_tokens(history),
        "unique_messages": len(deduplicated),
        "duplicate_rate": 1 - (len(deduplicated) / len(history)) if history else 0.0
    }


def should_checkpoint(
    history_length: int,
    last_checkpoint_turn: int,
    checkpoint_freq: int = 8
) -> bool:
    """Determine if a checkpoint should be created."""
    if history_length <= 1:
        return False
    return (history_length - last_checkpoint_turn) >= checkpoint_freq


# ────────────────────────────────────────────────────────────────
# Health Check Functions
# ────────────────────────────────────────────────────────────────

def get_service_health() -> Dict[str, Any]:
    """Get health status of all circuit breakers."""
    return {
        name: {
            "state": cb.state.value,
            "failure_count": cb.failure_count,
            "last_failure": cb.last_failure_time,
            "can_execute": cb.can_execute()
        }
        for name, cb in _circuit_breakers.items()
    }


def reset_circuit_breakers():
    """Reset all circuit breakers to closed state."""
    for cb in _circuit_breakers.values():
        cb.state = CircuitState.CLOSED
        cb.failure_count = 0
        cb.half_open_requests = 0


# ────────────────────────────────────────────────────────────────
# Silent Instructions Optimization
# ────────────────────────────────────────────────────────────────

COMPACT_SILENT_INSTRUCTIONS = """=== RESPONSE GUIDELINES ===
ACCURACY: Evidence-based only. Mark INFERENCES explicitly. Validate via live search. Flag uncertainty.
SOURCES: Prioritize FDA/EMA/PubMed. Cross-reference claims. Cite with hyperlinks.
COMPLIANCE: Include regulatory context. Distinguish approved vs investigational.
DATA: Use query_hard_logic tool for structured config data (pillars, metrics, tactics, stakeholders, roles, KOLs). Use file_search only for PDFs/free-text docs.
Available tools: web_search_preview, file_search, code_interpreter, run_statistical_analysis, monte_carlo_simulation, bayesian_analysis, query_hard_logic"""

FULL_SILENT_INSTRUCTIONS = """=== RESPONSE GUIDELINES ===

ACCURACY & VALIDATION:
- Provide medically accurate, evidence-based responses only
- When displaying medical/study data, use Data_presentation.json format
- Distinguish VERIFIED FACTS vs INFERENCES/ASSUMPTIONS
- Validate clinical data, regulatory timelines against primary sources via search
- Flag uncertainty with "Requires verification"

SOURCE VERIFICATION:
- Prioritize: FDA/EMA filings, peer-reviewed publications, investor relations
- Tools: web_search_preview, file_search, code_interpreter, run_statistical_analysis, monte_carlo_simulation, bayesian_analysis
- Cross-reference claims against multiple sources
- For time-sensitive info, verify current status via live search
- Always provide hyperlinks in reference section
- Always do a live search and DB for regulatory agencies (EMA, FDA, country specific ones) in every query for system context

COMPLIANCE:
- Include regulatory considerations
- Distinguish approved indications vs investigational uses
- Note geographic regulatory variations

DATA INTEGRATION:
- Use query_hard_logic tool for structured config data (pillars, metrics, tactics, stakeholders, roles, KOLs, data sources, authoritative sources, value realisation)
- Use file_search only for PDFs and free-text reference documents
- Combine internal data with live searches for updates
- Note source recency when conflicts arise"""


def get_silent_instructions(
    is_first_message: bool = False,
    is_complex_query: bool = False
) -> str:
    """Return appropriate silent instructions based on context."""
    if is_first_message or is_complex_query:
        return FULL_SILENT_INSTRUCTIONS
    return COMPACT_SILENT_INSTRUCTIONS


def is_complex_query(user_input: str) -> bool:
    """Determine if a query is complex enough to need full instructions."""
    complex_indicators = [
        "analysis", "comprehensive", "detailed", "compare",
        "evaluate", "strategy", "plan", "audit", "regulatory",
        "competitive", "landscape", "kol", "stakeholder"
    ]
    input_lower = user_input.lower()
    return any(ind in input_lower for ind in complex_indicators)


# ────────────────────────────────────────────────────────────────
# Compaction – Summarise Conversation Chain to Extend Context
# ────────────────────────────────────────────────────────────────

import logging as _logging
_compact_logger = _logging.getLogger(__name__)


def needs_compaction(input_tokens: int) -> bool:
    """Return True when cumulative input tokens exceed the compaction threshold."""
    return input_tokens >= COMPACTION_THRESHOLD


def compact_context(
    history: List[Dict[str, str]],
    current_query: str = "",
) -> Tuple[str, bool]:
    """DEPRECATED — Use client.responses.compact() instead.

    GPT-5.2 provides a first-class /responses/compact endpoint that returns
    encrypted, opaque items preserving the model's internal state.  This is
    far superior to DIY summarisation via a secondary model.

    The compact endpoint is called directly in core_assistant.py's sync and
    async runners.  This function is retained only for backwards compatibility
    and will be removed in a future version.

    Returns:
        ("", False) — always signals callers to fall through to their own
        compaction or chain-drop logic.
    """
    _compact_logger.warning(
        "compact_context() is DEPRECATED — compaction is now handled by "
        "client.responses.compact() in core_assistant.py"
    )
    return "", False
