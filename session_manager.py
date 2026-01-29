"""
session_manager.py
Thread-based session context management with cost optimization and resilience patterns.

Refactored for OpenAI Assistants API thread_id-based context:
- OpenAI threads maintain full conversation history server-side
- Local context assembly is minimal - threads handle persistence
- Focus on thread validation, checkpointing, and resilience

Key features:
1. Circuit breaker pattern for API failures
2. Cached thread validation (reduces redundant API calls)
3. GPT-4.1 checkpoint summaries for session continuity
4. Thread-centric context management
5. Lazy context loading with increased token budgets

UPDATED v2.0:
- Integrated with tool_config.py for centralized configuration
- Token budgets and context settings now from centralized config
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
# Configuration Constants - Optimized for GPT-4.1 with Thread-based Context
# UPDATED v2.0: Now imports from centralized tool_config.py
# ────────────────────────────────────────────────────────────────

# Try to import from centralized config
try:
    from tool_config import (
        CONTEXT_CONFIG,
        TOKEN_BUDGETS,
        API_LIMITS,
    )
    # Use centralized config values
    DEFAULT_TOKEN_BUDGET = CONTEXT_CONFIG.get("default_token_budget", 24_000)
    EXTENDED_TOKEN_BUDGET = CONTEXT_CONFIG.get("extended_token_budget", 48_000)
    MAXIMUM_TOKEN_BUDGET = CONTEXT_CONFIG.get("maximum_token_budget", 96_000)
    MAX_CONTEXT_TOKENS = CONTEXT_CONFIG.get("max_context_tokens", 1_000_000)
    MAX_HISTORY_FOR_CONTEXT = CONTEXT_CONFIG.get("max_history_for_context", 40)
    CONTEXT_CACHE_TTL = CONTEXT_CONFIG.get("context_cache_ttl", 300)
    THREAD_VALIDATION_CACHE_TTL = CONTEXT_CONFIG.get("thread_validation_cache_ttl", 60)
    CHECKPOINT_MAX_TOKENS = CONTEXT_CONFIG.get("checkpoint_max_tokens", 600)
    _CONFIG_LOADED = True
except ImportError:
    # Fallback to local definitions if tool_config not available
    _CONFIG_LOADED = False
    # GPT-4.1 Context Optimization - INCREASED WINDOW SIZES
    DEFAULT_TOKEN_BUDGET = 24_000       # Increased from 16k for richer context
    EXTENDED_TOKEN_BUDGET = 48_000      # Increased from 32k for complex queries
    MAXIMUM_TOKEN_BUDGET = 96_000       # Increased from 64k for deep analysis
    MAX_CONTEXT_TOKENS = 1_000_000      # GPT-4.1 full capacity (1M tokens)
    CONTEXT_CACHE_TTL = 300  # 5 minutes
    THREAD_VALIDATION_CACHE_TTL = 60  # 1 minute
    MAX_HISTORY_FOR_CONTEXT = 40  # Increased for better continuity
    CHECKPOINT_MAX_TOKENS = 600

# GPT-4.1 Model Selection - Use gpt-4.1 throughout for consistency
MODEL_GPT41 = "gpt-4.1"  # Primary model for all operations
CHECKPOINT_MODEL = "gpt-4.1"  # Full GPT-4.1 for high-quality summaries

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD = 3  # Failures before opening circuit
CIRCUIT_BREAKER_TIMEOUT = 60  # Seconds before trying again
CIRCUIT_BREAKER_HALF_OPEN_REQUESTS = 1  # Requests to allow in half-open state


# ────────────────────────────────────────────────────────────────
# Circuit Breaker Pattern
# ────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


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
            # Check if timeout has passed
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                return True
            return False

        # Half-open: allow limited requests
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
# Cached Thread Validation
# ────────────────────────────────────────────────────────────────

@dataclass
class ThreadValidationCache:
    """Cache for thread validation results to reduce API calls."""
    _cache: Dict[str, Tuple[bool, str, float]] = field(default_factory=dict)

    def get(self, thread_id: str) -> Optional[Tuple[bool, str]]:
        """Get cached validation result if not expired."""
        if thread_id in self._cache:
            is_valid, error, timestamp = self._cache[thread_id]
            if time.time() - timestamp < THREAD_VALIDATION_CACHE_TTL:
                return (is_valid, error)
            del self._cache[thread_id]
        return None

    def set(self, thread_id: str, is_valid: bool, error: str = ""):
        """Cache validation result."""
        self._cache[thread_id] = (is_valid, error, time.time())

    def invalidate(self, thread_id: str):
        """Remove thread from cache."""
        self._cache.pop(thread_id, None)

    def clear(self):
        """Clear entire cache."""
        self._cache.clear()


# Global validation cache
_thread_validation_cache = ThreadValidationCache()


def validate_thread_exists_cached(thread_id: str) -> Tuple[bool, str]:
    """
    Check if thread exists with caching to reduce API calls.
    Returns (is_valid, error_message).
    """
    # Check cache first
    cached = _thread_validation_cache.get(thread_id)
    if cached is not None:
        return cached

    # Check circuit breaker
    cb = get_circuit_breaker("openai_threads")
    if cb.is_open():
        return (False, "Service temporarily unavailable - too many recent failures")

    if not cb.can_execute():
        # Return cached result if available, even if expired
        if thread_id in _thread_validation_cache._cache:
            is_valid, error, _ = _thread_validation_cache._cache[thread_id]
            return (is_valid, error + " (cached, circuit open)")
        return (False, "Service temporarily unavailable")

    try:
        openai.beta.threads.retrieve(thread_id)
        _thread_validation_cache.set(thread_id, True, "")
        cb.record_success()
        return (True, "")
    except openai.NotFoundError:
        _thread_validation_cache.set(thread_id, False, f"Thread {thread_id} not found")
        cb.record_success()  # API worked, just thread not found
        return (False, f"Thread {thread_id} not found - it may have been deleted or expired")
    except openai.BadRequestError as e:
        _thread_validation_cache.set(thread_id, False, str(e))
        cb.record_success()
        return (False, f"Thread {thread_id} is invalid: {str(e)}")
    except openai.RateLimitError as e:
        cb.record_failure()
        return (True, f"Rate limited - assuming thread valid: {str(e)}")
    except openai.APIConnectionError as e:
        cb.record_failure()
        return (True, f"Connection error - assuming thread valid: {str(e)}")
    except Exception as e:
        cb.record_failure()
        return (True, f"Warning: Could not validate thread: {str(e)}")


def invalidate_thread_cache(thread_id: str):
    """Invalidate cache for a specific thread (e.g., after deletion)."""
    _thread_validation_cache.invalidate(thread_id)


# ────────────────────────────────────────────────────────────────
# Context Assembly with Deduplication
# ────────────────────────────────────────────────────────────────

def _hash_message(msg: Dict[str, str]) -> str:
    """Create a hash of a message for deduplication."""
    content = f"{msg.get('role', '')}:{msg.get('content', '')[:500]}"
    return hashlib.md5(content.encode()).hexdigest()


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
    token_budget: int = DEFAULT_TOKEN_BUDGET,  # Increased to 24k for richer context
    priority_keywords: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Build minimal local context - OpenAI thread maintains full conversation server-side.

    Thread-based context management:
    - OpenAI Assistants API stores full conversation history in threads
    - Local context is supplementary for checkpointing and emergency recovery
    - Focus on recent exchanges for immediate continuity

    GPT-4.1 Optimizations:
    - Budget increased to 24k for richer context when needed
    - Last 8 exchanges included for good continuity (increased from 6)
    - Handles 40 messages in history for recent context

    Features:
    1. Deduplicates messages
    2. Limits history processing
    3. Better priority categorization
    4. More efficient token estimation
    """
    if not history:
        return []

    # Deduplicate and limit history size
    clean_history = deduplicate_history(history[-MAX_HISTORY_FOR_CONTEXT:])

    if len(clean_history) <= 8:
        return clean_history

    # Include last 8 exchanges for good continuity (increased from 6)
    # Thread maintains full history server-side, this is for local reference
    must_include = clean_history[-8:]
    remaining_budget = token_budget - estimate_tokens(must_include)

    if remaining_budget <= 0:
        return must_include

    # Categorize older messages by priority
    priority_buckets: Dict[str, List[Dict[str, str]]] = {
        "document_analysis": [],
        "regulatory": [],
        "compliance": [],
        "kol": [],
        "strategic": [],
        "general": [],
    }

    # Keywords for categorization
    category_keywords = {
        "document_analysis": ["upload", "document", "pdf", "file", "review", "analysis"],
        "regulatory": ["fda", "ema", "regulatory", "approval", "submission", "nda", "bla"],
        "compliance": ["compliance", "gdpr", "clinical trial", "audit", "mlr"],
        "kol": ["kol", "investigator", "opinion leader", "expert", "author"],
        "strategic": ["market", "competitive", "launch", "strategy", "roi", "stakeholder"],
    }

    # Add custom priority keywords if provided
    if priority_keywords:
        category_keywords["strategic"].extend(priority_keywords)

    for msg in clean_history[:-8]:  # Exclude last 8 (already in must_include)
        content_lower = msg.get("content", "").lower()
        categorized = False

        for category, keywords in category_keywords.items():
            if any(kw in content_lower for kw in keywords):
                priority_buckets[category].append(msg)
                categorized = True
                break

        if not categorized:
            priority_buckets["general"].append(msg)

    # Build context by priority
    chosen: List[Dict[str, str]] = []
    priority_order = [
        "document_analysis",
        "regulatory",
        "compliance",
        "kol",
        "strategic",
        "general"
    ]

    for bucket in priority_order:
        for msg in reversed(priority_buckets[bucket]):  # Most recent first within bucket
            cost = estimate_tokens([msg])
            if remaining_budget - cost < 100:
                break
            chosen.append(msg)
            remaining_budget -= cost

    # Sort by original order and combine with must-include
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
    Create a checkpoint summary using GPT-4.1.

    Thread-based Context:
    - Checkpoints complement OpenAI thread storage
    - Useful for session resumption and cross-session context
    - Stored locally as backup to server-side thread history

    GPT-4.1 Optimization:
    - Uses full gpt-4.1 for high-quality summaries
    - 600 max tokens for comprehensive checkpoints (increased from 500)
    - Leverages GPT-4.1's improved instruction following
    """
    # Check circuit breaker
    cb = get_circuit_breaker("openai_chat")
    if not cb.can_execute():
        return {
            "turn": len(history_slice),
            "summary": "Checkpoint skipped - service temporarily unavailable",
            "timestamp": datetime.now().isoformat(),
            "user": user_email,
            "error": "circuit_breaker_open"
        }

    # Compact the conversation for summarization
    compact_history = []
    for msg in history_slice[-16:]:  # Last 16 messages max
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate long messages
        if len(content) > 500:
            content = content[:250] + "...[truncated]..." + content[-200:]
        compact_history.append(f"{role}: {content}")

    convo_text = "\n".join(compact_history)

    # Focused prompt for efficiency
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

    Thread-based Context:
    - OpenAI threads maintain full conversation server-side
    - Local context is supplementary for checkpointing
    - Reduces redundant context building across multiple calls
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._history_hash: Optional[str] = None

    def _compute_history_hash(self, history: List[Dict[str, str]]) -> str:
        """Compute a hash of the history for cache invalidation."""
        if not history:
            return "empty"
        # Hash based on length and last message
        last_content = history[-1].get("content", "")[:100] if history else ""
        return f"{len(history)}:{hashlib.md5(last_content.encode()).hexdigest()}"

    def get_context(
        self,
        history: List[Dict[str, str]],
        token_budget: int = DEFAULT_TOKEN_BUDGET,  # Increased to 24k
        force_refresh: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get context with caching based on history state.

        Thread-based: OpenAI threads store full history, this is supplementary.
        Balanced for fast response times with GPT-4.1.
        """
        history_hash = self._compute_history_hash(history)
        cache_key = f"{history_hash}:{token_budget}"

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached_context, timestamp = self._cache[cache_key]
            if time.time() - timestamp < CONTEXT_CACHE_TTL:
                return cached_context

        # Build fresh context
        context = adaptive_context_builder(history, token_budget)
        self._cache[cache_key] = (context, time.time())

        # Clean old cache entries
        self._cleanup_cache()

        return context

    def _cleanup_cache(self):
        """Remove expired cache entries."""
        now = time.time()
        expired = [
            key for key, (_, timestamp) in self._cache.items()
            if now - timestamp > CONTEXT_CACHE_TTL * 2
        ]
        for key in expired:
            del self._cache[key]

    def invalidate(self):
        """Force cache invalidation."""
        self._cache.clear()


# Global context manager instance
_context_manager = LazyContextManager()


def get_optimized_context(
    history: List[Dict[str, str]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,  # Increased to 24k
    force_refresh: bool = False
) -> List[Dict[str, str]]:
    """
    Public interface for optimized context retrieval.

    Thread-based Context:
    - OpenAI threads store full conversation server-side via thread_id
    - This returns supplementary local context for checkpointing

    GPT-4.1 Optimized:
    - Default 24k budget for richer context (increased from 16k)
    - Quality context with deduplication
    """
    return _context_manager.get_context(history, token_budget, force_refresh)


# ────────────────────────────────────────────────────────────────
# Thread-based Context Management
# ────────────────────────────────────────────────────────────────

def get_thread_context_config() -> Dict[str, Any]:
    """
    Get configuration for thread-based context management.

    Thread-based context means:
    - OpenAI threads store full conversation history server-side
    - thread_id is the primary context reference
    - Local history is supplementary for checkpointing/UI display

    Returns configuration for GPT-4.1 thread-based context.
    """
    return {
        "model": MODEL_GPT41,
        "default_token_budget": DEFAULT_TOKEN_BUDGET,
        "extended_token_budget": EXTENDED_TOKEN_BUDGET,
        "maximum_token_budget": MAXIMUM_TOKEN_BUDGET,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "checkpoint_model": CHECKPOINT_MODEL,
        "checkpoint_max_tokens": CHECKPOINT_MAX_TOKENS,
        "context_cache_ttl": CONTEXT_CACHE_TTL,
        "thread_validation_cache_ttl": THREAD_VALIDATION_CACHE_TTL,
        "max_history_for_local": MAX_HISTORY_FOR_CONTEXT,
    }


def get_token_budget_for_mode(mode: str) -> int:
    """
    Get token budget based on context mode selection.

    Args:
        mode: One of "fast", "extended", "maximum"

    Returns:
        Token budget for the mode
    """
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
    """
    Get session metrics for monitoring and debugging.
    """
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
    """
    Determine if a checkpoint should be created.
    """
    if history_length <= 1:
        return False
    return (history_length - last_checkpoint_turn) >= checkpoint_freq


# ────────────────────────────────────────────────────────────────
# Health Check Functions
# ────────────────────────────────────────────────────────────────

def get_service_health() -> Dict[str, Any]:
    """
    Get health status of all circuit breakers.
    """
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

# Compact version of silent instructions (reduced from ~1000 to ~400 tokens)
COMPACT_SILENT_INSTRUCTIONS = """=== RESPONSE GUIDELINES ===
ACCURACY: Evidence-based only. Mark INFERENCES explicitly. Validate via live search. Flag uncertainty.
SOURCES: Prioritize FDA/EMA/PubMed. Cross-reference claims. Cite with hyperlinks.
COMPLIANCE: Include regulatory context. Distinguish approved vs investigational.
DATA: Search internal files first. Use stakeholder_taxonomy_v2.json, tactics_taxonomy_v2.json, pillars_v2.json, metrics_v2.json.
Available tools: get_med_affairs_data, get_pubmed_data, get_fda_data, get_ema_data, tavily_tool"""

# Full instructions only when needed (first message, complex queries)
FULL_SILENT_INSTRUCTIONS = """=== RESPONSE GUIDELINES ===

ACCURACY & VALIDATION:
- Provide medically accurate, evidence-based responses only
- When displaying medical/study data, use Data_presentation.json format
- Distinguish VERIFIED FACTS vs INFERENCES/ASSUMPTIONS
- Validate clinical data, regulatory timelines against primary sources via search
- Flag uncertainty with "Requires verification"

SOURCE VERIFICATION:
- Prioritize: FDA/EMA filings, peer-reviewed publications, investor relations
- Tools: get_med_affairs_data, get_pubmed_data, get_fda_data, get_ema_data, tavily_tool
- Cross-reference claims against multiple sources
- For time-sensitive info, verify current status via live search
- Always provide hyperlinks in reference section
- Always do a live search and DB for regulatory agencies (EMA, FDA, country specific ones) in every query for system context

COMPLIANCE:
- Include regulatory considerations 
- Distinguish approved indications vs investigational uses
- Note geographic regulatory variations

DATA INTEGRATION:
- Search internal files first for baseline
- Load stakeholder_taxonomy_v2.json, tactics_taxonomy_v2.json, pillars_v2.json, metrics_v2.json
- Combine internal data with live searches for updates
- Note source recency when conflicts arise"""


def get_silent_instructions(
    is_first_message: bool = False,
    is_complex_query: bool = False
) -> str:
    """
    Return appropriate silent instructions based on context.
    Saves ~600 tokens per request when compact version is sufficient.
    """
    if is_first_message or is_complex_query:
        return FULL_SILENT_INSTRUCTIONS
    return COMPACT_SILENT_INSTRUCTIONS


def is_complex_query(user_input: str) -> bool:
    """
    Determine if a query is complex enough to need full instructions.
    """
    complex_indicators = [
        "analysis", "comprehensive", "detailed", "compare",
        "evaluate", "strategy", "plan", "audit", "regulatory",
        "competitive", "landscape", "kol", "stakeholder"
    ]
    input_lower = user_input.lower()
    return any(ind in input_lower for ind in complex_indicators)
