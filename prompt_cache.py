"""
prompt_cache.py
GPT-5.2 Prompt Caching Architecture

Implements OpenAI's automatic prompt caching optimization for the
Responses API to maximize the 90% Cached Input Token Discount.

Three components:

1. **Static Context Block** (STATIC_CONTEXT_BLOCK)
   Immutable prefix assembled once at startup from core static assets.
   Verified to exceed 1024 tokens (OpenAI's cache eligibility threshold).
   Must be the absolute first content in every API call (the `instructions`
   parameter in the Responses API).

2. **Prefix Match Enforcement**
   STATIC_CONTEXT_BLOCK is used as the default `instructions` parameter
   in both sync and async runners.  No dynamic data (timestamps, user_ids,
   session context) is injected into this string -- all dynamic content
   goes into the `input` parameter (the "user" message).

3. **Cache Heartbeat Daemon** (CacheHeartbeat)
   Background thread that sends a minimal API request every 240 seconds
   to reset the ~5-minute server-side TTL on the cached prefix.
   Uses max_output_tokens=1 and no tools to minimize cost.

v8.0 -- Initial prompt caching architecture.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from typing import Optional

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Core static asset files loaded into the immutable prefix.
# system_instructions.txt is the primary asset (~16 KB, ~4 000 tokens).
# The three named dossier files are loaded if present; they are product-
# specific and may be added by the deployment team.
STATIC_ASSET_FILES = [
    os.path.join(_BASE_DIR, "system_instructions.txt"),
    os.path.join(_BASE_DIR, "Global_Value_Dossier.txt"),
    os.path.join(_BASE_DIR, "Product_Label.txt"),
    os.path.join(_BASE_DIR, "Compliance_Guidelines.txt"),
]

# OpenAI requires >1024 tokens in the prefix for cache activation.
CACHE_TOKEN_MINIMUM = 1024

# Heartbeat fires every 240 s (4 min) to stay inside the ~5 min TTL.
HEARTBEAT_INTERVAL_SECONDS = 240

# Maximum time (seconds) for a single heartbeat ping API call.
# Prevents the daemon thread from hanging indefinitely when OpenAI is slow.
HEARTBEAT_PING_TIMEOUT_SECONDS = 30

# Rough token estimation constant (1 token ~ 4 characters).
_CHARS_PER_TOKEN = 4


# ──────────────────────────────────────────────────────────────────────
#  Medical Affairs Glossary Padding
#  Appended ONLY when the assembled static content falls below 1024
#  tokens.  Contains neutral, domain-standard definitions that do not
#  alter model behaviour.
# ──────────────────────────────────────────────────────────────────────
_GLOSSARY_PADDING = """\

--- MEDICAL AFFAIRS GLOSSARY (REFERENCE DEFINITIONS) ---

Adverse Event (AE): Any untoward medical occurrence in a patient or \
clinical investigation subject administered a pharmaceutical product, \
which does not necessarily have a causal relationship with this treatment.

Benefit-Risk Assessment: A systematic evaluation comparing the \
therapeutic benefits of a medicinal product against its known and \
potential risks, used by regulatory authorities and sponsors to support \
decision-making throughout the product lifecycle.

Bioequivalence (BE): The absence of a significant difference in the \
rate and extent of absorption of the active ingredient or active moiety \
from pharmaceutical equivalents or alternatives under similar \
experimental conditions.

Clinical Development Plan (CDP): A comprehensive document detailing \
the clinical studies needed to establish the safety, efficacy, and \
optimal use of a drug across its intended patient population.

Compassionate Use: Access to an unauthorized medicinal product outside \
a clinical trial for patients with serious or life-threatening conditions \
who have no satisfactory authorized treatment alternatives.

Drug Safety Update (DSU): A periodic safety communication submitted to \
regulatory authorities during clinical development summarizing new safety \
information and updated benefit-risk assessments.

Efficacy: The capacity of a drug to produce the desired therapeutic \
effect under ideal and controlled conditions, as demonstrated in clinical \
trials.

FDA (Food and Drug Administration): The United States federal agency \
responsible for protecting and promoting public health through the \
regulation and supervision of pharmaceutical drugs, biologics, medical \
devices, food, and tobacco products.

EMA (European Medicines Agency): The decentralized agency of the \
European Union responsible for the scientific evaluation, supervision, \
and safety monitoring of medicines developed by pharmaceutical companies \
for use in the EU.

Good Clinical Practice (GCP): An international ethical and scientific \
quality standard for designing, conducting, recording, and reporting \
clinical trials that involve human subjects, ensuring participant rights, \
safety, and data integrity.

Health Technology Assessment (HTA): The systematic evaluation of the \
properties, effects, and impacts of health technologies and interventions \
to inform resource allocation and policy decisions.

Indication: The specific disease, condition, symptom, or clinical \
circumstance for which a drug or therapeutic product is approved or \
intended to be used.

Key Opinion Leader (KOL): A respected expert in a specific therapeutic \
area whose clinical experience, published research, and professional \
standing influence treatment practices and clinical decision-making among \
peers and healthcare systems.

Label Extension: The regulatory process of expanding a drug's approved \
indications, patient populations, dosage forms, or routes of \
administration beyond its original marketing authorization.

MAPS (Medical Affairs Professional Society): The global organization \
dedicated to advancing the medical affairs profession through education, \
networking, and the establishment of professional standards and best \
practices.

Medical Science Liaison (MSL): A scientific expert employed by a \
pharmaceutical or biotechnology company to serve as a bridge between the \
company and the medical community, communicating complex scientific and \
clinical data to healthcare professionals.

NDA (New Drug Application): A regulatory submission to the FDA \
containing all data and information necessary for the agency to evaluate \
whether a new drug is safe and effective for its proposed use.

Pharmacovigilance (PV): The science and activities relating to the \
detection, assessment, understanding, and prevention of adverse effects \
or any other drug-related problem throughout the product lifecycle.

Real-World Evidence (RWE): Clinical evidence regarding the usage, \
benefits, and risks of a medical product derived from the analysis of \
real-world data collected outside of conventional randomized controlled \
trials.

Standard of Care (SOC): The diagnostic and treatment process that a \
clinician should follow for a certain type of patient, illness, or \
clinical circumstance, reflecting current best practices and clinical \
guidelines.

Therapeutic Area (TA): A medical specialty or disease category such as \
oncology, cardiology, immunology, or neurology around which pharmaceutical \
development, clinical research, and medical affairs activities are \
organized.

Unmet Medical Need: A condition whose treatment or diagnosis is not \
adequately addressed by currently available therapies, representing an \
opportunity for new therapeutic development and clinical innovation.

Value Dossier: A comprehensive document presenting the clinical, \
economic, and humanistic value proposition of a pharmaceutical product to \
support health technology assessment, reimbursement decisions, and \
formulary placement.
"""


# ──────────────────────────────────────────────────────────────────────
#  Static Context Construction
# ──────────────────────────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token)."""
    return len(text) // _CHARS_PER_TOKEN


def _load_file_content(filepath: str) -> str:
    """Load a text file, returning empty string if not found."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        _logger.info(
            "Loaded static asset: %s (%d chars, ~%d tokens)",
            os.path.basename(filepath), len(content),
            _estimate_tokens(content),
        )
        return content
    except FileNotFoundError:
        _logger.debug("Static asset not found (skipped): %s", filepath)
        return ""
    except Exception as exc:
        _logger.warning("Failed to load static asset %s: %s", filepath, exc)
        return ""


def build_static_context() -> str:
    """Assemble the immutable static context block from core assets.

    Loads all available files listed in STATIC_ASSET_FILES and
    concatenates them into a single string.  Verifies the result
    exceeds the 1024-token cache threshold; if not, appends the
    medical affairs glossary as neutral padding.

    This function is called ONCE at module load time.  The returned
    string must NEVER be modified per-request -- no timestamps,
    user_ids, session state, or any other dynamic content.

    Returns:
        Assembled static context string (>1024 tokens guaranteed
        when sufficient source material exists).
    """
    sections: list[str] = []

    for filepath in STATIC_ASSET_FILES:
        content = _load_file_content(filepath)
        if content:
            sections.append(content.strip())

    if not sections:
        _logger.error(
            "No static assets found at any configured path. "
            "Prompt caching will not activate."
        )
        sections.append(
            "You are sAImone, an expert Medical Affairs AI assistant "
            "powered by GPT-5.2."
        )

    assembled = "\n\n".join(sections)
    token_count = _estimate_tokens(assembled)

    _logger.info(
        "Static context assembled: %d chars, ~%d tokens (from %d files)",
        len(assembled), token_count, len(sections),
    )

    # ── Token minimum enforcement ──
    if token_count < CACHE_TOKEN_MINIMUM:
        deficit = CACHE_TOKEN_MINIMUM - token_count
        _logger.info(
            "Static context below %d-token minimum (~%d tokens). "
            "Appending glossary padding (~%d deficit tokens).",
            CACHE_TOKEN_MINIMUM, token_count, deficit,
        )
        assembled = assembled + "\n\n" + _GLOSSARY_PADDING.strip()
        token_count = _estimate_tokens(assembled)
        _logger.info(
            "After padding: %d chars, ~%d tokens",
            len(assembled), token_count,
        )

    if token_count < CACHE_TOKEN_MINIMUM:
        _logger.warning(
            "Static context STILL below %d-token threshold after padding "
            "(%d tokens). OpenAI cache may not activate.",
            CACHE_TOKEN_MINIMUM, token_count,
        )

    return assembled


# ──────────────────────────────────────────────────────────────────────
#  Module-level immutable global  (built once at import time)
# ──────────────────────────────────────────────────────────────────────
STATIC_CONTEXT_BLOCK: str = build_static_context()

_static_tokens = _estimate_tokens(STATIC_CONTEXT_BLOCK)
_logger.info(
    "STATIC_CONTEXT_BLOCK ready: ~%d tokens | cache eligible: %s",
    _static_tokens, _static_tokens > CACHE_TOKEN_MINIMUM,
)


# ──────────────────────────────────────────────────────────────────────
#  Cache Heartbeat Daemon  (TTL Management)
# ──────────────────────────────────────────────────────────────────────
class CacheHeartbeat:
    """Background daemon that prevents prompt cache eviction.

    Sends a lightweight API request every ``interval`` seconds with
    the same ``STATIC_CONTEXT_BLOCK`` as instructions.  This resets
    the ~5-minute TTL on OpenAI's server-side prefix cache without
    affecting user conversation state.

    Heartbeat payload:
        model        = gpt-5.2  (same as production)
        instructions = STATIC_CONTEXT_BLOCK  (identical prefix)
        input        = "PING"
        max_output_tokens = 1  (minimize generation cost)
        reasoning    = {"effort": "none"}  (skip reasoning)
        store        = False  (don't persist heartbeat responses)
        (no tools)
    """

    def __init__(self, interval: int = HEARTBEAT_INTERVAL_SECONDS):
        self._interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._heartbeat_count = 0
        self._last_heartbeat_ts: Optional[float] = None
        self._last_cached_tokens: int = 0
        self._consecutive_failures: int = 0
        self._max_consecutive_failures = 5

    # ── Public properties ──

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict:
        return {
            "running": self.is_running,
            "heartbeat_count": self._heartbeat_count,
            "last_heartbeat_ts": self._last_heartbeat_ts,
            "last_cached_tokens": self._last_cached_tokens,
            "consecutive_failures": self._consecutive_failures,
            "interval_seconds": self._interval,
        }

    # ── Lifecycle ──

    def start(self) -> None:
        """Start the heartbeat daemon thread."""
        if self.is_running:
            _logger.debug("Cache heartbeat already running")
            return

        self._stop_event.clear()
        self._consecutive_failures = 0
        self._thread = threading.Thread(
            target=self._run_loop,
            name="prompt-cache-heartbeat",
            daemon=True,
        )
        self._thread.start()
        _logger.info(
            "Cache heartbeat started (interval=%ds, prefix=~%d tokens)",
            self._interval, _estimate_tokens(STATIC_CONTEXT_BLOCK),
        )

    def stop(self) -> None:
        """Stop the heartbeat daemon."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        _logger.info(
            "Cache heartbeat stopped after %d pings",
            self._heartbeat_count,
        )

    # ── Internal loop ──

    def _run_loop(self) -> None:
        """Main loop executed in the background thread."""
        while not self._stop_event.is_set():
            # Wait for the interval (or until stop is signalled)
            self._stop_event.wait(timeout=self._interval)
            if self._stop_event.is_set():
                break
            self._send_ping()

    def _send_ping(self) -> None:
        """Send a minimal API request to refresh the cache TTL.

        Uses a dedicated short-timeout client to prevent the daemon
        thread from hanging indefinitely when the API is slow or
        unresponsive.  The timeout is HEARTBEAT_PING_TIMEOUT_SECONDS
        (default 30s) — well under the 240s interval, so a slow ping
        never delays the next one.
        """
        try:
            # Lazy import to avoid circular dependency at module load time.
            # core_assistant imports prompt_cache for STATIC_CONTEXT_BLOCK;
            # the heartbeat only needs core_assistant at runtime.
            from core_assistant import get_client, DEFAULT_MODEL
            import httpx as _httpx

            client = get_client()

            # Use with_options to enforce a tight per-request timeout so
            # a stalled API call cannot block the daemon thread forever.
            ping_client = client.with_options(
                timeout=_httpx.Timeout(HEARTBEAT_PING_TIMEOUT_SECONDS),
            )
            response = ping_client.responses.create(
                model=DEFAULT_MODEL,
                instructions=STATIC_CONTEXT_BLOCK,
                input="PING",
                max_output_tokens=1,
                reasoning={"effort": "none"},
                store=False,
            )

            # Extract cache hit metrics from response usage
            cached_tokens = 0
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                # OpenAI reports cached tokens in input_tokens_details
                details = getattr(usage, "input_tokens_details", None)
                if details is not None:
                    if hasattr(details, "cached_tokens"):
                        cached_tokens = details.cached_tokens
                    elif isinstance(details, dict):
                        cached_tokens = details.get("cached_tokens", 0)

            self._heartbeat_count += 1
            self._last_heartbeat_ts = time.time()
            self._last_cached_tokens = cached_tokens
            self._consecutive_failures = 0

            _logger.info(
                "Cache heartbeat #%d OK (cached_tokens=%d)",
                self._heartbeat_count, cached_tokens,
            )

        except Exception as exc:
            self._consecutive_failures += 1
            _logger.warning(
                "Cache heartbeat ping failed (%d/%d): %s",
                self._consecutive_failures,
                self._max_consecutive_failures,
                exc,
            )
            if self._consecutive_failures >= self._max_consecutive_failures:
                _logger.error(
                    "Cache heartbeat halted: %d consecutive failures",
                    self._consecutive_failures,
                )
                self._stop_event.set()


# ──────────────────────────────────────────────────────────────────────
#  Module-level heartbeat instance + public API
# ──────────────────────────────────────────────────────────────────────
_heartbeat = CacheHeartbeat()


def start_cache_heartbeat() -> None:
    """Start the background cache-warming heartbeat.

    Should be called after the OpenAI API key is configured and the
    application is ready to make API calls.  Safe to call multiple
    times (idempotent).
    """
    _heartbeat.start()


def stop_cache_heartbeat() -> None:
    """Stop the background cache-warming heartbeat."""
    _heartbeat.stop()


def get_cache_stats() -> dict:
    """Return prompt cache statistics for monitoring/debugging."""
    return {
        **_heartbeat.stats,
        "static_block_chars": len(STATIC_CONTEXT_BLOCK),
        "static_block_tokens": _estimate_tokens(STATIC_CONTEXT_BLOCK),
        "cache_eligible": _estimate_tokens(STATIC_CONTEXT_BLOCK) > CACHE_TOKEN_MINIMUM,
        "token_minimum": CACHE_TOKEN_MINIMUM,
    }


# Clean up heartbeat on process exit
atexit.register(stop_cache_heartbeat)
