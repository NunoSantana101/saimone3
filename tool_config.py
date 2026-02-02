"""
tool_config.py - Centralized Configuration for Backend Tool Request Flow
────────────────────────────────────────────────────────────────────────────

DESIGN PHILOSOPHY:
Backend = "Dumb" transparent data provider (fetch, structure, audit trail)
Agent = Reasoning engine (interpret, prioritize, synthesize, recommend)

In a regulated medical affairs environment, the backend should:
- Be auditable: Show exactly what was searched and how
- Be transparent: Include MeSH expansions, filters applied, sources tried
- Provide provenance: Source authority, timestamps, data lineage
- NOT make clinical judgments: That's exclusively the agent's role
- Preserve all potentially relevant data for agent reasoning

The agent (GPT-5.2) handles:
- Intent interpretation from user queries
- Result prioritization and relevance ranking
- Cross-source synthesis and analysis
- Regulatory-appropriate recommendations
- Determining what's relevant from the rich data provided

Configuration Categories:
1. Result Truncation Limits - Controls data passed to agent (assistant.py)
2. Aggregated Search Limits - Per-source fetch limits (med_affairs_data.py)
3. Token Budgets - Context window management (session_manager.py)
4. Adaptive Limits - Intent-based dynamic adjustment
5. Field Preservation - Intent-specific important fields to keep
6. Search Transparency - Audit metadata for regulated environments
7. Clinical Outcome Extraction - Structured data extraction from abstracts
8. Refinement Suggestions - Follow-up query suggestions for agent
9. Agent Data Pipeline - Compression and structured data flow (v3.3)

OPTIMIZATION v4.0 – Token-Efficient Tiered Results:
- Reduced max_total across all intents (agent can't reason over 50+ items)
- Tiered content depth: top results get full text, tail gets headline-only
- Flat ranked list replaces results_by_source nesting (less JSON overhead)
- Deduplication removes cross-source duplicates before agent sees them
- Structured summary header gives agent reasoning head-start
- mesh_context and refinement_suggestions moved to lazy/optional
- URLs stripped by default (added back only for citation requests)
- Net effect: ~40-60% fewer tokens with equal or better agent reasoning

v4.3 – Reduced Default Limits (Lightweight Pipeline):
- Default max_total: 30→20, per-source: 12→8 (matches standard profile)
- Intent limits reduced ~30% across all categories
- Result tiers now driven by query profile from assistant_config.py
- Deep queries still get v4.2-level limits via profile override
- Estimated payload: ~60-100KB on standard queries (was 100-160KB)

v3.3 Agent Data Pipeline:
- Structured data compression (gzip+base64) for large results
- Data manifests with searchable summaries
- Chunk-based access for very large datasets
- Pipeline tools: decompress_pipeline_data, search_pipeline_manifest
- Configurable compression thresholds and chunk sizes
"""

from __future__ import annotations
from typing import Dict, Optional
from enum import Enum


# ────────────────────────────────────────────────────────────────
# 1. RESULT TRUNCATION LIMITS (used by core_assistant.py)
#    These control how much data reaches the agent after fetching
# ────────────────────────────────────────────────────────────────

# v4.3: Reduced defaults — standard queries don't need 30 results.
# Deep queries get higher limits via query profile override.
DEFAULT_MAX_PER_SOURCE = 8           # Was 12 — 8 is enough for standard queries
DEFAULT_MAX_TOTAL = 20               # Was 30 — 20 ranked results for standard
DEFAULT_MAX_REFINEMENT_SUGGESTIONS = 0  # Moved to lazy tool; 0 = don't include inline

# ────────────────────────────────────────────────────────────────
# OUTPUT SIZE SAFETY LIMITS
# ────────────────────────────────────────────────────────────────
# v4.2 CALCULATION BASIS (increased top-k count and content depth):
# - Tier-1 (top 8):  ~title 150 + content 1500 + fields 400 = ~2,050 chars
# - Tier-2 (9-20):   ~title 150 + content 800  + fields 300 = ~1,250 chars
# - Tier-3 (21+):    ~title 150 + content 200  + fields 150 = ~500 chars
# - Summary header:  ~500 chars
# - 30 results: 8×2050 + 12×1250 + 10×500 = 16,400+15,000+5,000 = ~36,400 chars
# - With JSON overhead: ~45KB base; ~100-160KB on heavy broad queries (200KB hard limit)
# ────────────────────────────────────────────────────────────────

# Adaptive limits by query intent
# v4.3: Reduced ~30% across all intents.  The agent gets ranked, deduped,
# tiered results — quality over quantity.  Deep queries can override these
# via query profile.
INTENT_BASED_LIMITS = {
    "safety": {
        "max_per_source": 10,
        "max_total": 25,          # Was 35 — still generous for safety signals
        "priority_sources": ["faers", "fda_safety_communications", "pubmed"],
    },
    "regulatory": {
        "max_per_source": 10,
        "max_total": 25,          # Was 40 — 25 is plenty for regulatory landscape
        "priority_sources": ["regulatory_combined", "fda_drugs", "ema"],
    },
    "kol": {
        "max_per_source": 10,
        "max_total": 25,          # Was 35
        "priority_sources": ["openalex_kol", "pubmed_investigators", "pubmed"],
    },
    "clinical_trial": {
        "max_per_source": 10,
        "max_total": 25,          # Was 35
        "priority_sources": ["clinicaltrials", "eu_clinical_trials", "who_ictrp"],
    },
    "latam": {
        "max_per_source": 10,
        "max_total": 25,          # Was 35
        "priority_sources": ["latam_trials", "anvisa", "latam_regulatory", "paho", "regional_kol"],
    },
    "asia": {
        "max_per_source": 10,
        "max_total": 25,          # Was 35
        "priority_sources": ["asia_trials", "pmda", "nmpa", "asia_regulatory", "regional_kol"],
    },
    "general": {
        "max_per_source": 8,
        "max_total": 20,          # Was 30
        "priority_sources": ["clinicaltrials", "pubmed", "regulatory_combined"],
    },
}

# ────────────────────────────────────────────────────────────────
# TIERED CONTENT DEPTH (NEW in v4.0)
# Top results get full content, tail results get headline-only.
# This matches how the agent actually reads: deep on top hits,
# scan the rest for breadth.
# ────────────────────────────────────────────────────────────────
RESULT_TIERS = {
    "tier_1_count": 5,       # Top N results: full content (was 8, now standard profile)
    "tier_1_content": 1200,  # chars of content for tier-1 (was 1500)
    "tier_2_count": 8,       # Next N results: summary content (was 12)
    "tier_2_content": 600,   # chars of content for tier-2 (was 800)
    "tier_3_content": 150,   # Remaining: title + key fields + brief snippet (was 200)
}

# Field truncation limits for individual results
RESULT_FIELD_LIMITS = {
    "title_max_chars": 150,
    "content_max_chars": 1500,       # Tier-1 default (was 900)
    "extra_field_max_chars": 300,    # More metadata room (was 250)
    "extra_list_max_items": 5,       # Restored (was 4)
}


# ────────────────────────────────────────────────────────────────
# 2. AGGREGATED SEARCH SOURCE LIMITS (used by med_affairs_data.py)
#    These control how much data is fetched from each source
# ────────────────────────────────────────────────────────────────

# Per-source fetch limits for aggregated_search
# v4.3: Reduced to sensible defaults.  The previous "doubled" limits
# fetched far more data than the agent could consume, inflating token
# cost and latency.  Deep queries can still override via query profile.
AGGREGATED_SOURCE_LIMITS = {
    # Literature
    "pubmed": 20,                    # Was 40 — 20 is enough for synthesis
    "europe_pmc": 15,                # Was 30

    # Clinical Trials - Global
    "clinicaltrials": 25,            # Was 50
    "eu_clinical_trials": 20,        # Was 40
    "who_ictrp": 15,                 # Was 30

    # Clinical Trials - LATAM (v3.1)
    "rebec": 15,                     # Was 35
    "latam_trials": 25,              # Was 50

    # Clinical Trials - Asia (v3.1)
    "ctri": 15,                      # Was 35
    "chictr": 15,                    # Was 35
    "jprn": 15,                      # Was 35
    "asia_trials": 25,               # Was 50

    # Conference abstracts
    "asco": 15,                      # Was 30
    "esmo": 15,                      # Was 30

    # Regulatory - US
    "regulatory_combined": 30,       # Was 80 — 30 covers most queries
    "fda_drugs": 25,                 # Was 60

    # Regulatory - EU
    "ema": 25,                       # Was 60

    # Regulatory - LATAM (v3.1)
    "anvisa": 15,                    # Was 35
    "cofepris": 15,                  # Was 30
    "latam_regulatory": 25,          # Was 50

    # Regulatory - Asia (v3.1)
    "pmda": 15,                      # Was 35
    "nmpa": 15,                      # Was 35
    "cdsco": 15,                     # Was 30
    "asia_regulatory": 25,           # Was 50

    # Safety/Pharmacovigilance
    "faers": 20,                     # Was 40
    "fda_safety_communications": 15, # Was 30

    # KOL Discovery
    "openalex_kol": 15,              # Was 30
    "pubmed_investigators": 15,      # Was 30
    "regional_kol": 20,              # Was 40

    # Epidemiology - Regional (v3.1)
    "paho": 15,                      # Was 35
}


# ────────────────────────────────────────────────────────────────
# 3. MESH METADATA LIMITS
#    v4.0: Stripped from default output.  The agent rarely references
#    MeSH qualifiers in responses.  Drug context (indications + mechanism)
#    is kept as a one-liner when available; everything else is dropped.
# ────────────────────────────────────────────────────────────────
MESH_METADATA_LIMITS = {
    "include_inline": False,         # v4.0: Don't include in default output
    "drug_mapping_indications": 3,   # Only if include_inline=True
    "drug_mapping_mechanism": 2,
}

INTENT_MESH_LIMITS = {
    "safety": {"include_inline": True, "drug_mapping_indications": 4},
    "regulatory": {"include_inline": True, "drug_mapping_indications": 4},
}


# ────────────────────────────────────────────────────────────────
# 5. INTENT-SPECIFIC FIELD PRESERVATION
#    v4.0: Trimmed to top-5 most-populated fields per intent.
#    Every extra field costs ~50 tokens.  Only keep the fields
#    the agent actually references in synthesis.
# ────────────────────────────────────────────────────────────────

INTENT_PRESERVE_FIELDS = {
    "clinical_trial": [
        "phase", "status", "enrollment", "conditions", "sponsor",
    ],
    "safety": [
        "serious", "outcome", "reaction", "manufacturer", "product_names",
    ],
    "regulatory": [
        "decision_type", "approval_date", "indication", "applicant",
        "application_number",
    ],
    "kol": [
        "authors", "affiliation", "h_index", "citation_count", "institution",
    ],
    "competitive": [
        "manufacturer", "market_status", "launch_date", "patent_expiry",
    ],
    "real_world": [
        "study_design", "sample_size", "outcomes_measured", "population",
    ],
    "latam": [
        "country", "regulatory_agency", "phase", "status", "institution",
    ],
    "asia": [
        "country", "regulatory_agency", "phase", "indication", "institution",
    ],
    "general": [
        "authors", "journal", "publication_type",
    ],
}

# Fields that should ALWAYS be preserved regardless of intent (for audit trail)
ALWAYS_PRESERVE_FIELDS = [
    "id", "title", "url", "date", "source",
]


# ────────────────────────────────────────────────────────────────
# 6. SEARCH TRANSPARENCY / AUDIT METADATA
#    v4.0: Audit metadata is logged server-side but NOT sent to the
#    agent by default.  It consumes tokens without helping synthesis.
#    The agent gets a one-line "sources_searched" count instead.
# ────────────────────────────────────────────────────────────────

SEARCH_TRANSPARENCY_CONFIG = {
    "include_search_metadata": False,    # v4.0: Logged, not sent to agent
    "include_mesh_metadata": False,      # v4.0: Stripped (see MESH_METADATA_LIMITS)
    "include_authority_scores": False,    # v4.0: Used for ranking, not sent raw
    "include_intent_context": False,      # v4.0: Agent detects intent itself
}

# Source authority scores – used internally for ranking results before
# sending to agent.  The agent sees ranked output, not raw scores.
SOURCE_AUTHORITY_CONFIG = {
    "include_in_results": False,         # v4.0: Used for ranking only
    "score_field_name": "_authority",
    "categories": ["regulatory", "peer_reviewed", "preprint", "database", "conference"],
}


# ────────────────────────────────────────────────────────────────
# 7. CLINICAL OUTCOME EXTRACTION (NEW in v3.0)
#    Configuration for extracting structured clinical data from text
#    Backend extracts; Agent interprets and synthesizes
# ────────────────────────────────────────────────────────────────

CLINICAL_OUTCOME_CONFIG = {
    # Enable extraction by default for clinical trial intent
    "enabled_by_default": True,
    "enabled_for_intents": ["clinical_trial", "safety", "regulatory"],

    # What outcomes to extract
    "extract_efficacy": True,      # PFS, OS, ORR, CR, DOR
    "extract_safety": True,        # AE rates, SAE, discontinuation
    "extract_statistics": True,    # HR, CI, p-values, sample size
    "extract_identifiers": True,   # NCT IDs, DOIs, PMIDs

    # Limits (to prevent bloat)
    "max_outcomes_per_result": 10,
    "include_confidence": True,    # Include extraction confidence
}


# ────────────────────────────────────────────────────────────────
# 8. REFINEMENT SUGGESTIONS CONFIG
#    v4.0: Disabled inline.  Agent can generate its own follow-up
#    suggestions and these consumed ~100 tokens per search with
#    near-zero usage.  Kept for backwards compat if re-enabled.
# ────────────────────────────────────────────────────────────────

REFINEMENT_SUGGESTION_CONFIG = {
    "include_inline": False,           # v4.0: Disabled – agent generates its own
    "max_suggestions": 3,              # If re-enabled, reduced from 5
    "include_rationale": False,        # Stripped – token-expensive, rarely read
    "include_expected_results": False,
    "include_source_hint": False,
}


# ────────────────────────────────────────────────────────────────
# 4. TOKEN BUDGETS (used by session_manager.py)
# ────────────────────────────────────────────────────────────────

CONTEXT_CONFIG = {
    "default_token_budget": 24_000,
    "extended_token_budget": 48_000,
    "maximum_token_budget": 96_000,
    "max_context_tokens": 400_000,
    "max_history_for_context": 40,
    "must_include_exchanges": 8,
    "checkpoint_frequency": 8,
    "checkpoint_max_tokens": 600,
    "context_cache_ttl": 300,
    "thread_validation_cache_ttl": 60,
    # Compaction: when cumulative input_tokens across tool rounds exceeds this
    # threshold, trigger /responses/compact to avoid context_length_exceeded.
    # GPT-5.2 first-class compaction uses the same model — no separate model needed.
    "compaction_threshold": 300_000,     # 75% of 400K context window
    # DEPRECATED: compaction_model and compaction_max_tokens are no longer used.
    # Compaction is now handled by client.responses.compact() which uses the
    # same GPT-5.2 model and returns encrypted opaque items (no token limit).
    "compaction_model": "gpt-5.2",      # Kept for backwards compat only
    "compaction_max_tokens": 0,          # Unused — /responses/compact manages this
}

TOKEN_BUDGETS = {
    "fast": 24_000,
    "standard": 48_000,
    "extended": 64_000,
    "comprehensive": 96_000,
    "maximum": 128_000,
}

API_LIMITS = {
    "max_retries": 4,
    "initial_retry_delay": 1.0,
    "retry_backoff_multiplier": 2,
    "request_timeout": 30,
    "polling_interval": 1.0,
}


# ────────────────────────────────────────────────────────────────
# 5. QUERY INTENT DETECTION
#    Used by assistant.py to select adaptive limits
# ────────────────────────────────────────────────────────────────

class QueryIntent(Enum):
    """Query intent categories for adaptive configuration."""
    SAFETY = "safety"
    REGULATORY = "regulatory"
    KOL = "kol"
    CLINICAL_TRIAL = "clinical_trial"
    LATAM = "latam"          # v3.1: LATAM regional intent
    ASIA = "asia"            # v3.1: Asia regional intent
    GENERAL = "general"


def detect_query_intent(query: str) -> QueryIntent:
    """
    Detect query intent for adaptive limit selection.
    This helps the backend pass more relevant data to the agent.

    v3.1: Added regional intent detection for LATAM and Asia markets.
    Regional intents take priority when explicit regional context is detected.
    """
    query_lower = query.lower()

    # v3.1: Regional intent detection (check first for explicit regional context)
    latam_keywords = [
        "brazil", "brasil", "mexico", "méxico", "argentina", "colombia",
        "chile", "peru", "perú", "latam", "latin america", "south america",
        "anvisa", "cofepris", "anmat", "invima", "rebec", "paho",
        # Portuguese/Spanish medical terms
        "ensaio clínico", "ensayo clínico", "aprovação", "aprobación",
        "registro sanitário", "registro sanitario"
    ]
    if any(kw in query_lower for kw in latam_keywords):
        return QueryIntent.LATAM

    asia_keywords = [
        "japan", "china", "india", "korea", "singapore", "thailand",
        "asia", "apac", "asia pacific", "asian",
        "pmda", "nmpa", "cde", "cdsco", "mfds", "hsa",
        "ctri", "chictr", "jprn", "jrct",
        # Japanese/Chinese agency names
        "厚生労働省", "药监局", "国家药品监督管理局"
    ]
    if any(kw in query_lower for kw in asia_keywords):
        return QueryIntent.ASIA

    safety_keywords = [
        "safety", "adverse", "side effect", "toxicity", "faers",
        "pharmacovigilance", "black box", "warning", "rems", "recall"
    ]
    if any(kw in query_lower for kw in safety_keywords):
        return QueryIntent.SAFETY

    regulatory_keywords = [
        "approval", "fda", "ema", "submission", "pdufa", "nda", "bla",
        "505", "breakthrough", "accelerated", "label", "indication"
    ]
    if any(kw in query_lower for kw in regulatory_keywords):
        return QueryIntent.REGULATORY

    kol_keywords = [
        "kol", "key opinion", "expert", "author", "investigator",
        "principal", "researcher", "publication"
    ]
    if any(kw in query_lower for kw in kol_keywords):
        return QueryIntent.KOL

    trial_keywords = [
        "trial", "study", "phase", "enrollment", "efficacy",
        "endpoint", "randomized", "placebo", "arm"
    ]
    if any(kw in query_lower for kw in trial_keywords):
        return QueryIntent.CLINICAL_TRIAL

    return QueryIntent.GENERAL


def get_truncation_limits(intent: Optional[QueryIntent] = None) -> Dict[str, int]:
    """Get truncation limits based on query intent."""
    if intent and intent.value in INTENT_BASED_LIMITS:
        return INTENT_BASED_LIMITS[intent.value]
    return {
        "max_per_source": DEFAULT_MAX_PER_SOURCE,
        "max_total": DEFAULT_MAX_TOTAL,
        "refinement_suggestions": DEFAULT_MAX_REFINEMENT_SUGGESTIONS,
    }


def get_mesh_limits(intent: Optional[QueryIntent] = None) -> Dict[str, int]:
    """Get MeSH metadata truncation limits based on query intent."""
    limits = dict(MESH_METADATA_LIMITS)
    if intent and intent.value in INTENT_MESH_LIMITS:
        limits.update(INTENT_MESH_LIMITS[intent.value])
    return limits


def get_source_limit(source: str) -> int:
    """Get per-source result limit."""
    return AGGREGATED_SOURCE_LIMITS.get(source, 30)


# ────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (NEW in v3.0)
# ────────────────────────────────────────────────────────────────

def get_preserve_fields(intent: Optional[QueryIntent] = None) -> list:
    """Get fields to preserve in truncation based on intent."""
    base_fields = list(ALWAYS_PRESERVE_FIELDS)
    if intent and intent.value in INTENT_PRESERVE_FIELDS:
        base_fields.extend(INTENT_PRESERVE_FIELDS[intent.value])
    else:
        base_fields.extend(INTENT_PRESERVE_FIELDS.get("general", []))
    return list(set(base_fields))  # Dedupe


def should_extract_clinical_outcomes(intent: Optional[QueryIntent] = None) -> bool:
    """Check if clinical outcome extraction should be enabled for this intent."""
    if not CLINICAL_OUTCOME_CONFIG.get("enabled_by_default", False):
        return False
    if intent and intent.value in CLINICAL_OUTCOME_CONFIG.get("enabled_for_intents", []):
        return True
    return CLINICAL_OUTCOME_CONFIG.get("enabled_by_default", False)


def get_intent_context(intent: str, days: int, description: str) -> dict:
    """Build intent context metadata for agent transparency."""
    return {
        "detected_intent": intent,
        "description": description,
        "temporal_window_days": days,
        "priority_sources": INTENT_BASED_LIMITS.get(intent, {}).get("priority_sources", []),
        "preserve_fields": INTENT_PRESERVE_FIELDS.get(intent, INTENT_PRESERVE_FIELDS.get("general", [])),
    }


# ────────────────────────────────────────────────────────────────
# 9. AGENT DATA PIPELINE CONFIG (NEW in v3.3)
#    Configuration for structured data compression and decompression
#    Enables efficient data flow: agent → tool → search → compress → agent
# ────────────────────────────────────────────────────────────────

PIPELINE_CONFIG = {
    # Compression thresholds
    "min_size_for_compression": 10_000,      # Only compress if >10KB
    "compression_benefit_threshold": 0.7,     # Keep compressed if <70% of original
    "max_uncompressed_output": 150_000,       # Send uncompressed if under 150KB
    "max_compressed_payload": 200_000,        # 200KB max for compressed payloads

    # Manifest configuration
    "max_manifest_items": 100,                # Max items in manifest summary
    "max_preview_length": 150,                # Characters for preview text
    "max_field_preview": 50,                  # Characters for field previews

    # Chunking configuration
    "default_chunk_size": 25,                 # Results per chunk
    "max_chunks": 20,                         # Maximum number of chunks

    # Pipeline tool behavior
    "auto_compress": True,                    # Automatically compress large results
    "cache_compressed_data": True,            # Cache for decompression requests
    "cache_max_size": 50,                     # Max cached payloads

    # v1.1 Hybrid Pipeline - Vector Store Configuration
    "vector_store_threshold": 500_000,        # Use vector store if >500KB
    "vector_store_enabled": True,             # Enable vector store strategy
    "vector_store_upload_timeout": 60,        # Seconds to wait for file processing
    "vector_store_poll_interval": 2,          # Seconds between status checks
    "vector_file_results_per_file": 50,       # Max results per uploaded file
    "vector_store_cleanup_on_session_end": True,  # Auto-cleanup vector stores
}

# Pipeline tool schemas for OpenAI assistant
PIPELINE_TOOL_SCHEMAS = {
    "decompress_pipeline_data": {
        "name": "decompress_pipeline_data",
        "description": (
            "Decompress data that was compressed by the agent data pipeline. "
            "Use this when you receive a response with '_compression': 'gzip+base64' and need to access the full data. "
            "You can decompress all data at once, or access specific chunks for very large datasets. "
            "The manifest.items_summary contains searchable previews - check it first to find relevant items."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "manifest_id": {
                    "type": "string",
                    "description": "The manifest_id from the compressed response's manifest object"
                },
                "chunk_index": {
                    "type": "integer",
                    "description": "For chunked data (manifest.chunks_available > 1), specify which chunk to decompress (0-indexed). Each chunk contains ~25 results."
                },
                "item_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional: Extract only specific items by index after decompression (0-indexed, based on manifest.items_summary order)"
                }
            },
            "required": ["manifest_id"]
        }
    },
    "search_pipeline_manifest": {
        "name": "search_pipeline_manifest",
        "description": (
            "Search through compressed data manifest WITHOUT decompressing the full data. "
            "Use this to find relevant items before deciding which to decompress. "
            "The manifest contains searchable title/preview summaries for all results. "
            "This is much faster than decompressing all data when you only need specific items."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "manifest_id": {
                    "type": "string",
                    "description": "The manifest_id from the compressed response"
                },
                "search_term": {
                    "type": "string",
                    "description": "Text to search for in titles and preview snippets"
                },
                "source_filter": {
                    "type": "string",
                    "description": "Filter results by data source name (e.g., 'pubmed', 'clinicaltrials', 'fda')"
                },
                "date_filter": {
                    "type": "string",
                    "description": "Filter results by date (partial match, e.g., '2024' or '2024-01')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching items to return (default: 20)"
                }
            },
            "required": ["manifest_id"]
        }
    }
}


# ────────────────────────────────────────────────────────────────
# 10. READ_WEBPAGE TOOL SCHEMA (NEW in v4.1)
#     Copy-paste the JSON below into the Assistants API UI as a
#     new function tool alongside the existing search tools.
#     Enables the Search → Decide → Read pattern: the agent scans
#     truncated search results, picks the 1-2 most relevant, and
#     calls read_webpage to get the full page content on demand.
# ────────────────────────────────────────────────────────────────

READ_WEBPAGE_TOOL_SCHEMA = {
    "name": "read_webpage",
    "description": (
        "Read the full text content of a specific URL from search results. "
        "Use this AFTER reviewing search results when 1-2 results look particularly "
        "relevant and you need the complete content to answer the user's question accurately. "
        "The search results contain truncated snippets; this tool fetches the full page text. "
        "Only call this for URLs that appear in the search results – do not guess URLs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to read (from the 'url' field in search results)"
            },
            "context_query": {
                "type": "string",
                "description": "Optional: the user's original query, helps rank content within long pages"
            }
        },
        "required": ["url"]
    }
}

# The above schema formatted for the Assistants API tools array:
# {
#     "type": "function",
#     "function": {
#         "name": "read_webpage",
#         "description": "Read the full text content of a specific URL from search results. Use this AFTER reviewing search results when 1-2 results look particularly relevant and you need the complete content to answer the user's question accurately. The search results contain truncated snippets; this tool fetches the full page text. Only call this for URLs that appear in the search results – do not guess URLs.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "url": {
#                     "type": "string",
#                     "description": "The URL to read (from the 'url' field in search results)"
#                 },
#                 "context_query": {
#                     "type": "string",
#                     "description": "Optional: the user's original query, helps rank content within long pages"
#                 }
#             },
#             "required": ["url"]
#         }
#     }
# }


# ────────────────────────────────────────────────────────────────
# EXPORTS
# ────────────────────────────────────────────────────────────────

__all__ = [
    # Constants
    "DEFAULT_MAX_PER_SOURCE",
    "DEFAULT_MAX_TOTAL",
    "DEFAULT_MAX_REFINEMENT_SUGGESTIONS",
    "INTENT_BASED_LIMITS",
    "AGGREGATED_SOURCE_LIMITS",
    "MESH_METADATA_LIMITS",
    "RESULT_FIELD_LIMITS",
    "RESULT_TIERS",
    "TOKEN_BUDGETS",
    "CONTEXT_CONFIG",
    "API_LIMITS",
    # v3.0 Constants
    "INTENT_PRESERVE_FIELDS",
    "ALWAYS_PRESERVE_FIELDS",
    "SEARCH_TRANSPARENCY_CONFIG",
    "SOURCE_AUTHORITY_CONFIG",
    "CLINICAL_OUTCOME_CONFIG",
    "REFINEMENT_SUGGESTION_CONFIG",
    # v3.3 Pipeline Constants
    "PIPELINE_CONFIG",
    "PIPELINE_TOOL_SCHEMAS",
    # v4.1 Read Webpage
    "READ_WEBPAGE_TOOL_SCHEMA",
    # Enums
    "QueryIntent",
    # Functions
    "detect_query_intent",
    "get_truncation_limits",
    "get_mesh_limits",
    "get_source_limit",
    "get_preserve_fields",
    "should_extract_clinical_outcomes",
    "get_intent_context",
]
