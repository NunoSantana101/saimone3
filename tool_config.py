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

The agent (GPT-4.1) handles:
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

OPTIMIZATION v3.0:
- Increased max_total from 40 to 80 for richer agent context
- Increased max_per_source from 15 to 25 for better coverage
- Doubled aggregated source limits across all sources
- Added intent-based adaptive limits for specialized queries
- Added intent-specific field preservation for rich context
- Added search transparency/audit metadata configuration
- Added clinical outcome extraction configuration
- Enabled include_mesh_metadata=True by default for transparency

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
# 1. RESULT TRUNCATION LIMITS (used by assistant.py)
#    These control how much data reaches the agent after fetching
# ────────────────────────────────────────────────────────────────

# Default truncation limits (v3.2: OPTIMIZED to prevent output overflow)
# These values are carefully calibrated to stay under 200KB output limit
DEFAULT_MAX_PER_SOURCE = 20          # Reduced from 25 - per-source result cap
DEFAULT_MAX_TOTAL = 50               # Reduced from 80 - aggregate result cap
DEFAULT_MAX_REFINEMENT_SUGGESTIONS = 4  # Reduced from 5 - follow-up suggestions

# ────────────────────────────────────────────────────────────────
# OUTPUT SIZE SAFETY LIMITS (v3.2 - Prevents Agent Silent Failures)
# ────────────────────────────────────────────────────────────────
# OpenAI's submit_tool_outputs has an undocumented limit (~256KB per output).
# These limits are calibrated to keep output under 200KB safely.
#
# CALCULATION BASIS:
# - Each result: ~500-700 bytes (title 200 + snippet 400 + fields)
# - MeSH metadata: ~2-5KB
# - Refinement suggestions: ~2KB
# - 60 results × 700 bytes = ~42KB (safe)
# - 80 results × 700 bytes = ~56KB (safe)
# - 100 results × 700 bytes = ~70KB (borderline with metadata)
# ────────────────────────────────────────────────────────────────

# Adaptive limits by query intent (agent gets more data for complex queries)
# v3.2: REDUCED limits to prevent output size overflow causing silent failures
INTENT_BASED_LIMITS = {
    "safety": {
        "max_per_source": 25,     # Reduced from 30
        "max_total": 60,          # Reduced from 100 (safety reports can be verbose)
        "refinement_suggestions": 5,  # Reduced from 6
        "priority_sources": ["faers", "fda_safety_communications", "pubmed"],
    },
    "regulatory": {
        "max_per_source": 30,     # Reduced from 50
        "max_total": 80,          # Reduced from 160 (was WAY too high)
        "refinement_suggestions": 5,  # Reduced from 8
        "priority_sources": ["regulatory_combined", "fda_drugs", "ema"],
    },
    "kol": {
        "max_per_source": 20,     # Reduced from 25
        "max_total": 60,          # Reduced from 80
        "refinement_suggestions": 4,  # Reduced from 5
        "priority_sources": ["openalex_kol", "pubmed_investigators", "pubmed"],
    },
    "clinical_trial": {
        "max_per_source": 25,     # Reduced from 30
        "max_total": 70,          # Reduced from 100 (trial data can be verbose)
        "refinement_suggestions": 4,  # Reduced from 5
        "priority_sources": ["clinicaltrials", "eu_clinical_trials", "who_ictrp"],
    },
    # v3.1: Regional intent types for LATAM and Asia markets
    "latam": {
        "max_per_source": 25,     # Reduced from 35
        "max_total": 70,          # Reduced from 100
        "refinement_suggestions": 4,  # Reduced from 6
        "priority_sources": ["latam_trials", "anvisa", "latam_regulatory", "paho", "regional_kol"],
    },
    "asia": {
        "max_per_source": 25,     # Reduced from 35
        "max_total": 70,          # Reduced from 100
        "refinement_suggestions": 4,  # Reduced from 6
        "priority_sources": ["asia_trials", "pmda", "nmpa", "asia_regulatory", "regional_kol"],
    },
    "general": {
        "max_per_source": 20,     # Reduced from 25
        "max_total": 50,          # Reduced from 80
        "refinement_suggestions": 4,  # Reduced from 5
        "priority_sources": ["clinicaltrials", "pubmed", "regulatory_combined"],
    },
}

# Field truncation limits for individual results
# v4: Content-first – the agent needs substance, not metadata overhead.
#     Increased content budget, stripped redundant fields in core_assistant.
RESULT_FIELD_LIMITS = {
    "title_max_chars": 150,
    "content_max_chars": 800,        # Up from 300 – actual text the agent reasons over
}


# ────────────────────────────────────────────────────────────────
# 2. AGGREGATED SEARCH SOURCE LIMITS (used by med_affairs_data.py)
#    These control how much data is fetched from each source
# ────────────────────────────────────────────────────────────────

# Per-source limits for aggregated_search (DOUBLED for richer results)
AGGREGATED_SOURCE_LIMITS = {
    # Literature
    "pubmed": 40,                    # Was 20
    "europe_pmc": 30,                # Was 15

    # Clinical Trials - Global
    "clinicaltrials": 50,            # Was 25
    "eu_clinical_trials": 40,        # Was 20
    "who_ictrp": 30,                 # Was 15

    # Clinical Trials - LATAM (v3.1)
    "rebec": 35,                     # Brazil registry
    "latam_trials": 50,              # Aggregated LATAM

    # Clinical Trials - Asia (v3.1)
    "ctri": 35,                      # India registry
    "chictr": 35,                    # China registry
    "jprn": 35,                      # Japan registry
    "asia_trials": 50,               # Aggregated Asia

    # Conference abstracts
    "asco": 30,
    "esmo": 30,

    # Regulatory - US
    "regulatory_combined": 80,       # Was 20
    "fda_drugs": 60,                 # Was 15

    # Regulatory - EU
    "ema": 60,                       # Was 15

    # Regulatory - LATAM (v3.1)
    "anvisa": 35,                    # Brazil - largest LATAM market
    "cofepris": 30,                  # Mexico - 2nd largest LATAM
    "latam_regulatory": 50,          # Combined LATAM

    # Regulatory - Asia (v3.1)
    "pmda": 35,                      # Japan - 3rd largest global
    "nmpa": 35,                      # China - 2nd largest global
    "cdsco": 30,                     # India - major hub
    "asia_regulatory": 50,           # Combined Asia

    # Safety/Pharmacovigilance (increased significantly for safety queries)
    "faers": 40,                     # Was 15
    "fda_safety_communications": 30, # Was 15

    # KOL Discovery
    "openalex_kol": 30,              # Was 15
    "pubmed_investigators": 30,      # Was 15
    "regional_kol": 40,              # v3.1: Regional KOL discovery

    # Epidemiology - Regional (v3.1)
    "paho": 35,                      # LATAM health data
}


# ────────────────────────────────────────────────────────────────
# 3. MESH METADATA LIMITS (used by assistant.py truncation)
#    Controls drug/indication metadata passed to agent
# ────────────────────────────────────────────────────────────────

# MeSH metadata limits (v3.2: Reduced to prevent output size overflow)
MESH_METADATA_LIMITS = {
    "qualifiers": 6,                 # Reduced from 10
    "mesh_records": 5,               # Reduced from 7
    "tree_numbers": 5,               # Reduced from 8
    "pharmacological_actions": 3,    # Reduced from 5
    "drug_mapping_indications": 3,   # Reduced from 5
    "drug_mapping_mechanism": 2,     # Reduced from 4
}

# Intent-based MeSH limits (more metadata for specialized queries)
INTENT_MESH_LIMITS = {
    "safety": {
        "qualifiers": 12,
        "mesh_records": 10,
        "pharmacological_actions": 8,
    },
    "regulatory": {
        "qualifiers": 12,
        "mesh_records": 10,
        "drug_mapping_indications": 8,
    },
}


# ────────────────────────────────────────────────────────────────
# 5. INTENT-SPECIFIC FIELD PRESERVATION (NEW in v3.0)
#    Defines which "extra" fields to preserve per intent type
#    Backend preserves these fields; Agent decides what's relevant
# ────────────────────────────────────────────────────────────────

INTENT_PRESERVE_FIELDS = {
    "clinical_trial": [
        "phase", "status", "enrollment", "conditions", "sponsor",
        "primary_outcome", "secondary_outcome", "arms", "interventions",
        "start_date", "completion_date", "study_type", "locations_count",
    ],
    "safety": [
        "serious", "outcome", "reaction", "report_date", "patient_age",
        "patient_sex", "reporter_type", "manufacturer", "product_names",
        "event_date", "hospitalization", "death", "disability",
    ],
    "regulatory": [
        "decision_type", "approval_date", "indication", "application_number",
        "submission_type", "review_priority", "orphan_status", "accelerated",
        "breakthrough", "fast_track", "applicant", "active_ingredient",
        "manufacturer", "product_type", "content", "score",
    ],
    "kol": [
        "authors", "affiliation", "institution", "h_index", "citation_count",
        "orcid", "publication_count", "works_count", "last_known_institution",
        "topics", "concepts",
    ],
    "competitive": [
        "manufacturer", "market_status", "launch_date", "patent_expiry",
        "exclusivity_end", "generic_available", "biosimilar_available",
    ],
    "real_world": [
        "study_design", "data_source", "sample_size", "follow_up_period",
        "outcomes_measured", "population", "setting",
    ],
    # v3.1: Regional intent field preservation
    "latam": [
        # Regulatory fields (ANVISA, COFEPRIS)
        "registration_number", "regulatory_agency", "country", "region",
        "active_ingredient", "therapeutic_class", "registration_date",
        "expiry_date", "company", "status", "category",
        # Trial fields (REBEC, LATAM trials)
        "phase", "enrollment", "sponsor", "condition", "intervention",
        "registry",
        # KOL/Epidemiology
        "institution", "institution_country", "h_index", "works_count",
    ],
    "asia": [
        # Regulatory fields (PMDA, NMPA, CDSCO)
        "registration_number", "regulatory_agency", "country", "region",
        "approval_type", "indication", "company", "applicant",
        # Trial fields (CTRI, ChiCTR, JPRN)
        "phase", "enrollment", "sponsor", "condition", "intervention",
        "registry",
        # KOL
        "institution", "institution_country", "h_index", "works_count",
        "orcid", "topics",
    ],
    "general": [
        "authors", "journal", "publication_type", "doi", "mesh_terms",
        "content", "score",
    ],
}

# Fields that should ALWAYS be preserved regardless of intent (for audit trail)
ALWAYS_PRESERVE_FIELDS = [
    "id", "title", "url", "date", "source",
]


# ────────────────────────────────────────────────────────────────
# 6. SEARCH TRANSPARENCY / AUDIT METADATA (NEW in v3.0)
#    Configuration for what audit/provenance info to include
#    Critical for regulated medical affairs environments
# ────────────────────────────────────────────────────────────────

SEARCH_TRANSPARENCY_CONFIG = {
    # Always include these in response for audit trail
    "include_search_metadata": True,
    "include_mesh_metadata": True,  # Changed from False to True by default
    "include_authority_scores": True,
    "include_intent_context": True,

    # What search metadata to include
    "search_metadata_fields": [
        "original_query",       # Exact user query
        "expanded_query",       # Query after MeSH expansion
        "sources_attempted",    # All sources tried
        "sources_successful",   # Sources that returned results
        "filters_applied",      # Date ranges, other filters
        "timestamp",            # When search was executed
        "cache_hit",            # Whether result was cached
    ],

    # Limitations transparency - tell agent what WASN'T searched
    "include_limitations": True,
    "limitation_reasons": [
        "source_unavailable",
        "rate_limited",
        "timeout",
        "no_results_found",
        "filter_excluded",
    ],
}

# Source authority scores for agent reasoning about source reliability
# Agent uses these to weight/prioritize results, NOT the backend
SOURCE_AUTHORITY_CONFIG = {
    "include_in_results": True,
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
# 8. REFINEMENT SUGGESTIONS CONFIG (Enhanced in v3.0)
#    More context for agent to understand follow-up options
# ────────────────────────────────────────────────────────────────

REFINEMENT_SUGGESTION_CONFIG = {
    "max_suggestions": 5,
    "include_rationale": True,         # Why this suggestion
    "include_expected_results": True,  # Estimated result count
    "include_source_hint": True,       # Which source to try
    "priority_weights": {
        "lifecycle_focus": 1.0,
        "therapy_area_expansion": 0.9,
        "safety_profile": 0.95,
        "kol_identification": 0.8,
        "indication_focus": 0.85,
        "mechanism_exploration": 0.7,
        "geographic_focus": 0.75,
        "temporal_focus": 0.8,
    },
}


# ────────────────────────────────────────────────────────────────
# 4. TOKEN BUDGETS (used by session_manager.py)
# ────────────────────────────────────────────────────────────────

CONTEXT_CONFIG = {
    "default_token_budget": 24_000,
    "extended_token_budget": 48_000,
    "maximum_token_budget": 96_000,
    "max_context_tokens": 1_000_000,
    "max_history_for_context": 40,
    "must_include_exchanges": 8,
    "checkpoint_frequency": 8,
    "checkpoint_max_tokens": 600,
    "context_cache_ttl": 300,
    "thread_validation_cache_ttl": 60,
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
    "TOKEN_BUDGETS",
    "CONTEXT_CONFIG",
    "API_LIMITS",
    # New v3.0 Constants
    "INTENT_PRESERVE_FIELDS",
    "ALWAYS_PRESERVE_FIELDS",
    "SEARCH_TRANSPARENCY_CONFIG",
    "SOURCE_AUTHORITY_CONFIG",
    "CLINICAL_OUTCOME_CONFIG",
    "REFINEMENT_SUGGESTION_CONFIG",
    # v3.3 Pipeline Constants
    "PIPELINE_CONFIG",
    "PIPELINE_TOOL_SCHEMAS",
    # Enums
    "QueryIntent",
    # Functions
    "detect_query_intent",
    "get_truncation_limits",
    "get_mesh_limits",
    "get_source_limit",
    # New v3.0 Functions
    "get_preserve_fields",
    "should_extract_clinical_outcomes",
    "get_intent_context",
]
