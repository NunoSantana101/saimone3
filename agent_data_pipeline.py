"""
agent_data_pipeline.py
Structured data pipeline for agent communication with compression support.

ARCHITECTURE (v1.1 - Hybrid Pipeline):

    +-----------------+     +------------------+     +----------------+
    |   Data Source   | --> |  Hybrid Pipeline | --> |     Agent      |
    |   (med_affairs) |     |                  |     |                |
    +-----------------+     +------------------+     +----------------+
                                    |
                    +---------------+---------------+
                    |               |               |
               [SMALL]         [MEDIUM]         [LARGE]
              <150KB          150-500KB          >500KB
                |               |                  |
            Direct           Compress          Vector Store
            Return         + Manifest           Upload
                |               |                  |
                v               v                  v
            JSON data      gzip+base64        file_search
                          + searchable          semantic
                            manifest            retrieval

STRATEGIES:
    1. DIRECT: Small data (<150KB) returned as-is with metadata
    2. COMPRESSED: Medium data (150-500KB) compressed with searchable manifest
    3. VECTOR_STORE: Large data (>500KB) uploaded to OpenAI vector store

KEY FEATURES:
    - Schema-based data structuring for consistency
    - Smart compression (only when beneficial - >10KB savings)
    - Data manifests with search/filter capabilities
    - Chunked data for pagination support
    - Vector store upload for semantic search on large datasets
    - Automatic strategy selection based on data size

v1.0 - Initial implementation with compression pipeline
v1.1 - Hybrid pipeline with vector store upload strategy
"""

from __future__ import annotations

import gzip
import base64
import json
import hashlib
import logging
import tempfile
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import openai

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Configuration Constants
# ──────────────────────────────────────────────────────────────────────
# Compression thresholds
MIN_SIZE_FOR_COMPRESSION = 10_000  # Only compress if >10KB
COMPRESSION_BENEFIT_THRESHOLD = 0.7  # Only keep compressed if <70% of original
MAX_UNCOMPRESSED_OUTPUT = 150_000  # 150KB - send uncompressed if under this
MAX_COMPRESSED_PAYLOAD = 200_000  # 200KB max for compressed payloads

# Manifest limits
MAX_MANIFEST_ITEMS = 100  # Max items in manifest summary
MAX_PREVIEW_LENGTH = 150  # Characters for preview text
MAX_FIELD_PREVIEW = 50  # Characters for field previews in manifest

# Chunk configuration
DEFAULT_CHUNK_SIZE = 25  # Results per chunk
MAX_CHUNKS = 20  # Maximum number of chunks to create

# ──────────────────────────────────────────────────────────────────────
#  Vector Store Configuration (v1.1 Hybrid Pipeline)
# ──────────────────────────────────────────────────────────────────────
VECTOR_STORE_THRESHOLD = 500_000  # 500KB - use vector store if larger
VECTOR_STORE_UPLOAD_TIMEOUT = 60  # seconds to wait for file processing
VECTOR_STORE_POLL_INTERVAL = 2  # seconds between status checks

# File formatting for vector store
VECTOR_FILE_MAX_CHARS = 50_000  # Max chars per file section
VECTOR_FILE_RESULTS_PER_FILE = 50  # Max results per uploaded file


# ──────────────────────────────────────────────────────────────────────
#  Data Schemas (Enums)
# ──────────────────────────────────────────────────────────────────────
class DataCategory(str, Enum):
    """Categories of data that can flow through the pipeline."""
    SEARCH_RESULTS = "search_results"
    CLINICAL_TRIALS = "clinical_trials"
    REGULATORY = "regulatory"
    SAFETY = "safety"
    KOL = "kol"
    STATISTICAL = "statistical"
    LITERATURE = "literature"
    AGGREGATED = "aggregated"
    RAW = "raw"


class CompressionState(str, Enum):
    """State of data compression in the pipeline."""
    UNCOMPRESSED = "uncompressed"
    COMPRESSED = "compressed"
    CHUNKED = "chunked"
    HYBRID = "hybrid"  # Some compressed, some not
    VECTOR_STORE = "vector_store"  # v1.1: Uploaded to OpenAI vector store


class PipelineStrategy(str, Enum):
    """Strategy used by the hybrid pipeline."""
    DIRECT = "direct"  # Small data - return directly
    COMPRESSED = "compressed"  # Medium data - gzip + manifest
    VECTOR_STORE = "vector_store"  # Large data - upload to vector store


# ──────────────────────────────────────────────────────────────────────
#  Data Classes for Structured Pipeline
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DataManifestItem:
    """Single item in the data manifest - summarizes one result."""
    index: int
    id: str
    title: str
    source: str
    date: Optional[str] = None
    relevance_score: Optional[float] = None
    preview: Optional[str] = None
    fields_available: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DataManifest:
    """
    Manifest describing compressed/chunked data for the agent.

    The manifest provides:
    - Summary statistics about the data
    - Searchable index of items (titles, IDs, dates)
    - Instructions for accessing full data
    - Compression metadata
    """
    # Identification
    manifest_id: str
    category: DataCategory
    query: str
    timestamp: str

    # Statistics
    total_items: int
    compressed_size_bytes: int
    original_size_bytes: int
    compression_ratio: float

    # State
    compression_state: CompressionState
    chunks_available: int = 1

    # Searchable summary (always uncompressed for agent access)
    items_summary: List[Dict[str, Any]] = field(default_factory=list)

    # Field schema - what fields are available in full data
    available_fields: List[str] = field(default_factory=list)

    # Source metadata
    sources_included: List[str] = field(default_factory=list)
    intent_detected: Optional[str] = None

    # Access instructions for agent
    access_instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["category"] = self.category.value
        result["compression_state"] = self.compression_state.value
        return result


@dataclass
class CompressedPayload:
    """
    Compressed data payload with manifest.

    Structure:
    {
        "_pipeline_version": "1.0",
        "_compression": "gzip+base64",
        "manifest": { ... },  # Always readable
        "compressed_data": "...",  # Base64 encoded gzip
        "chunks": { "0": "...", "1": "...", ... }  # Optional chunked data
    }
    """
    manifest: DataManifest
    compressed_data: Optional[str] = None  # Base64 encoded gzip
    chunks: Dict[str, str] = field(default_factory=dict)  # Chunk index -> compressed data

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "_pipeline_version": "1.0",
            "_compression": "gzip+base64",
            "_decompression_available": True,
            "manifest": self.manifest.to_dict(),
        }
        if self.compressed_data:
            result["compressed_data"] = self.compressed_data
        if self.chunks:
            result["chunks"] = self.chunks
            result["_chunk_count"] = len(self.chunks)
        return result


# ──────────────────────────────────────────────────────────────────────
#  Core Compression Functions
# ──────────────────────────────────────────────────────────────────────
def compress_data(data: Union[str, bytes]) -> Tuple[str, int, int]:
    """
    Compress data using gzip and encode as base64.

    Returns:
        Tuple of (base64_encoded_compressed, original_size, compressed_size)
    """
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data

    original_size = len(data_bytes)
    compressed = gzip.compress(data_bytes, compresslevel=6)
    compressed_size = len(compressed)

    # Base64 encode for safe JSON transport
    encoded = base64.b64encode(compressed).decode('ascii')

    return encoded, original_size, compressed_size


def decompress_data(encoded_data: str) -> str:
    """
    Decompress base64-encoded gzip data.

    Args:
        encoded_data: Base64 encoded gzip compressed string

    Returns:
        Decompressed string
    """
    compressed = base64.b64decode(encoded_data)
    decompressed = gzip.decompress(compressed)
    return decompressed.decode('utf-8')


def generate_manifest_id(query: str, category: DataCategory) -> str:
    """Generate a unique manifest ID based on query and timestamp."""
    timestamp = datetime.utcnow().isoformat()
    content = f"{query}:{category.value}:{timestamp}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
#  Schema Detection and Field Mapping
# ──────────────────────────────────────────────────────────────────────
def detect_data_category(data: Dict[str, Any], source: str = None) -> DataCategory:
    """
    Detect the category of data based on structure and source.
    """
    # Check for aggregated search results
    if "results_by_source" in data:
        return DataCategory.AGGREGATED

    # Check source-based categories
    source_lower = (source or data.get("source", "")).lower()

    if any(s in source_lower for s in ["clinical", "trial", "ctgov", "euctr"]):
        return DataCategory.CLINICAL_TRIALS
    if any(s in source_lower for s in ["fda", "ema", "regulatory", "approval"]):
        return DataCategory.REGULATORY
    if any(s in source_lower for s in ["faers", "safety", "adverse"]):
        return DataCategory.SAFETY
    if any(s in source_lower for s in ["kol", "investigator", "openalex", "orcid"]):
        return DataCategory.KOL
    if any(s in source_lower for s in ["pubmed", "literature", "crossref", "biorxiv"]):
        return DataCategory.LITERATURE
    if any(s in source_lower for s in ["monte_carlo", "bayesian", "statistical"]):
        return DataCategory.STATISTICAL

    # Default to search results if has results array
    if "results" in data:
        return DataCategory.SEARCH_RESULTS

    return DataCategory.RAW


def extract_available_fields(results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract all available fields from a list of results.
    """
    if not results:
        return []

    fields = set()
    for result in results[:10]:  # Sample first 10 for efficiency
        fields.update(result.keys())
        # Also check 'extra' dict
        if "extra" in result and isinstance(result["extra"], dict):
            fields.update(f"extra.{k}" for k in result["extra"].keys())

    return sorted(list(fields))


def create_item_summary(
    item: Dict[str, Any],
    index: int,
    source: str = None,
) -> Dict[str, Any]:
    """
    Create a searchable summary of a single result item.
    """
    extra = item.get("extra", {})

    # Get preview text (snippet or abstract)
    preview_text = item.get("snippet", extra.get("abstract", ""))
    if preview_text and len(preview_text) > MAX_PREVIEW_LENGTH:
        preview_text = preview_text[:MAX_PREVIEW_LENGTH] + "..."

    # Collect available fields (excluding large ones)
    fields = [k for k in item.keys() if k not in ("extra", "snippet", "abstract")]
    if extra:
        fields.extend(f"extra.{k}" for k in extra.keys() if k not in ("abstract", "snippet"))

    return {
        "index": index,
        "id": item.get("id", item.get("pmid", item.get("nct_id", str(index)))),
        "title": str(item.get("title", ""))[:200],
        "source": source or item.get("_source", item.get("source", "unknown")),
        "date": item.get("date", extra.get("date", extra.get("publication_date"))),
        "preview": preview_text,
        "fields_available": fields[:15],  # Limit field list
    }


# ──────────────────────────────────────────────────────────────────────
#  Pipeline Processing Functions
# ──────────────────────────────────────────────────────────────────────
def process_through_pipeline(
    data: Dict[str, Any],
    query: str = "",
    force_compress: bool = False,
    max_uncompressed_size: int = MAX_UNCOMPRESSED_OUTPUT,
) -> Dict[str, Any]:
    """
    Main pipeline entry point - processes data for agent consumption.

    Decision tree:
    1. If data is small enough (<max_uncompressed_size), return as-is with metadata
    2. If compression provides significant benefit, compress full data
    3. If data is very large, chunk and compress
    4. Always include a readable manifest for the agent

    Args:
        data: Raw data from source (search results, etc.)
        query: Original query string
        force_compress: Force compression even for small data
        max_uncompressed_size: Max size before compression kicks in

    Returns:
        Processed data with manifest and optional compression
    """
    # Serialize to measure size
    try:
        serialized = json.dumps(data)
        original_size = len(serialized.encode('utf-8'))
    except (TypeError, ValueError) as e:
        _logger.error(f"Pipeline serialization failed: {e}")
        return {"error": "Pipeline serialization failed", "_pipeline_error": str(e)}

    # Detect category and extract metadata
    source = data.get("source", "unknown")
    category = detect_data_category(data, source)

    # Extract results for manifest
    results = []
    sources_included = []

    if "results_by_source" in data:
        for src, src_results in data.get("results_by_source", {}).items():
            if isinstance(src_results, list):
                sources_included.append(src)
                results.extend([(r, src) for r in src_results])
    elif "results" in data and isinstance(data["results"], list):
        sources_included = [source]
        results = [(r, source) for r in data["results"]]

    # Decision: compress or not?
    should_compress = force_compress or original_size > max_uncompressed_size

    if not should_compress:
        # Return with minimal pipeline metadata (no compression)
        return _add_pipeline_metadata(data, category, query, original_size, results)

    # Compress the data
    return _compress_and_package(
        data=data,
        serialized=serialized,
        original_size=original_size,
        category=category,
        query=query,
        results=results,
        sources_included=sources_included,
    )


def _add_pipeline_metadata(
    data: Dict[str, Any],
    category: DataCategory,
    query: str,
    original_size: int,
    results: List[Tuple[Dict, str]],
) -> Dict[str, Any]:
    """
    Add pipeline metadata to uncompressed data.
    """
    # Create mini-manifest for metadata
    data["_pipeline"] = {
        "version": "1.0",
        "compression_state": CompressionState.UNCOMPRESSED.value,
        "category": category.value,
        "query": query,
        "total_items": len(results),
        "size_bytes": original_size,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Add field index if results present
    if results:
        data["_pipeline"]["available_fields"] = extract_available_fields(
            [r for r, _ in results[:10]]
        )

    return data


def _compress_and_package(
    data: Dict[str, Any],
    serialized: str,
    original_size: int,
    category: DataCategory,
    query: str,
    results: List[Tuple[Dict, str]],
    sources_included: List[str],
) -> Dict[str, Any]:
    """
    Compress data and create manifest package.
    """
    # Compress
    encoded, _, compressed_size = compress_data(serialized)

    # Calculate compression benefit (account for base64 overhead ~33%)
    effective_compressed_size = len(encoded.encode('utf-8'))
    compression_ratio = effective_compressed_size / original_size if original_size > 0 else 1.0

    # If compression doesn't help much, return uncompressed with truncation
    if compression_ratio > COMPRESSION_BENEFIT_THRESHOLD:
        _logger.info(f"Compression ratio {compression_ratio:.2f} not beneficial, returning uncompressed")
        return _add_pipeline_metadata(data, category, query, original_size, results)

    # Create manifest with summaries
    items_summary = [
        create_item_summary(item, idx, source)
        for idx, (item, source) in enumerate(results[:MAX_MANIFEST_ITEMS])
    ]

    # Determine chunking
    total_items = len(results)
    should_chunk = total_items > DEFAULT_CHUNK_SIZE * 2 and effective_compressed_size > MAX_COMPRESSED_PAYLOAD

    manifest = DataManifest(
        manifest_id=generate_manifest_id(query, category),
        category=category,
        query=query,
        timestamp=datetime.utcnow().isoformat(),
        total_items=total_items,
        compressed_size_bytes=effective_compressed_size,
        original_size_bytes=original_size,
        compression_ratio=round(compression_ratio, 3),
        compression_state=CompressionState.CHUNKED if should_chunk else CompressionState.COMPRESSED,
        items_summary=items_summary,
        available_fields=extract_available_fields([r for r, _ in results[:10]]),
        sources_included=sources_included,
        intent_detected=data.get("intent_context", {}).get("detected_intent"),
        access_instructions=_generate_access_instructions(should_chunk, total_items),
    )

    payload = CompressedPayload(manifest=manifest)

    if should_chunk:
        # Create chunks
        payload.chunks = _create_chunks(results, category)
        manifest.chunks_available = len(payload.chunks)
    else:
        payload.compressed_data = encoded

    return payload.to_dict()


def _create_chunks(
    results: List[Tuple[Dict, str]],
    category: DataCategory,
) -> Dict[str, str]:
    """
    Create compressed chunks of results.
    """
    chunks = {}
    total = len(results)
    chunk_size = DEFAULT_CHUNK_SIZE

    for i in range(0, min(total, MAX_CHUNKS * chunk_size), chunk_size):
        chunk_items = results[i:i + chunk_size]
        chunk_data = [{"_source": src, **item} for item, src in chunk_items]
        chunk_json = json.dumps(chunk_data)
        encoded, _, _ = compress_data(chunk_json)
        chunks[str(i // chunk_size)] = encoded

    return chunks


def _generate_access_instructions(chunked: bool, total_items: int) -> str:
    """Generate instructions for the agent on how to access the data."""
    if chunked:
        return (
            f"Data contains {total_items} items across multiple chunks. "
            "Use 'decompress_pipeline_data' tool with chunk_index to access specific chunks. "
            "Review manifest.items_summary for item locations. "
            "Chunk 0 contains the first 25 items, chunk 1 contains items 25-49, etc."
        )
    else:
        return (
            f"Data contains {total_items} items in compressed format. "
            "Use 'decompress_pipeline_data' tool to access full results. "
            "manifest.items_summary contains searchable summaries of all items."
        )


# ──────────────────────────────────────────────────────────────────────
#  Decompression Functions (for agent tool calls)
# ──────────────────────────────────────────────────────────────────────
def decompress_pipeline_data(
    pipeline_data: Dict[str, Any],
    chunk_index: Optional[int] = None,
    item_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Decompress pipeline data - called by agent via tool.

    Args:
        pipeline_data: The compressed pipeline output
        chunk_index: Specific chunk to decompress (0-indexed)
        item_indices: Specific item indices to extract (after decompression)

    Returns:
        Decompressed data or specific items
    """
    if "_pipeline_version" not in pipeline_data:
        return {"error": "Not a pipeline-compressed payload"}

    manifest = pipeline_data.get("manifest", {})
    compression_state = manifest.get("compression_state", "unknown")

    # Handle chunked data
    if compression_state == CompressionState.CHUNKED.value:
        chunks = pipeline_data.get("chunks", {})
        if chunk_index is not None:
            chunk_key = str(chunk_index)
            if chunk_key not in chunks:
                available = list(chunks.keys())
                return {
                    "error": f"Chunk {chunk_index} not found",
                    "available_chunks": available,
                }
            try:
                decompressed = decompress_data(chunks[chunk_key])
                items = json.loads(decompressed)
                if item_indices:
                    items = [items[i] for i in item_indices if i < len(items)]
                return {
                    "chunk_index": chunk_index,
                    "items": items,
                    "total_in_chunk": len(items),
                }
            except Exception as e:
                return {"error": f"Decompression failed: {str(e)}"}
        else:
            # Return chunk info
            return {
                "available_chunks": list(chunks.keys()),
                "total_chunks": len(chunks),
                "items_per_chunk": DEFAULT_CHUNK_SIZE,
                "instruction": "Specify chunk_index to decompress a specific chunk",
            }

    # Handle single compressed payload
    elif compression_state == CompressionState.COMPRESSED.value:
        compressed_data = pipeline_data.get("compressed_data")
        if not compressed_data:
            return {"error": "No compressed data found"}

        try:
            decompressed = decompress_data(compressed_data)
            full_data = json.loads(decompressed)

            # If specific items requested
            if item_indices:
                results = full_data.get("results", [])
                if "results_by_source" in full_data:
                    results = []
                    for src_results in full_data["results_by_source"].values():
                        results.extend(src_results)
                selected = [results[i] for i in item_indices if i < len(results)]
                return {
                    "selected_items": selected,
                    "indices_requested": item_indices,
                    "total_available": len(results),
                }

            return full_data

        except Exception as e:
            return {"error": f"Decompression failed: {str(e)}"}

    return {"error": f"Unknown compression state: {compression_state}"}


def search_manifest(
    manifest: Dict[str, Any],
    search_term: str = None,
    source_filter: str = None,
    date_filter: str = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Search through manifest summaries without decompressing full data.

    Args:
        manifest: The manifest from pipeline output
        search_term: Text to search in titles/previews
        source_filter: Filter by source name
        date_filter: Filter by date (partial match)
        limit: Max results to return

    Returns:
        Matching items from manifest
    """
    items = manifest.get("items_summary", [])
    results = []

    for item in items:
        # Apply filters
        if search_term:
            search_lower = search_term.lower()
            title = str(item.get("title", "")).lower()
            preview = str(item.get("preview", "")).lower()
            if search_lower not in title and search_lower not in preview:
                continue

        if source_filter:
            if source_filter.lower() not in str(item.get("source", "")).lower():
                continue

        if date_filter:
            if date_filter not in str(item.get("date", "")):
                continue

        results.append(item)

        if len(results) >= limit:
            break

    return {
        "matches": results,
        "total_matches": len(results),
        "searched_items": len(items),
        "filters_applied": {
            "search_term": search_term,
            "source_filter": source_filter,
            "date_filter": date_filter,
        },
    }


# ──────────────────────────────────────────────────────────────────────
#  Tool Handler for Agent Access
# ──────────────────────────────────────────────────────────────────────
def handle_pipeline_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle pipeline-related tool calls from the agent.

    Available tools:
    - decompress_pipeline_data: Decompress full data or specific chunks
    - search_pipeline_manifest: Search manifest without decompression
    """
    if tool_name == "decompress_pipeline_data":
        pipeline_data = args.get("pipeline_data", {})
        chunk_index = args.get("chunk_index")
        item_indices = args.get("item_indices")
        return decompress_pipeline_data(pipeline_data, chunk_index, item_indices)

    elif tool_name == "search_pipeline_manifest":
        manifest = args.get("manifest", {})
        return search_manifest(
            manifest,
            search_term=args.get("search_term"),
            source_filter=args.get("source_filter"),
            date_filter=args.get("date_filter"),
            limit=args.get("limit", 20),
        )

    return {"error": f"Unknown pipeline tool: {tool_name}"}


# ──────────────────────────────────────────────────────────────────────
#  Utility Functions
# ──────────────────────────────────────────────────────────────────────
def estimate_compression_benefit(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate compression benefit without actually compressing.
    Useful for deciding whether to use pipeline.
    """
    try:
        serialized = json.dumps(data)
        original_size = len(serialized.encode('utf-8'))

        # Heuristic: JSON with repeated strings compresses well
        # Estimate based on unique character ratio
        unique_chars = len(set(serialized))
        total_chars = len(serialized)
        char_ratio = unique_chars / total_chars if total_chars > 0 else 1.0

        # Estimate compression ratio (empirical approximation)
        estimated_ratio = 0.3 + (char_ratio * 0.5)  # Range: 0.3-0.8

        return {
            "original_size_bytes": original_size,
            "estimated_compressed_size": int(original_size * estimated_ratio),
            "estimated_ratio": round(estimated_ratio, 3),
            "compression_recommended": original_size > MIN_SIZE_FOR_COMPRESSION and estimated_ratio < COMPRESSION_BENEFIT_THRESHOLD,
        }
    except Exception as e:
        return {"error": str(e)}


def get_pipeline_schema() -> Dict[str, Any]:
    """
    Return the schema for pipeline tool definitions.
    To be added to the OpenAI assistant's function schemas.
    """
    return {
        "decompress_pipeline_data": {
            "name": "decompress_pipeline_data",
            "description": "Decompress data that was compressed by the agent data pipeline. Use this when you receive compressed results and need to access the full data. You can decompress all data or specific chunks/items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "manifest_id": {
                        "type": "string",
                        "description": "The manifest_id from the compressed response"
                    },
                    "chunk_index": {
                        "type": "integer",
                        "description": "For chunked data, specify which chunk to decompress (0-indexed)"
                    },
                    "item_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific item indices to extract after decompression"
                    }
                },
                "required": ["manifest_id"]
            }
        },
        "search_pipeline_manifest": {
            "name": "search_pipeline_manifest",
            "description": "Search through compressed data manifest without decompressing. Use this to find relevant items before decompressing the full data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "manifest_id": {
                        "type": "string",
                        "description": "The manifest_id from the compressed response"
                    },
                    "search_term": {
                        "type": "string",
                        "description": "Text to search for in titles and previews"
                    },
                    "source_filter": {
                        "type": "string",
                        "description": "Filter results by source name"
                    },
                    "date_filter": {
                        "type": "string",
                        "description": "Filter results by date (partial match)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)"
                    }
                },
                "required": ["manifest_id"]
            }
        }
    }


# ──────────────────────────────────────────────────────────────────────
#  Pipeline Cache (for decompression lookups)
# ──────────────────────────────────────────────────────────────────────
_pipeline_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_SIZE = 50


def cache_pipeline_data(manifest_id: str, data: Dict[str, Any]) -> None:
    """Cache pipeline data for later decompression requests."""
    global _pipeline_cache

    # Simple LRU-like eviction
    if len(_pipeline_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(_pipeline_cache))
        del _pipeline_cache[oldest_key]

    _pipeline_cache[manifest_id] = data


def get_cached_pipeline_data(manifest_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached pipeline data by manifest ID."""
    return _pipeline_cache.get(manifest_id)


def clear_pipeline_cache() -> None:
    """Clear the pipeline cache."""
    global _pipeline_cache
    _pipeline_cache = {}


# ──────────────────────────────────────────────────────────────────────
#  Vector Store Upload Functions (v1.1 Hybrid Pipeline)
# ──────────────────────────────────────────────────────────────────────
def format_result_for_vector_store(
    item: Dict[str, Any],
    index: int,
    source: str,
    category: DataCategory,
) -> str:
    """
    Format a single result item as structured text for vector store.

    The format is optimized for semantic search:
    - Clear section headers
    - Key fields prominently placed
    - Metadata included for filtering context
    """
    extra = item.get("extra", {})

    # Build structured document
    lines = [
        f"=== RESULT {index + 1} ===",
        f"Source: {source}",
        f"Category: {category.value}",
        "",
    ]

    # Title (most important for search)
    title = item.get("title", "Untitled")
    lines.append(f"Title: {title}")

    # ID and URL
    item_id = item.get("id", item.get("pmid", item.get("nct_id", "")))
    if item_id:
        lines.append(f"ID: {item_id}")

    url = item.get("url", "")
    if url:
        lines.append(f"URL: {url}")

    # Date
    date = item.get("date", extra.get("date", extra.get("publication_date", "")))
    if date:
        lines.append(f"Date: {date}")

    lines.append("")

    # Abstract/Snippet (main content for semantic search)
    abstract = item.get("snippet", extra.get("abstract", ""))
    if abstract:
        lines.append("Content:")
        lines.append(abstract[:2000])  # Limit length
        lines.append("")

    # Category-specific fields
    if category == DataCategory.CLINICAL_TRIALS:
        for field in ["phase", "status", "enrollment", "sponsor", "conditions"]:
            value = extra.get(field, item.get(field))
            if value:
                lines.append(f"{field.title()}: {value}")

    elif category == DataCategory.SAFETY:
        for field in ["serious", "outcome", "reaction", "patient_age"]:
            value = extra.get(field, item.get(field))
            if value:
                lines.append(f"{field.title()}: {value}")

    elif category == DataCategory.REGULATORY:
        for field in ["decision_type", "approval_date", "indication", "application_number"]:
            value = extra.get(field, item.get(field))
            if value:
                lines.append(f"{field.title()}: {value}")

    elif category == DataCategory.KOL:
        for field in ["authors", "affiliation", "h_index", "citation_count"]:
            value = extra.get(field, item.get(field))
            if value:
                lines.append(f"{field.title()}: {value}")

    # Authority score if present
    if "_authority" in item:
        lines.append(f"Authority Score: {item['_authority']}")

    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def create_vector_store_document(
    results: List[Tuple[Dict, str]],
    query: str,
    category: DataCategory,
    sources_included: List[str],
    file_index: int = 0,
) -> str:
    """
    Create a structured document for vector store upload.

    Returns formatted text optimized for semantic search.
    """
    lines = [
        "=" * 60,
        "MEDICAL AFFAIRS SEARCH RESULTS",
        "=" * 60,
        "",
        f"Query: {query}",
        f"Category: {category.value}",
        f"Sources: {', '.join(sources_included)}",
        f"Total Results in File: {len(results)}",
        f"File Index: {file_index}",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        "=" * 60,
        "RESULTS",
        "=" * 60,
        "",
    ]

    for idx, (item, source) in enumerate(results):
        formatted = format_result_for_vector_store(item, idx, source, category)
        lines.append(formatted)

    return "\n".join(lines)


def upload_to_vector_store(
    data: Dict[str, Any],
    query: str,
    category: DataCategory,
    results: List[Tuple[Dict, str]],
    sources_included: List[str],
    vector_store_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload structured data to OpenAI vector store.

    Args:
        data: Original data dict
        query: Search query
        category: Data category
        results: List of (result, source) tuples
        sources_included: List of source names
        vector_store_id: Optional existing vector store ID to use

    Returns:
        Dict with file_ids, vector_store_id, and access instructions
    """
    file_ids = []
    total_results = len(results)

    try:
        # Split results into chunks for multiple files if needed
        chunks = []
        for i in range(0, total_results, VECTOR_FILE_RESULTS_PER_FILE):
            chunk = results[i:i + VECTOR_FILE_RESULTS_PER_FILE]
            chunks.append(chunk)

        _logger.info(f"Uploading {total_results} results in {len(chunks)} file(s) to vector store")

        # Create files and upload
        for file_idx, chunk in enumerate(chunks):
            # Create document content
            content = create_vector_store_document(
                results=chunk,
                query=query,
                category=category,
                sources_included=sources_included,
                file_index=file_idx,
            )

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                delete=False,
                encoding='utf-8',
            ) as f:
                f.write(content)
                temp_path = f.name

            try:
                # Upload to OpenAI
                with open(temp_path, 'rb') as f:
                    file_obj = openai.files.create(
                        file=f,
                        purpose="assistants",
                    )
                file_ids.append(file_obj.id)
                _logger.info(f"Uploaded file {file_idx + 1}/{len(chunks)}: {file_obj.id}")

            finally:
                # Clean up temp file
                os.unlink(temp_path)

        # Wait for files to be processed
        for file_id in file_ids:
            _wait_for_file_processing(file_id)

        # Create or use vector store
        if not vector_store_id:
            # Create a temporary vector store for this search
            vs = openai.beta.vector_stores.create(
                name=f"search_results_{generate_manifest_id(query, category)}",
                file_ids=file_ids,
            )
            vector_store_id = vs.id
            _logger.info(f"Created vector store: {vector_store_id}")
        else:
            # Add files to existing vector store
            for file_id in file_ids:
                openai.beta.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=file_id,
                )
            _logger.info(f"Added {len(file_ids)} files to existing vector store: {vector_store_id}")

        # Wait for vector store to be ready
        _wait_for_vector_store_ready(vector_store_id)

        return {
            "success": True,
            "file_ids": file_ids,
            "vector_store_id": vector_store_id,
            "total_results": total_results,
            "files_created": len(file_ids),
        }

    except Exception as e:
        _logger.error(f"Vector store upload failed: {e}")
        # Clean up any uploaded files on failure
        for file_id in file_ids:
            try:
                openai.files.delete(file_id)
            except Exception:
                pass
        return {
            "success": False,
            "error": str(e),
        }


def _wait_for_file_processing(file_id: str, timeout: int = VECTOR_STORE_UPLOAD_TIMEOUT) -> bool:
    """Wait for a file to be processed by OpenAI."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            file_info = openai.files.retrieve(file_id)
            if file_info.status == "processed":
                return True
            if file_info.status == "error":
                raise RuntimeError(f"File processing failed: {file_id}")
        except Exception as e:
            _logger.warning(f"Error checking file status: {e}")
        time.sleep(VECTOR_STORE_POLL_INTERVAL)
    raise TimeoutError(f"File processing timeout: {file_id}")


def _wait_for_vector_store_ready(vector_store_id: str, timeout: int = VECTOR_STORE_UPLOAD_TIMEOUT) -> bool:
    """Wait for vector store to be ready for queries."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            vs = openai.beta.vector_stores.retrieve(vector_store_id)
            if vs.status == "completed":
                return True
            if vs.status == "failed":
                raise RuntimeError(f"Vector store indexing failed: {vector_store_id}")
        except Exception as e:
            _logger.warning(f"Error checking vector store status: {e}")
        time.sleep(VECTOR_STORE_POLL_INTERVAL)
    # Don't fail on timeout - it might still work
    _logger.warning(f"Vector store ready check timed out, proceeding anyway: {vector_store_id}")
    return True


# ──────────────────────────────────────────────────────────────────────
#  Hybrid Pipeline (v1.1) - Unified Entry Point
# ──────────────────────────────────────────────────────────────────────
def select_pipeline_strategy(size_bytes: int) -> PipelineStrategy:
    """
    Select the appropriate pipeline strategy based on data size.

    Thresholds:
    - <150KB: DIRECT (return as-is)
    - 150KB-500KB: COMPRESSED (gzip + manifest)
    - >500KB: VECTOR_STORE (upload for file_search)
    """
    if size_bytes <= MAX_UNCOMPRESSED_OUTPUT:
        return PipelineStrategy.DIRECT
    elif size_bytes <= VECTOR_STORE_THRESHOLD:
        return PipelineStrategy.COMPRESSED
    else:
        return PipelineStrategy.VECTOR_STORE


def process_with_hybrid_pipeline(
    data: Dict[str, Any],
    query: str = "",
    vector_store_id: Optional[str] = None,
    force_strategy: Optional[PipelineStrategy] = None,
) -> Dict[str, Any]:
    """
    Process data through the hybrid pipeline.

    Automatically selects the best strategy based on data size:
    - Small (<150KB): Return directly with metadata
    - Medium (150-500KB): Compress with searchable manifest
    - Large (>500KB): Upload to vector store for semantic search

    Args:
        data: Raw data from source
        query: Original query string
        vector_store_id: Optional existing vector store to use
        force_strategy: Force a specific strategy (for testing)

    Returns:
        Processed data with strategy-specific access instructions
    """
    # Serialize to measure size
    try:
        serialized = json.dumps(data)
        original_size = len(serialized.encode('utf-8'))
    except (TypeError, ValueError) as e:
        _logger.error(f"Pipeline serialization failed: {e}")
        return {"error": "Pipeline serialization failed", "_pipeline_error": str(e)}

    # Detect category and extract metadata
    source = data.get("source", "unknown")
    category = detect_data_category(data, source)

    # Extract results
    results = []
    sources_included = []

    if "results_by_source" in data:
        for src, src_results in data.get("results_by_source", {}).items():
            if isinstance(src_results, list):
                sources_included.append(src)
                results.extend([(r, src) for r in src_results])
    elif "results" in data and isinstance(data["results"], list):
        sources_included = [source]
        results = [(r, source) for r in data["results"]]

    # Select strategy
    strategy = force_strategy or select_pipeline_strategy(original_size)
    _logger.info(f"Hybrid pipeline: {original_size} bytes -> strategy={strategy.value}")

    # Execute strategy
    if strategy == PipelineStrategy.DIRECT:
        return _add_pipeline_metadata(data, category, query, original_size, results)

    elif strategy == PipelineStrategy.COMPRESSED:
        return _compress_and_package(
            data=data,
            serialized=serialized,
            original_size=original_size,
            category=category,
            query=query,
            results=results,
            sources_included=sources_included,
        )

    elif strategy == PipelineStrategy.VECTOR_STORE:
        return _process_vector_store_strategy(
            data=data,
            query=query,
            category=category,
            results=results,
            sources_included=sources_included,
            original_size=original_size,
            vector_store_id=vector_store_id,
        )

    return {"error": f"Unknown strategy: {strategy}"}


def _process_vector_store_strategy(
    data: Dict[str, Any],
    query: str,
    category: DataCategory,
    results: List[Tuple[Dict, str]],
    sources_included: List[str],
    original_size: int,
    vector_store_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process data using vector store upload strategy.

    Returns a response with:
    - Summary manifest (for immediate context)
    - Vector store ID and file IDs (for file_search)
    - Instructions for the agent
    """
    # Upload to vector store
    upload_result = upload_to_vector_store(
        data=data,
        query=query,
        category=category,
        results=results,
        sources_included=sources_included,
        vector_store_id=vector_store_id,
    )

    if not upload_result.get("success"):
        # Fallback to compression if upload fails
        _logger.warning("Vector store upload failed, falling back to compression")
        serialized = json.dumps(data)
        return _compress_and_package(
            data=data,
            serialized=serialized,
            original_size=original_size,
            category=category,
            query=query,
            results=results,
            sources_included=sources_included,
        )

    # Create summary manifest for immediate context
    items_summary = [
        create_item_summary(item, idx, source)
        for idx, (item, source) in enumerate(results[:MAX_MANIFEST_ITEMS])
    ]

    manifest_id = generate_manifest_id(query, category)

    return {
        "_pipeline_version": "1.1",
        "_strategy": PipelineStrategy.VECTOR_STORE.value,
        "_vector_store_available": True,
        "manifest": {
            "manifest_id": manifest_id,
            "category": category.value,
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "total_items": len(results),
            "original_size_bytes": original_size,
            "compression_state": CompressionState.VECTOR_STORE.value,
            "items_summary": items_summary,
            "available_fields": extract_available_fields([r for r, _ in results[:10]]),
            "sources_included": sources_included,
            "intent_detected": data.get("intent_context", {}).get("detected_intent"),
        },
        "vector_store": {
            "vector_store_id": upload_result["vector_store_id"],
            "file_ids": upload_result["file_ids"],
            "files_created": upload_result["files_created"],
            "total_results_indexed": upload_result["total_results"],
        },
        "access_instructions": (
            f"Data contains {len(results)} results uploaded to vector store for semantic search. "
            f"Use file_search with vector_store_id '{upload_result['vector_store_id']}' to query the data. "
            "The manifest.items_summary contains previews of all results for quick reference. "
            "For specific items, search the vector store with relevant terms from your query."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
#  Vector Store Cache (for cleanup tracking)
# ──────────────────────────────────────────────────────────────────────
_vector_store_cache: Dict[str, Dict[str, Any]] = {}


def cache_vector_store_info(manifest_id: str, info: Dict[str, Any]) -> None:
    """Cache vector store info for later cleanup."""
    global _vector_store_cache
    _vector_store_cache[manifest_id] = info


def get_cached_vector_store_info(manifest_id: str) -> Optional[Dict[str, Any]]:
    """Get cached vector store info."""
    return _vector_store_cache.get(manifest_id)


def cleanup_vector_store(manifest_id: str) -> bool:
    """
    Clean up vector store resources for a manifest.

    Call this when the data is no longer needed to avoid storage costs.
    """
    info = _vector_store_cache.pop(manifest_id, None)
    if not info:
        return False

    try:
        # Delete files
        for file_id in info.get("file_ids", []):
            try:
                openai.files.delete(file_id)
                _logger.info(f"Deleted file: {file_id}")
            except Exception as e:
                _logger.warning(f"Failed to delete file {file_id}: {e}")

        # Delete vector store if we created it
        vs_id = info.get("vector_store_id")
        if vs_id:
            try:
                openai.beta.vector_stores.delete(vs_id)
                _logger.info(f"Deleted vector store: {vs_id}")
            except Exception as e:
                _logger.warning(f"Failed to delete vector store {vs_id}: {e}")

        return True

    except Exception as e:
        _logger.error(f"Vector store cleanup failed: {e}")
        return False
