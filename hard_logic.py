"""
hard_logic.py
In-memory Hard Logic layer — loads all JSON config files into pandas
DataFrames at session start so the LLM operates on deterministic,
complete, instantly-queryable structured data.

Architecture:
    HARD LOGIC  = This module (pandas DataFrames in memory)
                  Deterministic, complete, queryable.
    SOFT LOGIC  = GPT-5.2 (strategy, synthesis, judgment)
                  Uses query_hard_logic tool to access DataFrames.
    file_search = Reserved for PDFs, proprietary docs, and free-text
                  reference files — NOT for JSON configs.

Loading order:
    1. Local files in data/ directory (fast, no API dependency)
    2. OpenAI Files API download (fallback)

Usage:
    from hard_logic import get_store

    store = get_store()           # singleton, lazy-loads on first call
    store.load_all(client)        # load from local files or download
    result = store.query("pillars", operation="list_all")
    schema = store.get_schema_summary()
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Local data directory — JSON files here are loaded first (no API call)
# ──────────────────────────────────────────────────────────────────────
LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data"

# ──────────────────────────────────────────────────────────────────────
#  File Registry — JSON configs with known OpenAI file IDs
#  These match the IDs in system_instructions.txt
# ──────────────────────────────────────────────────────────────────────
FILE_REGISTRY: Dict[str, Dict[str, str]] = {
    "metrics": {
        "file_id": "file-2yuMdjM9UmEdauNs2iHK5U",
        "filename": "metrics_v2.json",
        "description": "KPI definitions, measurement frameworks, thresholds",
    },
    "pillars": {
        "file_id": "file-3daCejyHWifvw5pphnM7cf",
        "filename": "pillars_v2.json",
        "description": "Strategic pillar definitions, success criteria, interdependencies",
    },
    "stakeholders": {
        "file_id": "file-5WWFcuwtnMqXSdqHguNuk3",
        "filename": "stakeholder_taxonomy_v2.json",
        "description": "Stakeholder roles, tiers, influence/interest scoring, engagement pathways",
    },
    "tactics": {
        "file_id": "file-3k37yooFLqYU1X1rus23NX",
        "filename": "tactics_taxonomy_v2.json",
        "description": "Tactical actions organised by pillar, resource requirements, timelines",
    },
    "roles": {
        "file_id": "file-7DywzaY7irPQMCGFHjCbgY",
        "filename": "role_taxonomy_v1.json",
        "description": "User role definitions and perspective filters (12 roles)",
    },
    "data_sources": {
        "file_id": "file-32k1NdwdTy7sY9bA9gMDCH",
        "filename": "Data_sources.json",
        "description": "Authoritative data source specs, authority ranking, validation rules",
    },
    "vr": {
        "file_id": "file-DJLsX7SAP4bASK4MKJa6Mz",
        "filename": "saimone_vr.json",
        "description": "Value realisation model, system-wide parameters",
    },
    "kol": {
        "file_id": "file-14AqzUAqpcW1GHYKygGBda",
        "filename": "kol.json",
        "description": "Key Opinion Leaders database — specialties, influence, publications",
    },
    "auth_sources": {
        "file_id": "file-WqiPeqkxpaFrPduqW1QX4Y",
        "filename": "authoritative_sources_speyside_starter_v1_1.json",
        "description": "Source credibility rankings, primary/secondary classifications",
    },
}

# Datasets that the LLM can request via query_hard_logic
QUERYABLE_DATASETS = list(FILE_REGISTRY.keys())


# ──────────────────────────────────────────────────────────────────────
#  HardLogicStore — singleton that holds all DataFrames in memory
# ──────────────────────────────────────────────────────────────────────
class HardLogicStore:
    """Downloads JSON configs from OpenAI Files API and holds them as
    pandas DataFrames for the duration of the session.

    Thread-safe for reads after ``load_all()`` completes.
    """

    _instance: Optional["HardLogicStore"] = None

    def __init__(self) -> None:
        self._raw: Dict[str, Any] = {}               # original parsed JSON
        self._frames: Dict[str, pd.DataFrame] = {}   # normalised DataFrames
        self._load_errors: Dict[str, str] = {}        # per-dataset errors
        self._loaded: bool = False
        self._load_time: float = 0.0

    # ── singleton access ────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "HardLogicStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (useful in tests)."""
        cls._instance = None

    # ── bulk loader ─────────────────────────────────────────────────
    def load_all(self, client) -> Dict[str, str]:
        """Load every JSON in FILE_REGISTRY into DataFrames.

        Loading order per dataset:
            1. Local file in data/<filename> (fast, no API call)
            2. OpenAI Files API download (fallback)

        If an OpenAI download succeeds, the file is saved locally so
        subsequent sessions load instantly without an API call.

        Args:
            client: An initialised ``openai.OpenAI`` instance.

        Returns:
            Dict mapping dataset key → status string ("ok" or error msg).
        """
        status: Dict[str, str] = {}
        t0 = time.time()

        for key, meta in FILE_REGISTRY.items():
            try:
                text = self._load_local(meta["filename"])
                source = "local"

                if text is None:
                    text = self._download_from_openai(client, meta)
                    source = "openai"
                    # Cache locally for future sessions
                    if text is not None:
                        self._save_local(meta["filename"], text)

                if text is None:
                    raise RuntimeError(
                        f"Could not load {meta['filename']} from local data/ "
                        f"directory or OpenAI Files API. Place the JSON file in "
                        f"{LOCAL_DATA_DIR / meta['filename']}"
                    )

                parsed = json.loads(text)
                self._raw[key] = parsed
                self._frames[key] = _to_dataframe(key, parsed)
                status[key] = "ok"
                _logger.info(
                    "Loaded %s (%s) from %s → %d rows",
                    key,
                    meta["filename"],
                    source,
                    len(self._frames[key]),
                )
            except Exception as exc:
                self._load_errors[key] = str(exc)
                status[key] = f"error: {exc}"
                _logger.warning("Failed to load %s: %s", key, exc)

        self._loaded = True
        self._load_time = time.time() - t0
        _logger.info(
            "Hard logic loaded: %d/%d datasets in %.1fs",
            sum(1 for v in status.values() if v == "ok"),
            len(FILE_REGISTRY),
            self._load_time,
        )
        return status

    # ── local file I/O ───────────────────────────────────────────────
    @staticmethod
    def _load_local(filename: str) -> Optional[str]:
        """Try to read a JSON file from the local data/ directory."""
        path = LOCAL_DATA_DIR / filename
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                _logger.info("Found local file: %s", path)
                return text
            except Exception as exc:
                _logger.warning("Could not read local file %s: %s", path, exc)
        return None

    @staticmethod
    def _save_local(filename: str, text: str) -> None:
        """Cache a downloaded JSON file locally for future sessions."""
        try:
            LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = LOCAL_DATA_DIR / filename
            path.write_text(text, encoding="utf-8")
            _logger.info("Cached locally: %s", path)
        except Exception as exc:
            _logger.warning("Could not cache %s locally: %s", filename, exc)

    @staticmethod
    def _download_from_openai(client, meta: Dict[str, str]) -> Optional[str]:
        """Download a file from the OpenAI Files API.

        Returns the file text, or None if the download fails.
        Handles the 'purpose=assistants' restriction gracefully.
        """
        try:
            raw_bytes = client.files.content(meta["file_id"])
            return raw_bytes.read().decode("utf-8")
        except Exception as exc:
            err_str = str(exc)
            if "Not allowed to download" in err_str and "purpose" in err_str:
                _logger.warning(
                    "Cannot download %s (file_id=%s): file was uploaded with "
                    "purpose='assistants' which blocks direct downloads. "
                    "Place the JSON file in %s instead.",
                    meta["filename"],
                    meta["file_id"],
                    LOCAL_DATA_DIR / meta["filename"],
                )
            else:
                _logger.warning(
                    "OpenAI download failed for %s: %s",
                    meta["filename"],
                    exc,
                )
            return None

    # ── properties ──────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def available_datasets(self) -> List[str]:
        return list(self._frames.keys())

    @property
    def load_errors(self) -> Dict[str, str]:
        return dict(self._load_errors)

    # ── raw JSON access ─────────────────────────────────────────────
    def get_raw(self, dataset: str) -> Optional[Any]:
        """Return the original parsed JSON for *dataset*."""
        return self._raw.get(dataset)

    # ── DataFrame access ────────────────────────────────────────────
    def get_frame(self, dataset: str) -> Optional[pd.DataFrame]:
        """Return the normalised DataFrame for *dataset*."""
        return self._frames.get(dataset)

    # ── query interface (called by the tool router) ─────────────────
    def query(
        self,
        dataset: str,
        operation: str = "list_all",
        filter_column: Optional[str] = None,
        filter_value: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: int = 50,
        search_term: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a structured query against a loaded DataFrame.

        Operations
        ----------
        schema        – column names, dtypes, row count
        list_all      – return all rows (up to *limit*)
        filter        – rows where *filter_column* contains *filter_value*
        lookup        – exact-match lookup on *filter_column*
        search        – full-text search across all string columns
        describe      – pandas .describe() statistical summary
        cross_ref     – return rows from *dataset* that reference *filter_value*
                        in any column (useful for pillar↔tactic mapping)

        Returns a JSON-serialisable dict.
        """
        if dataset not in self._frames:
            available = ", ".join(self._frames.keys()) or "(none loaded)"
            return {
                "error": f"Dataset '{dataset}' not available. Loaded: {available}",
            }

        df = self._frames[dataset]

        try:
            if operation == "schema":
                return _op_schema(df, dataset)

            if operation == "describe":
                return _op_describe(df, dataset)

            if operation == "list_all":
                return _op_list(df, columns, limit, dataset)

            if operation == "filter":
                if not filter_column or filter_value is None:
                    return {"error": "filter requires filter_column and filter_value"}
                return _op_filter(df, filter_column, filter_value, columns, limit, dataset)

            if operation == "lookup":
                if not filter_column or filter_value is None:
                    return {"error": "lookup requires filter_column and filter_value"}
                return _op_lookup(df, filter_column, filter_value, columns, dataset)

            if operation == "search":
                if not search_term:
                    return {"error": "search requires search_term"}
                return _op_search(df, search_term, columns, limit, dataset)

            if operation == "cross_ref":
                if filter_value is None:
                    return {"error": "cross_ref requires filter_value"}
                return _op_cross_ref(df, filter_value, columns, limit, dataset)

            return {"error": f"Unknown operation '{operation}'"}

        except Exception as exc:
            _logger.error("Query error [%s/%s]: %s", dataset, operation, exc)
            return {"error": str(exc)}

    # ── schema summary for prompt injection ─────────────────────────
    def get_schema_summary(self) -> str:
        """Return a compact text summary of all loaded DataFrames.

        Injected into the system prompt so the LLM knows what Hard Logic
        data is available and how to query it.
        """
        if not self._frames:
            return "(No Hard Logic datasets loaded)"

        lines: List[str] = [
            "HARD LOGIC DATASETS IN MEMORY (query via query_hard_logic tool):",
            "",
        ]
        for key in sorted(self._frames.keys()):
            df = self._frames[key]
            meta = FILE_REGISTRY.get(key, {})
            desc = meta.get("description", "")
            cols = ", ".join(df.columns[:12].tolist())
            if len(df.columns) > 12:
                cols += f", ... (+{len(df.columns) - 12} more)"
            lines.append(f"  {key} ({len(df)} rows) — {desc}")
            lines.append(f"    columns: [{cols}]")

        lines.append("")
        lines.append(
            "Use the query_hard_logic tool to query these datasets. "
            "Operations: schema, list_all, filter, lookup, search, "
            "cross_ref, describe."
        )
        return "\n".join(lines)

    # ── load status summary ─────────────────────────────────────────
    def get_load_status(self) -> Dict[str, Any]:
        """Return a summary of load results for debugging/UI."""
        return {
            "loaded": self._loaded,
            "load_time_s": round(self._load_time, 2),
            "datasets_ok": [k for k in self._frames],
            "datasets_failed": dict(self._load_errors),
            "total_rows": sum(len(df) for df in self._frames.values()),
        }


# ──────────────────────────────────────────────────────────────────────
#  Module-level convenience
# ──────────────────────────────────────────────────────────────────────
def get_store() -> HardLogicStore:
    """Return the singleton HardLogicStore."""
    return HardLogicStore.get_instance()


# ──────────────────────────────────────────────────────────────────────
#  JSON → DataFrame conversion
# ──────────────────────────────────────────────────────────────────────
def _to_dataframe(key: str, raw: Any) -> pd.DataFrame:
    """Convert parsed JSON into a flat(ish) pandas DataFrame.

    Handles the common JSON shapes:
    - list of objects  →  json_normalize directly
    - dict with a main data list  →  find the largest list value
    - dict of dicts  →  orient="index"
    - single object  →  single-row DataFrame
    """
    if isinstance(raw, list):
        if len(raw) == 0:
            return pd.DataFrame()
        if isinstance(raw[0], dict):
            return pd.json_normalize(raw, max_level=2)
        return pd.DataFrame(raw, columns=["value"])

    if isinstance(raw, dict):
        # Check for a top-level list (common: {"pillars": [...], "version": ...})
        list_candidates = {
            k: v for k, v in raw.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0] if v else None, dict)
        }
        if list_candidates:
            # Pick the largest list
            main_key = max(list_candidates, key=lambda k: len(list_candidates[k]))
            return pd.json_normalize(list_candidates[main_key], max_level=2)

        # Dict-of-dicts (e.g. {"role_a": {fields}, "role_b": {fields}})
        if all(isinstance(v, dict) for v in raw.values()):
            df = pd.DataFrame.from_dict(raw, orient="index")
            df.index.name = "key"
            return df.reset_index()

        # Single flat object → one-row DataFrame
        return pd.json_normalize(raw, max_level=2)

    # Scalar fallback
    return pd.DataFrame([{"value": raw}])


# ──────────────────────────────────────────────────────────────────────
#  Query operation helpers
# ──────────────────────────────────────────────────────────────────────
_MAX_CELL_LEN = 500  # truncate long cell values in output


def _truncate_cell(val: Any) -> Any:
    if isinstance(val, str) and len(val) > _MAX_CELL_LEN:
        return val[:_MAX_CELL_LEN] + "..."
    return val


def _df_to_records(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   limit: int = 50) -> List[dict]:
    """Convert a (possibly filtered) DataFrame to a list of dicts."""
    if columns:
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            df = df[valid_cols]
    records = df.head(limit).to_dict(orient="records")
    return [
        {k: _truncate_cell(v) for k, v in rec.items()}
        for rec in records
    ]


def _op_schema(df: pd.DataFrame, dataset: str) -> dict:
    return {
        "dataset": dataset,
        "rows": len(df),
        "columns": {
            col: str(df[col].dtype)
            for col in df.columns
        },
        "sample": _df_to_records(df, limit=3),
    }


def _op_describe(df: pd.DataFrame, dataset: str) -> dict:
    desc = df.describe(include="all").to_dict()
    return {"dataset": dataset, "rows": len(df), "statistics": desc}


def _op_list(df: pd.DataFrame, columns: Optional[List[str]],
             limit: int, dataset: str) -> dict:
    records = _df_to_records(df, columns, limit)
    return {
        "dataset": dataset,
        "total_rows": len(df),
        "returned": len(records),
        "records": records,
    }


def _op_filter(df: pd.DataFrame, col: str, val: str,
               columns: Optional[List[str]], limit: int,
               dataset: str) -> dict:
    if col not in df.columns:
        return {"error": f"Column '{col}' not in {dataset}. Available: {list(df.columns)}"}
    mask = df[col].astype(str).str.contains(val, case=False, na=False)
    filtered = df[mask]
    records = _df_to_records(filtered, columns, limit)
    return {
        "dataset": dataset,
        "filter": f"{col} contains '{val}'",
        "total_matches": len(filtered),
        "returned": len(records),
        "records": records,
    }


def _op_lookup(df: pd.DataFrame, col: str, val: str,
               columns: Optional[List[str]], dataset: str) -> dict:
    if col not in df.columns:
        return {"error": f"Column '{col}' not in {dataset}. Available: {list(df.columns)}"}
    mask = df[col].astype(str).str.lower() == val.lower()
    matched = df[mask]
    records = _df_to_records(matched, columns, limit=10)
    return {
        "dataset": dataset,
        "lookup": f"{col} == '{val}'",
        "matches": len(matched),
        "records": records,
    }


def _op_search(df: pd.DataFrame, term: str,
               columns: Optional[List[str]], limit: int,
               dataset: str) -> dict:
    """Full-text search across all string columns."""
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) == 0:
        return {"dataset": dataset, "matches": 0, "records": []}
    mask = pd.Series(False, index=df.index)
    for c in str_cols:
        mask |= df[c].astype(str).str.contains(term, case=False, na=False)
    filtered = df[mask]
    records = _df_to_records(filtered, columns, limit)
    return {
        "dataset": dataset,
        "search_term": term,
        "total_matches": len(filtered),
        "returned": len(records),
        "records": records,
    }


def _op_cross_ref(df: pd.DataFrame, val: str,
                  columns: Optional[List[str]], limit: int,
                  dataset: str) -> dict:
    """Find rows that reference *val* in any column (pillar↔tactic mapping etc.)."""
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        mask |= df[c].astype(str).str.contains(val, case=False, na=False)
    filtered = df[mask]
    records = _df_to_records(filtered, columns, limit)
    return {
        "dataset": dataset,
        "cross_ref": val,
        "total_matches": len(filtered),
        "returned": len(records),
        "records": records,
    }


# ──────────────────────────────────────────────────────────────────────
#  Tool definition for Responses API
# ──────────────────────────────────────────────────────────────────────
QUERY_HARD_LOGIC_TOOL: dict = {
    "type": "function",
    "name": "query_hard_logic",
    "description": (
        "Query the in-memory Hard Logic DataFrames that hold all JSON "
        "config data (pillars, metrics, tactics, stakeholders, roles, "
        "KOLs, data sources, authoritative sources, value realisation). "
        "This is faster and more reliable than file_search for structured "
        "config data. Use file_search only for PDFs and free-text docs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "description": "Which dataset to query.",
                "enum": QUERYABLE_DATASETS,
            },
            "operation": {
                "type": "string",
                "description": (
                    "Query operation: "
                    "schema (column names + dtypes + sample), "
                    "list_all (return rows up to limit), "
                    "filter (rows where column contains value), "
                    "lookup (exact-match on column), "
                    "search (full-text across all string columns), "
                    "cross_ref (rows referencing a value in any column), "
                    "describe (statistical summary)."
                ),
                "enum": [
                    "schema", "list_all", "filter", "lookup",
                    "search", "cross_ref", "describe",
                ],
            },
            "filter_column": {
                "type": "string",
                "description": "Column to filter/lookup on (for filter, lookup ops).",
            },
            "filter_value": {
                "type": "string",
                "description": "Value to match (for filter, lookup, cross_ref ops).",
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subset of columns to return (optional, returns all if omitted).",
            },
            "search_term": {
                "type": "string",
                "description": "Text to search for (for search op).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return. Default: 50.",
                "default": 50,
            },
        },
        "required": ["dataset", "operation"],
    },
}


def handle_query_hard_logic(args: Dict[str, Any]) -> str:
    """Route a query_hard_logic tool call to the HardLogicStore.

    Returns a JSON string for the Responses API tool output.
    """
    store = get_store()
    if not store.is_loaded:
        return json.dumps({"error": "Hard Logic not loaded yet. Session initialisation may still be in progress."})

    result = store.query(
        dataset=args.get("dataset", ""),
        operation=args.get("operation", "list_all"),
        filter_column=args.get("filter_column"),
        filter_value=args.get("filter_value"),
        columns=args.get("columns"),
        limit=args.get("limit", 50),
        search_term=args.get("search_term"),
    )
    return json.dumps(result, default=str)
