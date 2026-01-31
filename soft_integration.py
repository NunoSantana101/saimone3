"""
soft_integration.py - Minimalist URI Provider for Medical APIs
──────────────────────────────────────────────────────────────────────────────
"Link-First" architecture: instead of pulling full abstracts/labels into
context (500-1000 tokens each), this layer returns clean Markdown-formatted
links with minimal metadata (~50 tokens per result).

The model can then use its native web browsing tools (web_search_preview,
open_page) to deep-dive into specific sources when needed.

Supported sources:
  - PubMed   : ESearch → ESummary → markdown links
  - FDA      : openFDA search → DailyMed / Drugs@FDA links
  - EMA      : EMA medicines dataset → product page links

Comparison:
  Abstract-First: 500-1000 tokens/result, ~10-20 papers per context window
  Link-First:     ~50 tokens/result, 100+ links in one go

Usage:
  from soft_integration import search_medical_links
  result = search_medical_links("pubmed", "SGLT2 inhibitors heart failure")
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENFDA_BASE = "https://api.fda.gov"
EMA_MEDICINES_URL = "https://www.ema.europa.eu/en/medicines/field_ema_web_categories%253Aname_field/Human/ema_group_types/ema_medicine"

HEADERS = {"User-Agent": "saimone-soft-integration/1.0"}
REQUEST_TIMEOUT = 15  # seconds per API call
MAX_RESULTS_CAP = 100  # hard cap per source


def _secret(key: str) -> Optional[str]:
    """Get secret from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            return st.secrets.get(key)
    except Exception:
        pass
    for env_key in (key, f"AZURE_{key}", f"APPSETTING_{key}"):
        if (value := os.getenv(env_key)):
            return value
    return None


NCBI_API_KEY = _secret("NCBI_API_KEY")
OPENFDA_API_KEY = _secret("OPENFDA_API_KEY")

# Rate limit: 3 req/s without key, 10 req/s with key
NCBI_RATE_LIMIT = 0.1 if NCBI_API_KEY else 0.34


# ────────────────────────────────────────────────────────────────
# PubMed Link Provider
# ────────────────────────────────────────────────────────────────

def _pubmed_links(
    query: str,
    max_results: int = 20,
    date_range: Optional[str] = None,
    sort: str = "relevance",
) -> Dict[str, Any]:
    """
    PubMed ESearch → ESummary → Markdown links.

    Returns structured result with markdown-formatted links and minimal
    metadata (title, journal, year, PMID). Full-text access URLs are
    included when PMC versions are available.
    """
    max_results = min(max_results, MAX_RESULTS_CAP)

    # ── Step 1: ESearch to get PMIDs ──
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": sort,
        "usehistory": "n",
    }
    if NCBI_API_KEY:
        esearch_params["api_key"] = NCBI_API_KEY
    if date_range:
        match = re.match(r"(\d{4}(?:/\d{2}(?:/\d{2})?)?)\s*[-–]\s*(\d{4}(?:/\d{2}(?:/\d{2})?)?)", date_range)
        if match:
            esearch_params["mindate"] = match.group(1).replace("-", "/")
            esearch_params["maxdate"] = match.group(2).replace("-", "/")
            esearch_params["datetype"] = "pdat"

    try:
        resp = requests.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params=esearch_params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        esearch_data = resp.json()
    except Exception as exc:
        logger.error(f"PubMed ESearch failed: {exc}")
        return {"source": "pubmed", "error": str(exc), "links": [], "markdown": ""}

    pmids = esearch_data.get("esearchresult", {}).get("idlist", [])
    total_found = int(esearch_data.get("esearchresult", {}).get("count", 0))

    if not pmids:
        return {
            "source": "pubmed",
            "query": query,
            "total_found": total_found,
            "links": [],
            "markdown": f"No PubMed results found for: {query}",
        }

    # ── Step 2: ESummary to get titles and metadata ──
    time.sleep(NCBI_RATE_LIMIT)  # respect rate limit

    esummary_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    if NCBI_API_KEY:
        esummary_params["api_key"] = NCBI_API_KEY

    try:
        resp = requests.get(
            f"{NCBI_BASE}/esummary.fcgi",
            params=esummary_params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        summary_data = resp.json().get("result", {})
    except Exception as exc:
        logger.error(f"PubMed ESummary failed: {exc}")
        # Fallback: return links without titles
        links = []
        for pmid in pmids:
            links.append({
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "markdown": f"- [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)",
            })
        return {
            "source": "pubmed",
            "query": query,
            "total_found": total_found,
            "returned": len(links),
            "links": links,
            "markdown": "\n".join(l["markdown"] for l in links),
        }

    # ── Step 3: Format as markdown links ──
    links = []
    for pmid in pmids:
        info = summary_data.get(pmid, {})
        if not isinstance(info, dict):
            continue

        title = html.unescape(info.get("title", f"PMID {pmid}")).rstrip(".")
        journal = info.get("fulljournalname") or info.get("source", "")
        pub_date = info.get("pubdate", "")
        year = pub_date[:4] if pub_date else ""

        # Check for PMC full-text availability
        pmc_id = ""
        article_ids = info.get("articleids", [])
        for aid in article_ids:
            if isinstance(aid, dict) and aid.get("idtype") == "pmc":
                pmc_id = aid.get("value", "")
                break

        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/" if pmc_id else None

        # Build markdown line
        md_parts = [f"- [{title}]({pubmed_url})"]
        if journal and year:
            md_parts.append(f"  *{journal}* ({year})")
        elif journal:
            md_parts.append(f"  *{journal}*")
        elif year:
            md_parts.append(f"  ({year})")
        if pmc_url:
            md_parts.append(f"  | [Full text (PMC)]({pmc_url})")

        link_entry = {
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year,
            "url": pubmed_url,
            "pmc_url": pmc_url,
            "markdown": " ".join(md_parts),
        }
        links.append(link_entry)

    markdown_output = f"### PubMed Results ({len(links)} of {total_found:,} total)\n\n"
    markdown_output += "\n".join(l["markdown"] for l in links)

    return {
        "source": "pubmed",
        "query": query,
        "total_found": total_found,
        "returned": len(links),
        "links": links,
        "markdown": markdown_output,
    }


# ────────────────────────────────────────────────────────────────
# FDA Link Provider
# ────────────────────────────────────────────────────────────────

def _fda_links(
    query: str,
    max_results: int = 20,
    collection: Optional[str] = None,
    date_range: Optional[str] = None,
) -> Dict[str, Any]:
    """
    FDA openFDA search → Markdown links to DailyMed and Drugs@FDA pages.

    Searches drug labels by brand_name, generic_name, and substance_name.
    Returns direct links to:
      - DailyMed (full prescribing information)
      - Drugs@FDA (approval history)
      - openFDA label page
    """
    max_results = min(max_results, MAX_RESULTS_CAP)
    collection = collection or "drug/label"

    # Build openFDA search query
    safe_query = query.replace('"', '\\"')
    search_clauses = [
        f'openfda.brand_name:"{safe_query}"',
        f'openfda.generic_name:"{safe_query}"',
        f'openfda.substance_name:"{safe_query}"',
    ]
    search_string = "+".join(search_clauses)

    params = {
        "search": search_string,
        "limit": max_results,
    }
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY

    url = f"{OPENFDA_BASE}/{collection}.json"

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return {
                "source": "fda",
                "query": query,
                "total_found": 0,
                "links": [],
                "markdown": f"No FDA results found for: {query}",
            }
        logger.error(f"FDA API error: {exc}")
        return {"source": "fda", "error": str(exc), "links": [], "markdown": ""}
    except Exception as exc:
        logger.error(f"FDA API error: {exc}")
        return {"source": "fda", "error": str(exc), "links": [], "markdown": ""}

    results = data.get("results", [])
    total_found = data.get("meta", {}).get("results", {}).get("total", len(results))

    links = []
    seen_apps = set()  # deduplicate by application number

    for item in results:
        openfda = item.get("openfda", {})

        brand_names = openfda.get("brand_name", [])
        generic_names = openfda.get("generic_name", [])
        substance_names = openfda.get("substance_name", [])
        app_numbers = openfda.get("application_number", [])
        manufacturers = openfda.get("manufacturer_name", [])

        brand = brand_names[0] if brand_names else ""
        generic = generic_names[0] if generic_names else ""
        substance = substance_names[0] if substance_names else ""
        app_number = app_numbers[0] if app_numbers else ""
        manufacturer = manufacturers[0] if manufacturers else ""

        # Deduplicate
        dedup_key = app_number or f"{brand}_{generic}"
        if dedup_key in seen_apps:
            continue
        seen_apps.add(dedup_key)

        # Build display title
        if brand and generic:
            title = f"{brand} ({generic})"
        elif brand:
            title = brand
        elif generic:
            title = generic
        elif substance:
            title = substance
        else:
            title = f"FDA Label {app_number}"

        # Build URLs
        dailymed_url = None
        drugsfda_url = None
        label_set_id = item.get("set_id")

        if label_set_id:
            dailymed_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={label_set_id}"

        if app_number:
            # Clean app_number for Drugs@FDA URL (remove prefix like NDA/ANDA)
            app_num_clean = re.sub(r"^[A-Z]+", "", app_number)
            drugsfda_url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={app_num_clean}"

        # Extract effective date
        effective_date = item.get("effective_time", "")
        if effective_date and len(effective_date) >= 8:
            formatted_date = f"{effective_date[:4]}-{effective_date[4:6]}-{effective_date[6:8]}"
        else:
            formatted_date = ""

        # Build markdown
        md_parts = []
        if dailymed_url:
            md_parts.append(f"- [{title}]({dailymed_url})")
        elif drugsfda_url:
            md_parts.append(f"- [{title}]({drugsfda_url})")
        else:
            md_parts.append(f"- **{title}**")

        meta_parts = []
        if manufacturer:
            meta_parts.append(manufacturer)
        if app_number:
            meta_parts.append(app_number)
        if formatted_date:
            meta_parts.append(formatted_date)
        if meta_parts:
            md_parts.append(f"  ({' | '.join(meta_parts)})")

        # Add secondary link
        if dailymed_url and drugsfda_url:
            md_parts.append(f"  | [Drugs@FDA]({drugsfda_url})")

        link_entry = {
            "title": title,
            "app_number": app_number,
            "manufacturer": manufacturer,
            "date": formatted_date,
            "dailymed_url": dailymed_url,
            "drugsfda_url": drugsfda_url,
            "url": dailymed_url or drugsfda_url,
            "markdown": " ".join(md_parts),
        }
        links.append(link_entry)

    markdown_output = f"### FDA Results ({len(links)} of {total_found:,} total)\n\n"
    markdown_output += "\n".join(l["markdown"] for l in links)

    return {
        "source": "fda",
        "query": query,
        "total_found": total_found,
        "returned": len(links),
        "links": links,
        "markdown": markdown_output,
    }


# ────────────────────────────────────────────────────────────────
# EMA Link Provider
# ────────────────────────────────────────────────────────────────

# EMA public medicines JSON endpoint (no auth required)
EMA_JSON_URL = "https://www.ema.europa.eu/en/medicines/download-medicine-data"
EMA_API_URL = "https://www.ema.europa.eu/en/medicines/field_ema_web_categories%253Aname_field/Human"

def _ema_links(
    query: str,
    max_results: int = 20,
    date_range: Optional[str] = None,
) -> Dict[str, Any]:
    """
    EMA medicines search → Markdown links to EMA product pages.

    Uses the EMA public medicines dataset. Returns direct links to:
      - EMA medicine overview page
      - EPAR (European Public Assessment Report)
      - Product information (SmPC)
    """
    max_results = min(max_results, MAX_RESULTS_CAP)
    query_lower = query.lower().strip()
    query_terms = query_lower.split()

    # Try the EMA medicines API-style search
    # EMA doesn't have a public REST API, so we search their JSON dataset
    # which is cached in med_affairs_data.py. We'll try to import it.
    try:
        from med_affairs_data import search_ema as _search_ema_full
        results_raw, _ = _search_ema_full(
            query=query,
            max_results=max_results,
            prioritize_recent=True,
        )
    except Exception as exc:
        logger.warning(f"EMA full search unavailable, using fallback: {exc}")
        results_raw = []

    if results_raw:
        # Convert full results to link-only format
        links = []
        for item in results_raw[:max_results]:
            title = item.get("title", "Unknown Medicine")
            ema_url = item.get("url", "")
            extra = item.get("extra", {})

            # Build EMA product page URL if not present
            if not ema_url:
                product_number = extra.get("ema_product_number", "")
                medicine_name = title.split(" — ")[0].strip() if " — " in title else title
                slug = re.sub(r"[^a-z0-9]+", "-", medicine_name.lower()).strip("-")
                ema_url = f"https://www.ema.europa.eu/en/medicines/human/EPAR/{slug}"

            date = item.get("date", "")
            active_substance = extra.get("active_substance", "")
            mah = extra.get("marketing_authorisation_holder", "")
            status = extra.get("medicine_status", "")

            # Build markdown
            md_parts = [f"- [{title}]({ema_url})"]
            meta_parts = []
            if mah:
                meta_parts.append(mah)
            if status:
                meta_parts.append(status)
            if date:
                meta_parts.append(date)
            if meta_parts:
                md_parts.append(f"  ({' | '.join(meta_parts)})")

            link_entry = {
                "title": title,
                "active_substance": active_substance,
                "mah": mah,
                "status": status,
                "date": date,
                "url": ema_url,
                "markdown": " ".join(md_parts),
            }
            links.append(link_entry)

        markdown_output = f"### EMA Results ({len(links)} found)\n\n"
        markdown_output += "\n".join(l["markdown"] for l in links)

        return {
            "source": "ema",
            "query": query,
            "total_found": len(links),
            "returned": len(links),
            "links": links,
            "markdown": markdown_output,
        }

    # Fallback: construct search URL for the model to browse
    ema_search_url = (
        f"https://www.ema.europa.eu/en/medicines/search_api_aggregation_ema_medicine_types/"
        f"field_ema_med_type/human_use?search_api_views_fulltext={quote_plus(query)}"
    )
    return {
        "source": "ema",
        "query": query,
        "total_found": 0,
        "returned": 0,
        "links": [{
            "title": f"EMA Search: {query}",
            "url": ema_search_url,
            "markdown": f"- [Search EMA for: {query}]({ema_search_url})",
        }],
        "markdown": (
            f"### EMA Results\n\n"
            f"No cached results. Use the search link to browse:\n"
            f"- [Search EMA for: {query}]({ema_search_url})"
        ),
        "note": "EMA dataset not available. The model can follow the search link to browse results.",
    }


# ────────────────────────────────────────────────────────────────
# Unified Entry Point
# ────────────────────────────────────────────────────────────────

# Source registry
_SOURCE_HANDLERS = {
    "pubmed": _pubmed_links,
    "fda": _fda_links,
    "ema": _ema_links,
}

VALID_SOURCES = list(_SOURCE_HANDLERS.keys()) + ["all"]


def search_medical_links(
    source: str,
    query: str,
    max_results: int = 20,
    date_range: Optional[str] = None,
    sort: str = "relevance",
    collection: Optional[str] = None,
) -> str:
    """
    Unified entry point for the soft integration layer.

    Parameters
    ----------
    source : str
        One of "pubmed", "fda", "ema", or "all" (searches all three).
    query : str
        Natural language search query (drug name, condition, topic, etc.).
    max_results : int
        Maximum number of links to return per source. Default: 20, max: 100.
    date_range : str, optional
        Date filter, e.g. "2020/01/01-2025/12/31".
    sort : str
        Sort order for PubMed ("relevance" or "date"). Default: "relevance".
    collection : str, optional
        FDA collection to search (default: "drug/label").

    Returns
    -------
    str
        JSON string with markdown-formatted links and structured metadata.
    """
    source = source.lower().strip()
    query = query.strip()

    if not query:
        return json.dumps({"error": "Query cannot be empty."})

    max_results = max(1, min(max_results, MAX_RESULTS_CAP))

    if source == "all":
        # Search all sources, merge results
        all_results = []
        combined_markdown = []

        for src_name, handler in _SOURCE_HANDLERS.items():
            try:
                kwargs = {"query": query, "max_results": max_results}
                if src_name == "pubmed":
                    kwargs["date_range"] = date_range
                    kwargs["sort"] = sort
                elif src_name == "fda":
                    kwargs["date_range"] = date_range
                    kwargs["collection"] = collection
                elif src_name == "ema":
                    kwargs["date_range"] = date_range

                result = handler(**kwargs)
                all_results.append(result)
                if result.get("markdown"):
                    combined_markdown.append(result["markdown"])
            except Exception as exc:
                logger.error(f"Soft integration error for {src_name}: {exc}")
                all_results.append({
                    "source": src_name,
                    "error": str(exc),
                    "links": [],
                    "markdown": "",
                })

        total_links = sum(len(r.get("links", [])) for r in all_results)

        output = {
            "query": query,
            "sources_searched": list(_SOURCE_HANDLERS.keys()),
            "total_links": total_links,
            "results": all_results,
            "markdown": "\n\n---\n\n".join(combined_markdown),
        }
        return json.dumps(output, ensure_ascii=False)

    elif source in _SOURCE_HANDLERS:
        handler = _SOURCE_HANDLERS[source]
        kwargs: Dict[str, Any] = {"query": query, "max_results": max_results}
        if source == "pubmed":
            kwargs["date_range"] = date_range
            kwargs["sort"] = sort
        elif source == "fda":
            kwargs["date_range"] = date_range
            kwargs["collection"] = collection
        elif source == "ema":
            kwargs["date_range"] = date_range

        try:
            result = handler(**kwargs)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Soft integration error for {source}: {exc}")
            return json.dumps({"source": source, "error": str(exc), "links": [], "markdown": ""})
    else:
        return json.dumps({
            "error": f"Unknown source: {source}. Valid sources: {', '.join(VALID_SOURCES)}",
        })
