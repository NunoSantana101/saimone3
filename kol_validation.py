# kol_validation.py
from typing import List, Dict, Any
from med_affairs_data import get_med_affairs_data

MIN_CONFIDENCE = 0.75

def _score(evidence: List[dict]) -> float:
    # Simple, conservative scoring: coverage × recency × source diversity
    if not evidence:
        return 0.0
    sources = {e["source"] for e in evidence if e.get("source")}
    dates = [e.get("date") for e in evidence if e.get("date")]
    recency_factor = 1.0 if any(d for d in dates) else 0.5
    diversity_factor = min(1.0, len(sources) / 3.0)
    return round(0.5 * recency_factor + 0.5 * diversity_factor, 2)

def validate_kol(
    full_name: str,
    affiliation: str | None = None,
    specialty: str | None = None,
    claims: List[str] | None = None,
    evidence_sources: List[str] | None = None,
    recency_years: int = 3,
    strict: bool = True,
) -> Dict[str, Any]:

    evidence_sources = evidence_sources or [
        "openalex_kol", "pubmed_investigators", "orcid_record", "open_payments"
    ]

    evidence: List[dict] = []
    for src in evidence_sources:
        res = get_med_affairs_data(
            source=src,
            query=full_name,
            max_results=20,
            prioritize_recent=True
        )
        for r in res.get("results", []):
            evidence.append({
                "source": src,
                "title": r.get("title"),
                "url": r.get("url"),
                "date": r.get("date"),
                "extra": r.get("extra", {})
            })

    confidence = _score(evidence)

    verdict = "verified" if confidence >= MIN_CONFIDENCE else "unverified"
    issues = []
    if strict and verdict != "verified":
        issues.append("Insufficient authoritative, recent evidence.")
    if claims:
        issues.append("Claims provided require explicit citation per item (not yet mapped).")

    return {
        "kol": {
            "name": full_name,
            "affiliation": affiliation,
            "specialty": specialty
        },
        "verdict": verdict,
        "confidence": confidence,
        "evidence_count": len(evidence),
        "evidence": evidence[:10],  # capped for safety
        "issues": issues
    }
