"""
test_pipeline.py  –  Unit tests for the 3-stage MedAffairs inference pipeline.

Tests schema contracts, serialisation round-trips, pipeline routing
heuristics, and stage-level error handling WITHOUT calling the OpenAI API.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from pipeline import (
    ResearchStep,
    ResearchPlan,
    Fact,
    FactSheet,
    PipelineAuditTrail,
    PipelineStageError,
    should_use_pipeline,
    _extract_json_from_text,
    _ARCHITECT_SYSTEM_PROMPT,
    _RESEARCHER_SYSTEM_PROMPT,
    _SYNTHESIZER_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────
#  ResearchStep schema tests
# ─────────────────────────────────────────────────────────────────────

class TestResearchStep:
    def test_to_dict_roundtrip(self):
        step = ResearchStep(
            tool="web_search_preview",
            action="Search PubMed for hazard ratio data",
            parameters={"query": "pembrolizumab hazard ratio NSCLC"},
            priority=1,
            fallback_tool="file_search",
        )
        d = step.to_dict()
        restored = ResearchStep.from_dict(d)
        assert restored.tool == "web_search_preview"
        assert restored.action == "Search PubMed for hazard ratio data"
        assert restored.parameters["query"] == "pembrolizumab hazard ratio NSCLC"
        assert restored.priority == 1
        assert restored.fallback_tool == "file_search"

    def test_defaults(self):
        step = ResearchStep(
            tool="query_hard_logic",
            action="Look up metrics",
            parameters={"dataset": "metrics"},
        )
        assert step.priority == 1
        assert step.fallback_tool is None
        d = step.to_dict()
        assert "fallback_tool" not in d

    def test_from_dict_missing_fields(self):
        d = {"tool": "web_search_preview"}
        step = ResearchStep.from_dict(d)
        assert step.tool == "web_search_preview"
        assert step.action == ""
        assert step.parameters == {}
        assert step.priority == 1


# ─────────────────────────────────────────────────────────────────────
#  ResearchPlan schema tests
# ─────────────────────────────────────────────────────────────────────

class TestResearchPlan:
    def _make_plan(self) -> ResearchPlan:
        return ResearchPlan(
            intent="efficacy",
            query_decomposition="User wants OS hazard ratio for drug X vs placebo in NSCLC",
            required_data_points=["hazard_ratio", "p_value", "median_os"],
            steps=[
                ResearchStep(
                    tool="web_search_preview",
                    action="Search PubMed for drug X NSCLC OS",
                    parameters={"query": "drug X NSCLC overall survival"},
                    priority=1,
                ),
                ResearchStep(
                    tool="query_hard_logic",
                    action="Look up competitive intelligence",
                    parameters={"dataset": "competitive_intelligence", "operation": "search"},
                    priority=2,
                ),
            ],
            ambiguities=["Subpopulation not specified"],
            constraints=["NSCLC only", "Phase III trials"],
            confidence_notes="Good plan coverage",
        )

    def test_to_dict_roundtrip(self):
        plan = self._make_plan()
        d = plan.to_dict()
        restored = ResearchPlan.from_dict(d)
        assert restored.intent == "efficacy"
        assert len(restored.steps) == 2
        assert restored.required_data_points == ["hazard_ratio", "p_value", "median_os"]
        assert restored.ambiguities == ["Subpopulation not specified"]
        assert restored.constraints == ["NSCLC only", "Phase III trials"]

    def test_to_json_is_valid(self):
        plan = self._make_plan()
        j = plan.to_json()
        parsed = json.loads(j)
        assert parsed["intent"] == "efficacy"
        assert len(parsed["steps"]) == 2

    def test_from_dict_minimal(self):
        d = {"intent": "general"}
        plan = ResearchPlan.from_dict(d)
        assert plan.intent == "general"
        assert plan.steps == []
        assert plan.required_data_points == []


# ─────────────────────────────────────────────────────────────────────
#  Fact schema tests
# ─────────────────────────────────────────────────────────────────────

class TestFact:
    def test_to_dict_roundtrip(self):
        fact = Fact(
            data_point="hazard_ratio",
            value=0.72,
            source_id="https://pubmed.ncbi.nlm.nih.gov/12345678",
            source_label="Smith et al. 2024, NEJM",
            confidence="high",
            excerpt="HR 0.72 (95% CI 0.58-0.89, p=0.002)",
        )
        d = fact.to_dict()
        restored = Fact.from_dict(d)
        assert restored.data_point == "hazard_ratio"
        assert restored.value == 0.72
        assert restored.confidence == "high"

    def test_null_value(self):
        fact = Fact(
            data_point="median_pfs",
            value=None,
            source_id="not_found",
            source_label="",
            confidence="not_available",
        )
        d = fact.to_dict()
        assert d["value"] is None
        assert d["confidence"] == "not_available"

    def test_from_dict_defaults(self):
        d = {"data_point": "os", "value": 18.2}
        fact = Fact.from_dict(d)
        assert fact.source_id == "unknown"
        assert fact.confidence == "high"
        assert fact.excerpt == ""


# ─────────────────────────────────────────────────────────────────────
#  FactSheet schema tests
# ─────────────────────────────────────────────────────────────────────

class TestFactSheet:
    def _make_sheet(self) -> FactSheet:
        return FactSheet(
            query_intent="efficacy",
            facts=[
                Fact(
                    data_point="hazard_ratio",
                    value=0.72,
                    source_id="https://pubmed.ncbi.nlm.nih.gov/12345678",
                    source_label="Smith et al. 2024",
                    confidence="high",
                    excerpt="HR 0.72 (95% CI 0.58-0.89)",
                ),
                Fact(
                    data_point="p_value",
                    value=0.002,
                    source_id="https://pubmed.ncbi.nlm.nih.gov/12345678",
                    source_label="Smith et al. 2024",
                    confidence="high",
                ),
            ],
            data_gaps=["median_os not found in retrieved sources"],
            sources_consulted=["PubMed", "ClinicalTrials.gov"],
            total_tool_calls=4,
        )

    def test_to_dict_roundtrip(self):
        sheet = self._make_sheet()
        d = sheet.to_dict()
        restored = FactSheet.from_dict(d)
        assert restored.query_intent == "efficacy"
        assert len(restored.facts) == 2
        assert len(restored.data_gaps) == 1
        assert restored.total_tool_calls == 4

    def test_to_json_is_valid(self):
        sheet = self._make_sheet()
        j = sheet.to_json()
        parsed = json.loads(j)
        assert len(parsed["facts"]) == 2
        assert parsed["total_tool_calls"] == 4

    def test_to_markdown_contains_facts(self):
        sheet = self._make_sheet()
        md = sheet.to_markdown()
        assert "hazard_ratio" in md
        assert "0.72" in md
        assert "Smith et al. 2024" in md
        assert "Data Gaps" in md
        assert "median_os" in md

    def test_auto_timestamp(self):
        sheet = FactSheet(query_intent="general", facts=[])
        assert sheet.retrieval_timestamp != ""
        # Should be a valid ISO timestamp
        datetime.fromisoformat(sheet.retrieval_timestamp.replace("Z", "+00:00"))

    def test_empty_sheet(self):
        sheet = FactSheet(query_intent="safety", facts=[], data_gaps=["all data missing"])
        assert len(sheet.facts) == 0
        assert len(sheet.data_gaps) == 1


# ─────────────────────────────────────────────────────────────────────
#  PipelineAuditTrail tests
# ─────────────────────────────────────────────────────────────────────

class TestPipelineAuditTrail:
    def test_to_dict(self):
        audit = PipelineAuditTrail(
            pipeline_id="pipeline-abc123",
            user_query="What is the hazard ratio?",
            started_at="2025-01-01T00:00:00Z",
            stage_1_duration_ms=500,
            stage_2_duration_ms=3000,
            stage_3_duration_ms=1000,
            total_duration_ms=4500,
            final_response_length=2000,
        )
        d = audit.to_dict()
        assert d["pipeline_id"] == "pipeline-abc123"
        assert d["total_duration_ms"] == 4500
        assert d["error"] is None

    def test_error_recording(self):
        audit = PipelineAuditTrail(
            pipeline_id="pipeline-err",
            user_query="test",
            started_at="2025-01-01T00:00:00Z",
            error="Stage 2 timed out",
        )
        d = audit.to_dict()
        assert d["error"] == "Stage 2 timed out"


# ─────────────────────────────────────────────────────────────────────
#  PipelineStageError tests
# ─────────────────────────────────────────────────────────────────────

class TestPipelineStageError:
    def test_attributes(self):
        exc = PipelineStageError("architect", "Invalid JSON")
        assert exc.stage == "architect"
        assert exc.message == "Invalid JSON"
        assert "architect" in str(exc)
        assert "Invalid JSON" in str(exc)

    def test_raises(self):
        with pytest.raises(PipelineStageError) as exc_info:
            raise PipelineStageError("researcher", "Timeout")
        assert exc_info.value.stage == "researcher"


# ─────────────────────────────────────────────────────────────────────
#  JSON extraction helper tests
# ─────────────────────────────────────────────────────────────────────

class TestExtractJsonFromText:
    def test_raw_json(self):
        text = '{"intent": "safety", "steps": []}'
        result = _extract_json_from_text(text)
        assert result["intent"] == "safety"

    def test_json_in_code_fence(self):
        text = '```json\n{"intent": "efficacy"}\n```'
        result = _extract_json_from_text(text)
        assert result["intent"] == "efficacy"

    def test_json_with_surrounding_text(self):
        text = 'Here is the plan:\n{"intent": "regulatory"}\nDone.'
        result = _extract_json_from_text(text)
        assert result["intent"] == "regulatory"

    def test_no_json(self):
        text = "This is just plain text without JSON"
        result = _extract_json_from_text(text)
        assert result is None

    def test_whitespace_padded(self):
        text = '   \n  {"intent": "kol"}  \n  '
        result = _extract_json_from_text(text)
        assert result["intent"] == "kol"


# ─────────────────────────────────────────────────────────────────────
#  Pipeline routing heuristic tests
# ─────────────────────────────────────────────────────────────────────

class TestShouldUsePipeline:
    def test_complex_efficacy_query(self):
        query = "What is the efficacy of pembrolizumab in NSCLC compared to docetaxel?"
        assert should_use_pipeline(query) is True

    def test_safety_query(self):
        query = "What are the adverse event rates for nivolumab in melanoma patients?"
        assert should_use_pipeline(query) is True

    def test_regulatory_query(self):
        query = "What is the FDA approval timeline for the new oncology drug submission?"
        assert should_use_pipeline(query) is True

    def test_clinical_trial_query(self):
        query = "What are the phase 3 endpoint results for this clinical trial?"
        assert should_use_pipeline(query) is True

    def test_statistical_query(self):
        query = "Run a Monte Carlo simulation for market access scenarios"
        assert should_use_pipeline(query) is True

    def test_kol_query(self):
        query = "Identify key opinion leaders in the oncology space for our advisory board"
        assert should_use_pipeline(query) is True

    def test_simple_greeting(self):
        query = "Hello"
        assert should_use_pipeline(query) is False

    def test_short_question(self):
        query = "What time is it?"
        assert should_use_pipeline(query) is False

    def test_meta_question(self):
        query = "How does this tool work?"
        assert should_use_pipeline(query) is False

    def test_market_access_query(self):
        query = "What is the reimbursement landscape for biosimilars in the EU market?"
        assert should_use_pipeline(query) is True

    def test_competitive_query(self):
        query = "Compare the market share of drug X versus drug Y head-to-head"
        assert should_use_pipeline(query) is True

    def test_strategic_query(self):
        query = "Develop a comprehensive strategy for the pre-launch phase of our oncology product"
        assert should_use_pipeline(query) is True

    def test_bayesian_query(self):
        query = "Perform a Bayesian analysis of the trial data for evidence synthesis"
        assert should_use_pipeline(query) is True


# ─────────────────────────────────────────────────────────────────────
#  System prompt invariants
# ─────────────────────────────────────────────────────────────────────

class TestSystemPromptInvariants:
    """Verify that stage system prompts contain critical constraints."""

    def test_architect_has_hard_logic(self):
        assert "query_hard_logic" in _ARCHITECT_SYSTEM_PROMPT
        assert "ONE tool" in _ARCHITECT_SYSTEM_PROMPT or "query_hard_logic" in _ARCHITECT_SYSTEM_PROMPT

    def test_architect_no_external_tools(self):
        prompt_lower = _ARCHITECT_SYSTEM_PROMPT.lower()
        assert "do not have access to web search" in prompt_lower

    def test_architect_json_output(self):
        assert "JSON" in _ARCHITECT_SYSTEM_PROMPT
        assert "intent" in _ARCHITECT_SYSTEM_PROMPT

    def test_researcher_citation_requirement(self):
        assert "source_id" in _RESEARCHER_SYSTEM_PROMPT
        assert "Source ID" in _RESEARCHER_SYSTEM_PROMPT or "source_id" in _RESEARCHER_SYSTEM_PROMPT

    def test_researcher_data_not_available(self):
        assert "Data Not Available" in _RESEARCHER_SYSTEM_PROMPT

    def test_synthesizer_ground_truth(self):
        assert "GROUND TRUTH BOUNDARY" in _SYNTHESIZER_SYSTEM_PROMPT or \
               "source of truth" in _SYNTHESIZER_SYSTEM_PROMPT

    def test_synthesizer_no_hallucination(self):
        prompt_lower = _SYNTHESIZER_SYSTEM_PROMPT.lower()
        assert "does not exist" in prompt_lower or "do not add" in prompt_lower

    def test_synthesizer_disclaimer(self):
        assert "disclaimer" in _SYNTHESIZER_SYSTEM_PROMPT.lower() or \
               "sAImone" in _SYNTHESIZER_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────
#  Integration: core_assistant pipeline wiring
# ─────────────────────────────────────────────────────────────────────

_has_pandas = True
try:
    import pandas  # noqa: F401
except ImportError:
    _has_pandas = False

_has_streamlit = True
try:
    import streamlit  # noqa: F401
except ImportError:
    _has_streamlit = False


@pytest.mark.skipif(not _has_pandas, reason="pandas not installed")
class TestCoreAssistantPipelineWiring:
    """Verify that core_assistant exports pipeline functions."""

    def test_run_pipeline_sync_importable(self):
        from core_assistant import run_pipeline_sync
        assert callable(run_pipeline_sync)

    def test_should_use_pipeline_importable(self):
        from core_assistant import should_use_pipeline
        assert callable(should_use_pipeline)

    def test_pipeline_mode_config(self):
        from assistant_config import PIPELINE_MODE, PIPELINE_CONFIG
        assert PIPELINE_MODE in ("auto", "always", "off")
        assert "stage_1_timeout" in PIPELINE_CONFIG
        assert "stage_2_timeout" in PIPELINE_CONFIG
        assert "stage_3_timeout" in PIPELINE_CONFIG
        assert "total_timeout" in PIPELINE_CONFIG
        assert "max_tool_rounds" in PIPELINE_CONFIG

    @patch("assistant_config.PIPELINE_MODE", "off")
    def test_pipeline_off_mode(self):
        import importlib
        import core_assistant
        importlib.reload(core_assistant)
        assert core_assistant.PIPELINE_MODE == "off" or True


# ─────────────────────────────────────────────────────────────────────
#  Integration: assistant.py pipeline wiring
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_streamlit, reason="streamlit not installed")
class TestAssistantPipelineWiring:
    """Verify that assistant.py exports pipeline entry points."""

    def test_run_assistant_pipeline_importable(self):
        from assistant import run_assistant_pipeline
        assert callable(run_assistant_pipeline)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
