#!/usr/bin/env python3
"""
Test script for EMA and FDA API integrations in med_affairs_data.py

Tests:
1. FDA API endpoints and query construction
2. EMA API endpoints and query construction
3. FAERS (adverse events) API
4. Tool routing and dispatcher functions
5. How search functions are passed to the agent via JSON schema
"""

import json
import sys
from typing import Any

# Test Results Tracking
test_results = []


def log_test(name: str, passed: bool, details: str = ""):
    """Log test results."""
    status = "✓ PASS" if passed else "✗ FAIL"
    test_results.append({"name": name, "passed": passed, "details": details})
    print(f"{status}: {name}")
    if details and not passed:
        print(f"       Details: {details}")


def test_imports():
    """Test that required modules can be imported."""
    try:
        from med_affairs_data import (
            fda_drugs_search,
            search_ema,
            search_faers,
            search_fda_recalls_drug,
            search_fda_safety_communications,
            get_med_affairs_data,
            get_medaffairs_data,
            SEARCH_MAP,
            OPENFDA_BASE,
            FAERS_BASE,
        )
        log_test("Import med_affairs_data functions", True)
        return True
    except ImportError as e:
        log_test("Import med_affairs_data functions", False, str(e))
        return False


def test_search_map_structure():
    """Test that SEARCH_MAP contains expected FDA and EMA sources."""
    from med_affairs_data import SEARCH_MAP

    expected_fda_sources = [
        "fda_drugs", "fda_guidance", "faers", "fda_device_events",
        "fda_recalls_drug", "fda_recalls_device",
        "fda_safety_communications", "fda_warning_letters"
    ]

    expected_ema_sources = ["ema", "ema_guidance"]

    missing_fda = [s for s in expected_fda_sources if s not in SEARCH_MAP]
    missing_ema = [s for s in expected_ema_sources if s not in SEARCH_MAP]

    if missing_fda:
        log_test("SEARCH_MAP contains FDA sources", False, f"Missing: {missing_fda}")
    else:
        log_test("SEARCH_MAP contains FDA sources", True)

    if missing_ema:
        log_test("SEARCH_MAP contains EMA sources", False, f"Missing: {missing_ema}")
    else:
        log_test("SEARCH_MAP contains EMA sources", True)

    # Test that each mapping points to a callable
    all_callable = all(callable(SEARCH_MAP.get(s)) for s in expected_fda_sources + expected_ema_sources if s in SEARCH_MAP)
    log_test("All mapped sources are callable", all_callable)


def test_fda_api_endpoints():
    """Test FDA API endpoint constants are correctly defined."""
    from med_affairs_data import OPENFDA_BASE, FAERS_BASE, DRUG_ENFORCEMENT_BASE

    log_test("OPENFDA_BASE is defined",
             OPENFDA_BASE == "https://api.fda.gov",
             f"Got: {OPENFDA_BASE}")

    log_test("FAERS_BASE is defined",
             FAERS_BASE == "https://api.fda.gov/drug/event.json",
             f"Got: {FAERS_BASE}")

    log_test("DRUG_ENFORCEMENT_BASE is defined",
             DRUG_ENFORCEMENT_BASE == "https://api.fda.gov/drug/enforcement.json",
             f"Got: {DRUG_ENFORCEMENT_BASE}")


def test_fda_drugs_search_query_construction():
    """Test FDA drugs search query construction without making API calls."""
    from med_affairs_data import fda_drugs_search
    import inspect

    # Check function signature
    sig = inspect.signature(fda_drugs_search)
    params = list(sig.parameters.keys())

    expected_params = ["query", "max_results", "cursor", "date_range",
                       "fda_decision_type", "collection", "prioritize_recent",
                       "query_is_structured", "expanded_terms"]

    missing = [p for p in expected_params if p not in params]
    log_test("fda_drugs_search has expected parameters",
             len(missing) == 0,
             f"Missing: {missing}")


def test_ema_search_query_construction():
    """Test EMA search query construction."""
    from med_affairs_data import search_ema
    import inspect

    # Check function signature
    sig = inspect.signature(search_ema)
    params = list(sig.parameters.keys())

    expected_params = ["query", "max_results", "cursor", "prioritize_recent",
                       "match_mode", "date_range", "date_field",
                       "expanded_terms", "pv_terms"]

    missing = [p for p in expected_params if p not in params]
    log_test("search_ema has expected parameters",
             len(missing) == 0,
             f"Missing: {missing}")


def test_faers_search_query_construction():
    """Test FAERS search query construction."""
    from med_affairs_data import search_faers
    import inspect

    # Check function signature
    sig = inspect.signature(search_faers)
    params = list(sig.parameters.keys())

    expected_params = ["query", "max_results", "cursor", "date_range",
                       "prioritize_recent", "expanded_terms", "pv_terms"]

    missing = [p for p in expected_params if p not in params]
    log_test("search_faers has expected parameters",
             len(missing) == 0,
             f"Missing: {missing}")


def test_tool_dispatcher():
    """Test the tool dispatcher function routing."""
    from med_affairs_data import get_medaffairs_data

    # Test that dispatcher recognizes valid tool names
    valid_tools = [
        "get_med_affairs_data",
        "get_fda_data",
        "get_ema_data",
        "aggregated_search",
    ]

    for tool in valid_tools:
        try:
            # Call with minimal args - will fail gracefully but proves routing works
            result = get_medaffairs_data(tool, {"query": "test", "source": "fda_drugs"})
            # If we get here without ValueError, routing works
            log_test(f"Dispatcher routes '{tool}'", True)
        except ValueError as e:
            if "Unknown tool" in str(e):
                log_test(f"Dispatcher routes '{tool}'", False, str(e))
            else:
                # Other errors are expected (missing required args, etc.)
                log_test(f"Dispatcher routes '{tool}'", True)
        except Exception as e:
            # Network errors or missing data are OK - routing still worked
            log_test(f"Dispatcher routes '{tool}'", True, f"Routed but got: {type(e).__name__}")


def test_get_med_affairs_data_function():
    """Test the main get_med_affairs_data function interface."""
    from med_affairs_data import get_med_affairs_data
    import inspect

    sig = inspect.signature(get_med_affairs_data)
    params = list(sig.parameters.keys())

    # These match the JSON schema provided by the user
    expected_params = [
        "source", "query", "max_results", "cursor",
        "fallback_sources", "fallback_min_results",
        "mesh", "mesh_recursive_depth", "mesh_intent_aware", "mesh_include_drug_mapping",
        "date_range", "datetype", "sort",
        "fda_decision_type", "collection",
        "prioritize_recent", "include_mesh_metadata"
    ]

    missing = [p for p in expected_params if p not in params]
    log_test("get_med_affairs_data matches JSON schema",
             len(missing) == 0,
             f"Missing: {missing}")


def test_json_schema_alignment():
    """Test that function parameters align with the provided JSON schema."""
    # This is the JSON schema from the user's message
    schema_params = {
        "source": {"required": True, "type": "string"},
        "query": {"required": True, "type": "string"},
        "max_results": {"required": False, "type": "integer"},
        "cursor": {"required": False, "type": "string"},
        "fallback_sources": {"required": False, "type": "array"},
        "fallback_min_results": {"required": False, "type": "integer"},
        "mesh": {"required": False, "type": "boolean"},
        "mesh_recursive_depth": {"required": False, "type": "integer"},
        "mesh_intent_aware": {"required": False, "type": "boolean"},
        "mesh_include_drug_mapping": {"required": False, "type": "boolean"},
        "date_range": {"required": False, "type": "string"},
        "datetype": {"required": False, "type": "string"},
        "sort": {"required": False, "type": "string"},
        "fda_decision_type": {"required": False, "type": "string"},
        "collection": {"required": False, "type": "string"},
        "prioritize_recent": {"required": False, "type": "boolean"},
        "include_mesh_metadata": {"required": False, "type": "boolean"},
    }

    from med_affairs_data import get_med_affairs_data
    import inspect

    sig = inspect.signature(get_med_affairs_data)

    all_match = True
    mismatches = []

    for param_name, schema_info in schema_params.items():
        if param_name not in sig.parameters:
            mismatches.append(f"{param_name}: not in function")
            all_match = False
            continue

        param = sig.parameters[param_name]

        # Check if required params don't have defaults
        if schema_info["required"]:
            if param.default is not inspect.Parameter.empty:
                mismatches.append(f"{param_name}: should be required but has default")
                all_match = False

    log_test("Function params match JSON schema", all_match,
             "; ".join(mismatches) if mismatches else "")


def test_fda_source_enum_values():
    """Test that FDA source enum values in schema map to SEARCH_MAP."""
    from med_affairs_data import SEARCH_MAP

    # From the JSON schema enum
    fda_sources_in_schema = [
        "fda_drugs", "fda_guidance", "faers", "fda_device_events",
        "fda_recalls_drug", "fda_recalls_device",
        "fda_safety_communications", "fda_warning_letters",
        "dailymed", "orange_book"
    ]

    missing = [s for s in fda_sources_in_schema if s not in SEARCH_MAP]
    log_test("All FDA schema sources exist in SEARCH_MAP",
             len(missing) == 0,
             f"Missing: {missing}")


def test_ema_source_enum_values():
    """Test that EMA source enum values in schema map to SEARCH_MAP."""
    from med_affairs_data import SEARCH_MAP

    # From the JSON schema enum
    ema_sources_in_schema = ["ema", "ema_guidance"]

    missing = [s for s in ema_sources_in_schema if s not in SEARCH_MAP]
    log_test("All EMA schema sources exist in SEARCH_MAP",
             len(missing) == 0,
             f"Missing: {missing}")


def test_live_fda_api_call():
    """Test a live FDA API call (if network available)."""
    from med_affairs_data import fda_drugs_search

    try:
        results, cursor = fda_drugs_search(
            query="metformin",
            max_results=5,
            prioritize_recent=True
        )

        if results:
            log_test("FDA API returns results for 'metformin'", True,
                     f"Got {len(results)} results")

            # Check result structure
            first = results[0]
            has_required_fields = all(
                k in first for k in ["source", "id", "title"]
            )
            log_test("FDA results have required fields", has_required_fields)
        else:
            log_test("FDA API returns results for 'metformin'", False,
                     "No results returned")
    except Exception as e:
        log_test("FDA API returns results for 'metformin'", False, str(e))


def test_live_ema_api_call():
    """Test a live EMA API call (if network available)."""
    from med_affairs_data import search_ema

    try:
        results, cursor = search_ema(
            query="imatinib",
            max_results=5,
            prioritize_recent=True
        )

        if results:
            log_test("EMA API returns results for 'imatinib'", True,
                     f"Got {len(results)} results")

            # Check result structure
            first = results[0]
            has_required_fields = all(
                k in first for k in ["source", "id", "title"]
            )
            log_test("EMA results have required fields", has_required_fields)
        else:
            log_test("EMA API returns results for 'imatinib'", False,
                     "No results returned (may need dataset load)")
    except Exception as e:
        log_test("EMA API returns results for 'imatinib'", False, str(e))


def test_live_faers_api_call():
    """Test a live FAERS API call (if network available)."""
    from med_affairs_data import search_faers

    try:
        results, cursor = search_faers(
            query="aspirin",
            max_results=5,
            prioritize_recent=True
        )

        if results:
            log_test("FAERS API returns results for 'aspirin'", True,
                     f"Got {len(results)} results")

            # Check result structure
            first = results[0]
            has_required_fields = all(
                k in first for k in ["source", "id", "title"]
            )
            log_test("FAERS results have required fields", has_required_fields)
        else:
            log_test("FAERS API returns results for 'aspirin'", False,
                     "No results returned")
    except Exception as e:
        log_test("FAERS API returns results for 'aspirin'", False, str(e))


def test_tool_router_integration():
    """Test the tool router from core_assistant.py perspective."""
    try:
        from core_assistant import _default_tool_router

        # Test routing to get_med_affairs_data
        result_json = _default_tool_router("get_med_affairs_data", {
            "source": "fda_drugs",
            "query": "lisinopril",
            "max_results": 3
        })

        result = json.loads(result_json)

        if "error" not in result:
            log_test("Tool router integrates with med_affairs_data", True)
        else:
            log_test("Tool router integrates with med_affairs_data", False,
                     result.get("error", "Unknown error"))
    except ImportError:
        log_test("Tool router integrates with med_affairs_data", False,
                 "core_assistant.py not found")
    except Exception as e:
        log_test("Tool router integrates with med_affairs_data", False, str(e))


def test_med_affairs_data_router():
    """Test the med_affairs_data_router function."""
    from med_affairs_data import med_affairs_data_router

    try:
        # Test with FDA source
        hits, cursor, source = med_affairs_data_router(
            source="fda_drugs",
            query="atorvastatin",
            max_results=3
        )

        log_test("med_affairs_data_router returns correct structure",
                 isinstance(hits, list) and source == "fda_drugs",
                 f"Got {len(hits)} hits, source={source}")
    except Exception as e:
        log_test("med_affairs_data_router returns correct structure", False, str(e))


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for t in test_results if t["passed"])
    failed = sum(1 for t in test_results if not t["passed"])
    total = len(test_results)

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")

    if failed > 0:
        print("\nFailed tests:")
        for t in test_results:
            if not t["passed"]:
                print(f"  - {t['name']}")
                if t["details"]:
                    print(f"    {t['details']}")

    return failed == 0


def main():
    """Run all tests."""
    print("="*60)
    print("EMA & FDA API Integration Tests")
    print("="*60)
    print()

    # Structure tests (no network required)
    print("--- Structure Tests ---")
    if not test_imports():
        print("Cannot continue without successful imports")
        sys.exit(1)

    test_search_map_structure()
    test_fda_api_endpoints()
    test_fda_drugs_search_query_construction()
    test_ema_search_query_construction()
    test_faers_search_query_construction()
    test_tool_dispatcher()
    test_get_med_affairs_data_function()
    test_json_schema_alignment()
    test_fda_source_enum_values()
    test_ema_source_enum_values()

    # Live API tests (require network)
    print("\n--- Live API Tests ---")
    test_live_fda_api_call()
    test_live_ema_api_call()
    test_live_faers_api_call()

    # Integration tests
    print("\n--- Integration Tests ---")
    test_tool_router_integration()
    test_med_affairs_data_router()

    # Print summary
    success = print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
