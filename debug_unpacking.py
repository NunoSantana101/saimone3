#!/usr/bin/env python3
"""
Debug script to trace the unpacking error in med_affairs_data.py
"""

import sys
import traceback

# Import the module
from med_affairs_data import (
    SEARCH_MAP,
    med_affairs_data_router,
    get_med_affairs_data,
)


def test_adapter_return_values():
    """Test that adapters return 2 values."""
    print("=" * 60)
    print("Testing adapter return values")
    print("=" * 60)

    test_sources = ["pubmed", "clinicaltrials", "fda_drugs", "ema", "faers", "regulatory_combined"]

    for source in test_sources:
        if source not in SEARCH_MAP:
            print(f"SKIP: {source} not in SEARCH_MAP")
            continue

        adapter = SEARCH_MAP[source]
        print(f"\n--- Testing {source} adapter ---")
        print(f"Function: {adapter.__name__}")

        try:
            # Call the adapter directly with minimal params
            result = adapter("test", 3)

            # Check what type of result we got
            print(f"Result type: {type(result)}")
            print(f"Result length (if tuple): {len(result) if isinstance(result, tuple) else 'N/A'}")

            if isinstance(result, tuple):
                for i, item in enumerate(result):
                    print(f"  [{i}] type={type(item).__name__}, value preview={str(item)[:80]}...")

                # Try to unpack as 2 values
                try:
                    hits, cursor = result
                    print(f"SUCCESS: Unpacks to 2 values (hits={len(hits)}, cursor={cursor})")
                except ValueError as e:
                    print(f"ERROR: Cannot unpack to 2 values: {e}")

        except Exception as e:
            print(f"ERROR calling adapter: {e}")


def test_router_return_values():
    """Test that med_affairs_data_router returns 3 values."""
    print("\n" + "=" * 60)
    print("Testing med_affairs_data_router return values")
    print("=" * 60)

    test_sources = ["fda_drugs", "ema", "faers"]

    for source in test_sources:
        print(f"\n--- Testing router with {source} ---")

        try:
            result = med_affairs_data_router(source, "test", 3)

            print(f"Result type: {type(result)}")
            print(f"Result length (if tuple): {len(result) if isinstance(result, tuple) else 'N/A'}")

            if isinstance(result, tuple):
                for i, item in enumerate(result):
                    print(f"  [{i}] type={type(item).__name__}, value preview={str(item)[:80]}...")

                # Try to unpack as 3 values
                try:
                    hits, cursor, used_source = result
                    print(f"SUCCESS: Unpacks to 3 values (hits={len(hits)}, cursor={cursor}, source={used_source})")
                except ValueError as e:
                    print(f"ERROR: Cannot unpack to 3 values: {e}")

        except Exception as e:
            print(f"ERROR calling router: {e}")
            traceback.print_exc()


def test_get_med_affairs_data():
    """Test the full get_med_affairs_data function."""
    print("\n" + "=" * 60)
    print("Testing get_med_affairs_data")
    print("=" * 60)

    test_cases = [
        {"source": "fda_drugs", "query": "semaglutide", "max_results": 5},
        {"source": "ema", "query": "imatinib", "max_results": 5},
        {"source": "clinicaltrials", "query": "diabetes", "max_results": 5},
    ]

    for test_case in test_cases:
        print(f"\n--- Testing {test_case} ---")

        try:
            result = get_med_affairs_data(**test_case)

            print(f"Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"Keys: {list(result.keys())}")
                print(f"Source: {result.get('source')}")
                print(f"Total results: {result.get('total_results')}")
                if result.get('results'):
                    print(f"First result: {str(result['results'][0])[:100]}...")
            else:
                print(f"Unexpected result type: {result}")

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()


def test_specific_error_scenario():
    """Recreate the exact error scenario from the logs."""
    print("\n" + "=" * 60)
    print("Recreating exact error scenario")
    print("=" * 60)

    # This is what the aggregated_search calls
    common_kwargs = {
        "mesh": True,
        "mesh_intent_aware": True,
        "mesh_include_drug_mapping": True,
        "include_mesh_metadata": True,
        "date_range": None,
        "prioritize_recent": True,
    }

    for source in ["clinicaltrials", "pubmed", "regulatory_combined"]:
        print(f"\n--- Testing source: {source} ---")
        try:
            response = get_med_affairs_data(
                source=source,
                query="test",
                max_results=10,
                **common_kwargs
            )
            print(f"SUCCESS: Got response with {response.get('total_results', 0)} results")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    print("Debug script for unpacking error\n")

    # Skip network tests if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        print("Running local tests only (no network calls)\n")
        test_adapter_return_values()
    else:
        test_adapter_return_values()
        test_router_return_values()
        test_get_med_affairs_data()
        test_specific_error_scenario()
