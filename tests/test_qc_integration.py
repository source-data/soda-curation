# tests/test_qc_integration.py
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from soda_curation.qc.prompt_registry import registry
from soda_curation.qc.qc_tests.individual_data_points import (
    IndividualDataPointsAnalyzer,
)

# Add more imports as needed


# This test will use fixture data to avoid making real API calls
def test_integration_with_fixture_data():
    # Load fixture data
    fixture_path = Path(__file__).parent / "fixtures" / "figure_data.json"
    with open(fixture_path, "r") as f:
        figure_data = json.load(f)

    # Load expected results
    expected_path = Path(__file__).parent / "fixtures" / "expected_results.json"
    with open(expected_path, "r") as f:
        expected_results = json.load(f)

    # Create configuration
    config = {
        "model": "mock_model",  # We'll mock the model API
        "use_mock": True,  # Flag to use mock responses instead of real API
    }

    # Test individual_data_points analyzer
    analyzer = IndividualDataPointsAnalyzer(config)
    result = analyzer.analyze(figure_data)

    # Compare with expected results
    assert result == expected_results["individual_data_points"]
