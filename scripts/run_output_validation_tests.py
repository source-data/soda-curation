#!/usr/bin/env python3
"""
Comprehensive test runner for SODA curation pipeline output validation.

This script runs all output format validation tests to ensure the pipeline
produces the expected JSON structure and catches issues early.
"""
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ FAILED: {description}")
            print("Error output:")
            print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False


def check_test_data():
    """Check if test data is available."""
    test_data_dir = Path("tests/test_suite/test_data")
    example_file = test_data_dir / "EMBOJ-2024-119382.zip.json"

    if not example_file.exists():
        print("❌ Test data not found!")
        print(f"Expected file: {example_file}")
        print("Please ensure the example output file is in the test_data directory.")
        return False

    print(f"✅ Test data found: {example_file}")

    # Validate the JSON structure
    try:
        with open(example_file, "r") as f:
            data = json.load(f)

        # Basic structure validation
        required_fields = [
            "appendix",
            "figures",
            "data_availability",
            "locate_captions_hallucination_score",
            "locate_data_section_hallucination_score",
            "manuscript_id",
            "xml",
            "docx",
            "pdf",
            "ai_response_locate_captions",
            "ai_response_extract_individual_captions",
            "cost",
            "ai_provider",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f"❌ Example file missing required fields: {missing_fields}")
            return False

        print("✅ Example file has all required fields")
        print(f"   - {len(data['figures'])} figures")
        print(f"   - {sum(len(fig['panels']) for fig in data['figures'])} total panels")
        print(f"   - Manuscript ID: {data['manuscript_id']}")

        return True

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in example file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading example file: {e}")
        return False


def main():
    """Run all output validation tests."""
    print("🧪 SODA Curation Pipeline - Output Validation Test Suite")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Check test data
    if not check_test_data():
        sys.exit(1)

    # Define test commands
    test_commands = [
        {
            "cmd": "docker compose -f docker-compose.dev.yml run --rm soda python -m pytest tests/test_suite/test_pipeline_output_structure_validation.py -v",
            "description": "Pipeline Output Structure Validation Tests",
        },
        {
            "cmd": "docker compose -f docker-compose.dev.yml run --rm soda python -m pytest tests/test_suite/test_output_format_validation.py -v",
            "description": "Output Format Validation Tests",
        },
        {
            "cmd": "docker compose -f docker-compose.dev.yml run --rm soda python -m pytest tests/test_suite/test_pipeline_output_validation.py -v",
            "description": "Pipeline Output Validation Tests",
        },
        {
            "cmd": "docker compose -f docker-compose.dev.yml run --rm soda python -m pytest tests/test_suite/test_zip_validation.py -v",
            "description": "ZIP File Validation Tests",
        },
        {
            "cmd": "docker compose -f docker-compose.dev.yml run --rm soda python -m pytest tests/test_suite/test_object_detection_integration.py -v",
            "description": "Object Detection Integration Tests",
        },
        {
            "cmd": "docker compose -f docker-compose.dev.yml run --rm soda python -m pytest tests/test_suite/test_object_detection_fix.py -v",
            "description": "Object Detection Fix Tests",
        },
        {
            "cmd": "docker compose -f docker-compose.dev.yml run --rm soda python -m pytest tests/test_suite/test_object_detection.py -v",
            "description": "Object Detection Unit Tests",
        },
    ]

    # Run all tests
    results = []
    for test in test_commands:
        success = run_command(test["cmd"], test["description"])
        results.append((test["description"], success))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")

    print(f"\nOverall: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 All tests passed! The pipeline output format is validated.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
