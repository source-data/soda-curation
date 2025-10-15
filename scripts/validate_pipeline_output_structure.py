#!/usr/bin/env python3
"""
Pipeline Output Structure Validation Script

This script validates that the SODA curation pipeline produces output
that exactly matches the expected structure from the example file.
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


def check_example_file():
    """Check if the example output file is available."""
    example_file = Path("tests/test_suite/test_data/EMBOJ-2024-119382.zip.json")

    if not example_file.exists():
        print("❌ Example output file not found!")
        print(f"Expected file: {example_file}")
        print("Please ensure the example output file is in the test_data directory.")
        return False

    print(f"✅ Example output file found: {example_file}")

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
    """Run pipeline output structure validation tests."""
    print("🧪 SODA Curation Pipeline - Output Structure Validation")
    print("=" * 60)
    print(
        "This script validates that the pipeline output matches the expected structure"
    )
    print(
        "from the example file: tests/test_suite/test_data/EMBOJ-2024-119382.zip.json"
    )
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Check example file
    if not check_example_file():
        sys.exit(1)

    # Define test commands focused on output structure validation
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
    ]

    # Run all tests
    results = []
    for test in test_commands:
        success = run_command(test["cmd"], test["description"])
        results.append((test["description"], success))

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")

    print(f"\nOverall: {passed}/{total} validation suites passed")

    if passed == total:
        print("🎉 All validation tests passed!")
        print("✅ The pipeline output structure matches the expected format.")
        print("✅ All required fields are present and correctly typed.")
        print("✅ All data validation rules are satisfied.")
        print("✅ The pipeline is ready for production use.")
        return 0
    else:
        print("⚠️  Some validation tests failed.")
        print("❌ The pipeline output structure does not match the expected format.")
        print("🔧 Please check the output above for details and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
