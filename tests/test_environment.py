import os
import pytest

def test_required_env_vars():
    """Tests if the required environment variables are set"""
    required_vars = [
        'ANTHROPIC_API_KEY',
        'PANELIZATION_MODEL'
    ]
    for var in required_vars:
        assert var in os.environ, f"{var} is not set in the environment"

def test_optional_env_vars():
    """Tests if the optional environment variables are set and prints a message if they are"""
    optional_vars = [
        'OPENAI_ORG_KEY',
        'OPENAI_API_KEY'
    ]
    for var in optional_vars:
        if var in os.environ:
            print(f"Optional environment variable {var} is set")
