import pytest
import yaml
from pathlib import Path
from soda_curation.config import load_config

@pytest.fixture
def sample_config_file(tmp_path):
    """
    Fixture to create a sample configuration file for testing.

    This fixture creates a temporary YAML file with a sample configuration
    that includes settings for both Anthropic and OpenAI.

    Args:
        tmp_path: pytest fixture providing a temporary directory unique to the test invocation.

    Returns:
        Path: Path object pointing to the created sample configuration file.
    """
    config_content = """
    ai: "anthropic"
    anthropic:
      api_key: "test_key"
      model: "claude-3-sonnet-20240229"
      temperature: 0.7
      max_tokens_to_sample: 8000
    openai:
      api_key: "test_openai_key"
      model: "gpt-4-1106-preview"
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file

def test_load_config_valid(sample_config_file):
    """
    Test loading a valid configuration file.

    This test verifies that the load_config function correctly loads and parses
    a valid YAML configuration file. It checks that:
    1. The correct AI provider is identified.
    2. The Anthropic API key is correctly loaded.
    3. The OpenAI model is correctly loaded.

    Args:
        sample_config_file: pytest fixture providing a sample configuration file.
    """
    config = load_config(str(sample_config_file))
    assert config['ai'] == 'anthropic'
    assert config['anthropic']['api_key'] == 'test_key'
    assert config['openai']['model'] == 'gpt-4-1106-preview'

def test_load_config_file_not_found():
    """
    Test the behavior when the configuration file is not found.

    This test verifies that the load_config function raises a FileNotFoundError
    when attempting to load a non-existent configuration file.
    """
    with pytest.raises(FileNotFoundError):
        load_config('non_existent_config.yaml')

def test_load_config_invalid_yaml(tmp_path):
    """
    Test the behavior when the configuration file contains invalid YAML.

    This test verifies that the load_config function raises a yaml.YAMLError
    when attempting to load a configuration file with invalid YAML syntax.

    Args:
        tmp_path: pytest fixture providing a temporary directory unique to the test invocation.
    """
    invalid_config = tmp_path / "invalid_config.yaml"
    invalid_config.write_text("invalid: yaml: content")
    with pytest.raises(yaml.YAMLError):
        load_config(str(invalid_config))

def test_load_config_missing_required_fields(tmp_path):
    """
    Test the behavior when the configuration file is missing required fields.

    This test verifies that when a configuration file is missing required fields
    (in this case, the 'anthropic' section), accessing those fields raises a KeyError.

    Args:
        tmp_path: pytest fixture providing a temporary directory unique to the test invocation.
    """
    incomplete_config = tmp_path / "incomplete_config.yaml"
    incomplete_config.write_text("ai: anthropic")
    with pytest.raises(KeyError):
        config = load_config(str(incomplete_config))
        _ = config['anthropic']  # This should raise KeyError

def test_load_config_additional_fields(sample_config_file):
    """
    Test loading a configuration file with additional fields.

    This test verifies that the load_config function correctly handles configuration
    files that contain additional, unexpected fields. It checks that:
    1. The additional field is present in the loaded configuration.
    2. The original fields are still present and correct.

    Args:
        sample_config_file: pytest fixture providing a sample configuration file.
    """
    config_content = sample_config_file.read_text()
    config_content += "\n---\nadditional_field: value"
    sample_config_file.write_text(config_content)
    config = load_config(str(sample_config_file))
    assert 'additional_field' in config
    assert config['additional_field'] == 'value'
    assert 'ai' in config  # Ensure original fields are still present
