import pytest
import yaml
from pathlib import Path
from soda_curation.config import load_config

@pytest.fixture
def sample_config_file(tmp_path):
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
    config = load_config(str(sample_config_file))
    assert config['ai'] == 'anthropic'
    assert config['anthropic']['api_key'] == 'test_key'
    assert config['openai']['model'] == 'gpt-4-1106-preview'

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config('non_existent_config.yaml')

def test_load_config_invalid_yaml(tmp_path):
    invalid_config = tmp_path / "invalid_config.yaml"
    invalid_config.write_text("invalid: yaml: content")
    with pytest.raises(yaml.YAMLError):
        load_config(str(invalid_config))

def test_load_config_missing_required_fields(tmp_path):
    incomplete_config = tmp_path / "incomplete_config.yaml"
    incomplete_config.write_text("ai: anthropic")
    with pytest.raises(KeyError):
        config = load_config(str(incomplete_config))
        _ = config['anthropic']  # This should raise KeyError

def test_load_config_additional_fields(sample_config_file):
    config_content = sample_config_file.read_text()
    config_content += "\n---\nadditional_field: value"
    sample_config_file.write_text(config_content)
    config = load_config(str(sample_config_file))
    assert 'additional_field' in config
    assert config['additional_field'] == 'value'
    assert 'ai' in config  # Ensure original fields are still present
