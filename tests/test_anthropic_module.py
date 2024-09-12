import pytest
from unittest.mock import Mock, patch
from soda_curation.ai_modules.anthropic_module import StructureZipFileClaude
from soda_curation.ai_modules.general import StructureZipFile, ZipStructure, Figure

@pytest.fixture
def mock_anthropic_client():
    with patch('soda_curation.ai_modules.anthropic_module.Anthropic') as mock_anthropic:
        yield mock_anthropic

@pytest.fixture
def sample_config():
    return {
        'api_key': 'test_key',
        'model': 'claude-3-sonnet-20240229',
        'temperature': 0.7,
        'max_tokens_to_sample': 8000,
    }

@pytest.fixture
def sample_file_list():
    return [
        'manuscript.xml',
        'doc/manuscript.docx',
        'pdf/manuscript.pdf',
        'graphic/Figure1.eps',
        'suppl_data/Figure1_source.xlsx'
    ]

def test_structure_zip_file_claude_initialization(sample_config):
    claude = StructureZipFileClaude(sample_config)
    assert claude.config == sample_config

def test_process_zip_structure_success(mock_anthropic_client, sample_config, sample_file_list):
    mock_response = Mock()
    mock_response.content = """
    {
        "manuscript_id": "test_manuscript",
        "xml": "manuscript.xml",
        "docx": "doc/manuscript.docx",
        "pdf": "pdf/manuscript.pdf",
        "appendix": [],
        "figures": [
            {
                "figure_label": "Figure 1",
                "img_files": ["graphic/Figure1.eps"],
                "sd_files": ["suppl_data/Figure1_source.xlsx"],
                "figure_caption": "TO BE ADDED IN LATER STEP",
                "figure_panels": []
            }
        ]
    }
    """
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert isinstance(result, ZipStructure)
    assert result.manuscript_id == "test_manuscript"
    assert result.xml == "manuscript.xml"
    assert len(result.figures) == 1
    assert result.figures[0].figure_label == "Figure 1"

def test_process_zip_structure_invalid_json(mock_anthropic_client, sample_config, sample_file_list):
    mock_response = Mock()
    mock_response.content = "Invalid JSON"
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert result is None

def test_process_zip_structure_missing_fields(mock_anthropic_client, sample_config, sample_file_list):
    mock_response = Mock()
    mock_response.content = '{"manuscript_id": "test"}'
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert result is None

def test_process_zip_structure_api_error(mock_anthropic_client, sample_config, sample_file_list):
    mock_anthropic_client.return_value.messages.create.side_effect = Exception("API Error")

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert result is None

def test_custom_prompt_instructions(mock_anthropic_client, sample_config, sample_file_list):
    sample_config['custom_prompt_instructions'] = "Custom instructions here"
    claude = StructureZipFileClaude(sample_config)
    claude.process_zip_structure(sample_file_list)

    create_call = mock_anthropic_client.return_value.messages.create.call_args
    assert "Custom instructions here" in create_call[1]['messages'][0]['content']