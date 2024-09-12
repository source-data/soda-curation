import pytest
from unittest.mock import Mock, patch
from soda_curation.ai_modules.openai_module import StructureZipFileGPT
from soda_curation.ai_modules.general import StructureZipFile, ZipStructure, Figure

@pytest.fixture
def mock_openai_client():
    with patch('soda_curation.ai_modules.openai_module.openai.Client') as mock_client:
        yield mock_client

@pytest.fixture
def sample_config():
    return {
        'api_key': 'test_key',
        'model': 'gpt-4-1106-preview',
        'temperature': 0.7,
        'top_p': 1.0,
        'structure_zip_assistant_id': 'test_assistant_id'
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

def test_structure_zip_file_gpt_initialization(mock_openai_client, sample_config):
    gpt = StructureZipFileGPT(sample_config)
    assert gpt.config == sample_config
    mock_openai_client.assert_called_once_with(api_key='test_key')

def test_process_zip_structure_success(mock_openai_client, sample_config, sample_file_list):
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value="""
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
    """))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert isinstance(result, ZipStructure)
    assert result.manuscript_id == "test_manuscript"
    assert result.xml == "manuscript.xml"
    assert len(result.figures) == 1
    assert result.figures[0].figure_label == "Figure 1"

def test_process_zip_structure_run_failed(mock_openai_client, sample_config, sample_file_list):
    mock_run = Mock()
    mock_run.status = 'failed'
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result == 'failed'

def test_process_zip_structure_invalid_json(mock_openai_client, sample_config, sample_file_list):
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value="Invalid JSON"))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result is None

def test_process_zip_structure_missing_fields(mock_openai_client, sample_config, sample_file_list):
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value='{"manuscript_id": "test"}'))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result is None

def test_process_zip_structure_api_error(mock_openai_client, sample_config, sample_file_list):
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.side_effect = Exception("API Error")

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result is None

def test_custom_prompt_instructions(mock_openai_client, sample_config):
    sample_config['custom_prompt_instructions'] = "Custom instructions here"
    gpt = StructureZipFileGPT(sample_config)

    update_call = mock_openai_client.return_value.beta.assistants.update.call_args
    assert "Custom instructions here" in update_call[1]['instructions']
