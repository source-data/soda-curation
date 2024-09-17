import pytest
from unittest.mock import Mock, patch
from soda_curation.pipeline.zip_structure.zip_structure_openai import StructureZipFileGPT
from soda_curation.pipeline.zip_structure.zip_structure_base import StructureZipFile, ZipStructure, Figure
import logging

@pytest.fixture
def mock_openai_client():
    """
    Fixture to mock the OpenAI client.

    This fixture patches the openai.Client class to allow controlled testing
    of OpenAI API interactions without making actual API calls.

    Yields:
        Mock: A mock object representing the OpenAI client.
    """
    with patch('soda_curation.pipeline.zip_structure.zip_structure_openai.openai.Client') as mock_client:
        yield mock_client

@pytest.fixture
def sample_config():
    """
    Fixture to provide a sample configuration for StructureZipFileGPT.

    Returns:
        dict: A dictionary containing sample configuration values for OpenAI.
    """
    return {
        'api_key': 'test_key',
        'model': 'gpt-4-1106-preview',
        'temperature': 0.7,
        'top_p': 1.0,
        'structure_zip_assistant_id': 'test_assistant_id'
    }

@pytest.fixture
def sample_file_list():
    """
    Fixture to provide a sample list of files for testing.

    Returns:
        list: A list of sample file paths simulating a ZIP file structure.
    """
    return [
        'manuscript.xml',
        'doc/manuscript.docx',
        'pdf/manuscript.pdf',
        'graphic/Figure1.eps',
        'suppl_data/Figure1_source.xlsx'
    ]

@pytest.mark.usefixtures("capsys")
def test_structure_zip_file_gpt_initialization(mock_openai_client, sample_config):
    """
    Test the initialization of StructureZipFileGPT.

    This test verifies that StructureZipFileGPT is correctly initialized
    with the provided configuration and that the OpenAI client is created
    with the correct API key.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
    """
    gpt = StructureZipFileGPT(sample_config)
    assert gpt.config == sample_config
    mock_openai_client.assert_called_once_with(api_key='test_key')

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_success(mock_openai_client, sample_config, sample_file_list):
    """
    Test successful processing of ZIP structure.

    This test simulates a successful ZIP structure processing scenario.
    It verifies that the process_zip_structure method correctly interacts
    with the OpenAI API and returns a valid ZipStructure object.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
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

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_run_failed(mock_openai_client, sample_config, sample_file_list):
    """
    Test ZIP structure processing when the OpenAI run fails.

    This test simulates a scenario where the OpenAI API run fails.
    It verifies that the process_zip_structure method correctly handles
    the failure and returns the appropriate status.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
    mock_run = Mock()
    mock_run.status = 'failed'
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result == 'failed'

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_invalid_json(mock_openai_client, sample_config, sample_file_list):
    """
    Test ZIP structure processing with invalid JSON response.

    This test simulates a scenario where the OpenAI API returns an invalid JSON response.
    It verifies that the process_zip_structure method correctly handles the invalid
    JSON and returns None.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value="Invalid JSON"))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result is None

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_missing_fields(mock_openai_client, sample_config, sample_file_list):
    """
    Test ZIP structure processing with missing fields in the JSON response.

    This test simulates a scenario where the OpenAI API returns a JSON response
    missing required fields. It verifies that the process_zip_structure method
    correctly handles the incomplete JSON and returns None.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value='{"manuscript_id": "test"}'))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result is None

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_api_error(mock_openai_client, sample_config, sample_file_list):
    """
    Test ZIP structure processing when an API error occurs.

    This test simulates a scenario where the OpenAI API call raises an exception.
    It verifies that the process_zip_structure method correctly handles the
    exception and returns None.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.side_effect = Exception("API Error")

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result is None

@pytest.mark.usefixtures("capsys")
def test_custom_prompt_instructions(mock_openai_client, sample_config):
    """
    Test custom prompt instructions in StructureZipFileGPT initialization.

    This test verifies that when custom prompt instructions are provided in the
    configuration, they are correctly included in the assistant update call.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
    """
    sample_config['custom_prompt_instructions'] = "Custom instructions here"
    gpt = StructureZipFileGPT(sample_config)

    update_call = mock_openai_client.return_value.beta.assistants.update.call_args
    assert "Custom instructions here" in update_call[1]['instructions']

@pytest.mark.usefixtures("caplog")
def test_process_zip_structure_openai_error(mock_openai_client, sample_config, sample_file_list, caplog):
    """
    Test ZIP structure processing when a specific OpenAI API error occurs.

    This test simulates a scenario where the OpenAI API call raises a specific
    OpenAI API error. It verifies that the process_zip_structure method correctly
    handles the error, logs an appropriate error message, and returns None.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
        caplog: Pytest fixture to capture log messages.
    """
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.side_effect = Exception("OpenAI API Error")

    caplog.set_level(logging.ERROR)

    gpt = StructureZipFileGPT(sample_config)
    result = gpt.process_zip_structure(sample_file_list)

    assert result is None
    assert "Error in AI processing: OpenAI API Error" in caplog.text
