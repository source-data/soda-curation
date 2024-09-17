import pytest
from unittest.mock import Mock, patch
from soda_curation.pipeline.zip_structure.zip_structure_anthropic import StructureZipFileClaude
from soda_curation.pipeline.zip_structure.zip_structure_base import ZipStructure

@pytest.fixture
def mock_anthropic_client():
    """
    Fixture to mock the Anthropic client.

    This fixture patches the Anthropic class to allow controlled testing
    of Anthropic API interactions without making actual API calls.

    Yields:
        Mock: A mock object representing the Anthropic client.
    """
    with patch('soda_curation.pipeline.zip_structure.zip_structure_anthropic.Anthropic') as mock_anthropic:
        yield mock_anthropic

@pytest.fixture
def sample_config():
    """
    Fixture to provide a sample configuration for StructureZipFileClaude.

    Returns:
        dict: A dictionary containing sample configuration values for Anthropic.
    """
    return {
        'api_key': 'test_key',
        'model': 'claude-3-sonnet-20240229',
        'temperature': 0.7,
        'max_tokens_to_sample': 8000,
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
def test_structure_zip_file_claude_initialization(sample_config):
    """
    Test the initialization of StructureZipFileClaude.

    This test verifies that StructureZipFileClaude is correctly initialized
    with the provided configuration.

    Args:
        sample_config (dict): Sample configuration dictionary.
    """
    claude = StructureZipFileClaude(sample_config)
    assert claude.config == sample_config

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_success(mock_anthropic_client, sample_config, sample_file_list):
    """
    Test successful processing of ZIP structure.

    This test simulates a successful ZIP structure processing scenario.
    It verifies that the process_zip_structure method correctly interacts
    with the Anthropic API and returns a valid ZipStructure object.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
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

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_invalid_json(mock_anthropic_client, sample_config, sample_file_list):
    """
    Test ZIP structure processing with invalid JSON response.

    This test simulates a scenario where the Anthropic API returns an invalid JSON response.
    It verifies that the process_zip_structure method correctly handles the invalid
    JSON and returns None.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
    mock_response = Mock()
    mock_response.content = "Invalid JSON"
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert result is None

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_missing_fields(mock_anthropic_client, sample_config, sample_file_list):
    """
    Test ZIP structure processing with missing fields in the JSON response.

    This test simulates a scenario where the Anthropic API returns a JSON response
    missing required fields. It verifies that the process_zip_structure method
    correctly handles the incomplete JSON and returns None.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
    mock_response = Mock()
    mock_response.content = '{"manuscript_id": "test"}'
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert result is None

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_api_error(mock_anthropic_client, sample_config, sample_file_list):
    """
    Test ZIP structure processing when an API error occurs.

    This test simulates a scenario where the Anthropic API call raises an exception.
    It verifies that the process_zip_structure method correctly handles the
    exception and returns None.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
    """
    mock_anthropic_client.return_value.messages.create.side_effect = Exception("API Error")

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert result is None

@pytest.mark.usefixtures("capsys")
def test_process_zip_structure_anthropic_error(mock_anthropic_client, sample_config, sample_file_list, capsys):
    """
    Test ZIP structure processing when a specific Anthropic API error occurs.

    This test simulates a scenario where the Anthropic API call raises a specific
    Anthropic API error. It verifies that the process_zip_structure method correctly
    handles the error, prints an appropriate error message, and returns None.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_config (dict): Sample configuration dictionary.
        sample_file_list (list): Sample list of files in the ZIP.
        capsys: Pytest fixture to capture stdout and stderr.
    """
    mock_anthropic_client.return_value.messages.create.side_effect = Exception("Anthropic API Error")

    claude = StructureZipFileClaude(sample_config)
    result = claude.process_zip_structure(sample_file_list)

    assert result is None
    captured = capsys.readouterr()
    assert "Error in AI processing: Anthropic API Error" in captured.err  # Check stderr instead of stdout
