import pytest
from unittest.mock import patch, MagicMock
from src.assistants import get_file_structure, extract_figure_captions

@pytest.fixture
def mock_anthropic():
    """
    Fixture to mock the Anthropic API client.
    This allows us to test without making actual API calls.
    """
    with patch('src.assistants.Anthropic') as mock:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(text='''{
                    "manuscript": {
                        "id_": "JOURNAL-2023-12345",
                        "files": ["/path/to/manuscript.docx"],
                        "xml": "/path/to/manuscript.xml",
                        "figures": [
                            {
                                "figure_label": "Figure 1",
                                "img_file": ["/path/to/figure1.tif"],
                                "sd_file": ["/path/to/figure1-sd.zip"]
                            }
                        ]
                    }
                }''')
            ]
        )
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def mock_docx():
    """
    Fixture to mock the docx.Document class.
    This allows us to simulate a DOCX file with predefined content.
    """
    with patch('src.assistants.docx.Document') as mock:
        mock.return_value.paragraphs = [
            MagicMock(text="Figure 1. Test Caption"),
            MagicMock(text="A. Subcaption A"),
            MagicMock(text="B. Subcaption B"),
            MagicMock(text="Figure 2. Another Caption"),
            MagicMock(text="Description of Figure 2")
        ]
        yield mock

@patch('os.path.exists')
def test_extract_figure_captions(mock_exists, mock_anthropic, mock_docx):
    """
    Test the extract_figure_captions function.
    
    This test checks if the function correctly extracts full figure captions,
    including main captions and sub-captions, from a mocked DOCX file.
    It verifies that the function returns a dictionary with the expected structure
    and content for multiple figures.
    """
    mock_exists.return_value = True  # Simulate that the file exists
    mock_anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text='''{
            "Figure 1": "Figure 1. Test Caption\\nA. Subcaption A\\nB. Subcaption B",
            "Figure 2": "Figure 2. Another Caption\\nDescription of Figure 2"
        }''')]
    )
    
    result = extract_figure_captions(["/path/to/manuscript.docx"], "fake_api_key")
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "Figure 1" in result
    assert "Figure 2" in result
    assert "Test Caption" in result["Figure 1"]
    assert "Subcaption A" in result["Figure 1"]
    assert "Subcaption B" in result["Figure 1"]
    assert "Another Caption" in result["Figure 2"]
    assert "Description of Figure 2" in result["Figure 2"]

@patch('os.path.exists')
def test_get_file_structure_with_captions(mock_exists, mock_anthropic, mock_docx):
    """
    Test the get_file_structure function with figure caption extraction.
    
    This test verifies that the get_file_structure function correctly processes
    a list of files, extracts figure information, and includes full figure captions
    in the resulting structure. It checks for the presence of multiple figures
    with their respective captions and sub-captions.
    """
    mock_exists.return_value = True  # Simulate that the file exists
    file_list = [
        "/path/to/manuscript.docx",
        "/path/to/figure1.tif",
        "/path/to/figure1-sd.zip"
    ]
    
    mock_anthropic.return_value.messages.create.side_effect = [
            MagicMock(content=[MagicMock(text='''{
                "manuscript": {
                    "id_": "JOURNAL-2023-12345",
                    "files": ["/path/to/manuscript.docx"],
                    "xml": "/path/to/manuscript.xml",
                    "figures": [
                        {
                            "figure_label": "Figure 1",
                            "img_file": ["/path/to/figure1.tif"],
                            "sd_file": ["/path/to/figure1-sd.zip"]
                        },
                        {
                            "figure_label": "Figure 2",
                            "img_file": ["/path/to/figure2.tif"],
                            "sd_file": ["/path/to/figure2-sd.zip"]
                        }
                    ]
                }
            }''')]),
            MagicMock(content=[MagicMock(text='''{
                "Figure 1": "Figure 1. Test Caption\\nA. Subcaption A\\nB. Subcaption B",
                "Figure 2": "Figure 2. Another Caption\\nDescription of Figure 2"
            }''')]
        )
    ]

    result = get_file_structure(file_list)
    
    assert isinstance(result, dict)
    assert "manuscript" in result
    assert result["manuscript"]["id_"] == "JOURNAL-2023-12345"
    assert len(result["manuscript"]["figures"]) == 2
    assert "figure_caption" in result["manuscript"]["figures"][0]
    assert "Test Caption" in result["manuscript"]["figures"][0]["figure_caption"]
    assert "Subcaption A" in result["manuscript"]["figures"][0]["figure_caption"]
    assert "Subcaption B" in result["manuscript"]["figures"][0]["figure_caption"]
    assert "figure_caption" in result["manuscript"]["figures"][1]
    assert "Another Caption" in result["manuscript"]["figures"][1]["figure_caption"]
    assert "Description of Figure 2" in result["manuscript"]["figures"][1]["figure_caption"]

@pytest.mark.parametrize("env_var", ["ANTHROPIC_API_KEY"])
def test_required_env_vars(env_var):
    """
    Test if required environment variables are set.
    
    This test checks if the necessary environment variables (in this case, ANTHROPIC_API_KEY)
    are set in the system environment. It's crucial for ensuring that the application
    has access to required API keys or configuration values.
    """
    import os
    assert env_var in os.environ, f"{env_var} is not set in the environment"

def test_empty_file_list():
    """
    Test the behavior of get_file_structure with an empty file list.
    
    This test verifies that the get_file_structure function returns an empty dictionary
    when given an empty list of files. It ensures the function handles edge cases gracefully.
    """
    result = get_file_structure([])
    assert result == {}

@patch('src.assistants.Anthropic')
def test_api_error(mock_anthropic):
    """
    Test the behavior of get_file_structure when an API error occurs.
    
    This test simulates an API error and checks if the get_file_structure function
    handles the error gracefully by returning an empty dictionary. It's important
    for ensuring robustness in case of external service failures.
    """
    mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
    result = get_file_structure(["file1.txt"])
    assert result == {}