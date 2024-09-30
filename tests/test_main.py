"""
This module contains unit tests for the main functionality of the soda_curation package.

It tests various aspects of the main application, including command-line argument parsing,
configuration loading, ZIP file processing, and integration with different AI providers
(OpenAI and Anthropic) for caption extraction and structure analysis.
"""

import builtins
import json
import zipfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, mock_open, patch

import pytest

from soda_curation.main import main
from soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    ZipStructure,
)
from soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)


@pytest.fixture
def mock_argparse():
    """
    Fixture to mock the argparse.ArgumentParser.parse_args method.
    
    This allows us to simulate command-line arguments in tests without actually
    passing them from the command line.
    """
    with patch('argparse.ArgumentParser.parse_args') as mock_args:
        mock_args.return_value = Mock(zip='test.zip', config='config.yaml')
        yield mock_args

@pytest.fixture
def mock_load_config():
    """
    Fixture to mock the load_config function from the soda_curation.main module.
    
    This allows us to control the configuration returned in tests without reading
    from an actual configuration file.
    """
    with patch('soda_curation.main.load_config') as mock_config:
        mock_config.return_value = {
            'ai': 'openai',
            'openai': {
                'api_key': 'test_key',
                'model': 'gpt-4',
                'structure_zip_assistant_id': 'test_assistant_id'
            },
            'object_detection': {
                'model_path': 'test_model_path'
            }
        }
        yield mock_config

@pytest.fixture
def mock_zipfile():
    """
    Fixture to mock the zipfile.ZipFile class.
    
    This allows us to simulate operations on ZIP files without actually creating
    or reading real ZIP files during tests.
    """
    with patch('zipfile.ZipFile') as mock_zip:
        mock_zip.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']
        yield mock_zip

@pytest.fixture
def mock_json():
    """
    Fixture to mock the json.dumps function.
    
    This allows us to control JSON serialization in tests and verify that the
    correct data is being serialized.
    """
    with patch('json.dumps') as mock_dumps:
        yield mock_dumps

@pytest.fixture
def mock_openai_client():
    """
    Fixture to mock the OpenAI client.
    
    This allows us to simulate interactions with the OpenAI API without making actual API calls.
    """
    with patch('soda_curation.pipeline.extract_captions.extract_captions_openai.openai.OpenAI') as mock_client:
        mock_client.return_value.beta.assistants.update.return_value = MagicMock()
        mock_client.return_value.beta.assistants.create.return_value = MagicMock()
        yield mock_client

@pytest.fixture
def mock_anthropic_client():
    """
    Fixture to mock the Anthropic client.
    
    This allows us to simulate interactions with the Anthropic API without making actual API calls.
    """
    with patch('soda_curation.pipeline.extract_captions.extract_captions_anthropic.Anthropic') as mock_client:
        mock_client.return_value.messages.create.return_value = MagicMock()
        yield mock_client

@pytest.fixture
def mock_os():
    """
    Fixture to mock various os module functions.
    
    This allows us to simulate file system operations without affecting the actual file system.
    """
    with patch('os.path.exists', return_value=True), \
         patch('os.remove'), \
         patch('os.rmdir'), \
         patch('os.walk', return_value=[]):
        yield

@pytest.fixture
def mock_setup_logging():
    """
    Fixture to mock the setup_logging function.
    
    This allows us to test logging setup without actually configuring the logger.
    """
    with patch('soda_curation.main.setup_logging') as mock_logging:
        yield mock_logging

@pytest.fixture
def mock_processors():
    """
    Fixture to mock various processor classes used in the main module.
    
    This allows us to simulate the behavior of different processing components
    without instantiating them or running their actual logic.
    """
    with patch('soda_curation.main.XMLStructureExtractor') as mock_gpt, \
         patch('soda_curation.main.FigureCaptionExtractorGpt') as mock_caption_extractor, \
         patch('soda_curation.main.create_object_detection') as mock_object_detection:
        yield mock_gpt, mock_caption_extractor, mock_object_detection

@pytest.fixture
def mock_extract_captions():
    """
    Fixture to mock the caption extraction classes.
    
    This fixture provides mock objects for both OpenAI and Anthropic
    caption extraction classes.
    """
    with patch('soda_curation.main.FigureCaptionExtractorGpt') as mock_gpt, \
         patch('soda_curation.main.FigureCaptionExtractorClaude') as mock_claude:
        yield mock_gpt, mock_claude

@pytest.fixture
def mock_torch_load():
    with patch('torch.load') as mock_load:
        mock_load.return_value = {'model': MagicMock()}
        yield mock_load

def test_main_success(mock_argparse, mock_load_config, mock_zipfile, mock_json, mock_openai_client, mock_os, mock_torch_load):
    """
    Test the successful execution of the main function.

    This test verifies that when given valid inputs and configurations,
    the main function correctly processes a ZIP file and outputs the result.
    It checks that:
    1. The correct AI processor (OpenAI in this case) is instantiated.
    2. The ZIP file is correctly processed.
    3. The result is properly serialized to JSON.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml', output='test_output.json')
    mock_load_config.return_value = {
        'ai': 'openai',
        'openai': {
            'api_key': 'test',
            'model': 'gpt-4',
            'caption_extraction_assistant_id': 'test_assistant_id'
        },
        'object_detection': {
            'model_path': 'test_model_path'
        }
    }
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    expected_json = '{"test": "data"}'
    mock_json.return_value = expected_json

    with patch('soda_curation.main.Path.is_file', return_value=True), \
         patch('soda_curation.main.XMLStructureExtractor') as mock_extractor, \
         patch('os.listdir', return_value=['file1', 'file2']), \
         patch('builtins.open', mock_open(read_data=b"test content")), \
         patch('gettext.translation') as mock_translation, \
         patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock_yolo, \
         patch.object(Path, 'exists', return_value=True), \
         patch('openai._base_client.platform_headers', return_value={}):

        mock_translation.return_value.gettext = lambda x: x

        mock_extractor.return_value.extract_structure.return_value = ZipStructure(
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
        )
        mock_yolo.return_value = MagicMock()

        result = main('test.zip', 'config.yaml')

        assert result == expected_json
        mock_extractor.assert_called_once()
        mock_extractor.return_value.extract_structure.assert_called_once()
        mock_json.assert_called_once()

def test_main_anthropic(mock_argparse, mock_load_config, mock_zipfile, mock_json, mock_anthropic_client, mock_os):
    """
    Test the main function when using the Anthropic AI provider.

    This test verifies that when the configuration specifies Anthropic as the AI provider,
    the main function correctly uses the StructureZipFileClaude processor.
    It checks that:
    1. The correct AI processor (Anthropic Claude in this case) is instantiated.
    2. The ZIP file is correctly processed.
    3. The result is properly serialized to JSON.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml', output='test_output.json')
    mock_load_config.return_value = {
        'ai': 'anthropic',
        'anthropic': {
            'api_key': 'test',
            'model': 'claude-v1'
        },
        'object_detection': {
            'model_path': 'test_model_path'
        },
        'openai': {
            'api_key': 'dummy_key'
        }
    }
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    expected_json = '{"test": "data"}'
    mock_json.return_value = expected_json

    with patch('soda_curation.main.Path.is_file', return_value=True), \
         patch('soda_curation.main.XMLStructureExtractor') as mock_extractor, \
         patch('os.listdir', return_value=['file1', 'file2']), \
         patch('builtins.open', mock_open(read_data=b"test content")), \
         patch('gettext.translation') as mock_translation, \
         patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock_yolo, \
         patch.object(Path, 'exists', return_value=True), \
         patch('soda_curation.main.FigureCaptionExtractorClaude') as mock_claude:

        mock_translation.return_value.gettext = lambda x: x

        mock_extractor.return_value.extract_structure.return_value = ZipStructure(
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
        )
        mock_yolo.return_value = MagicMock()
        mock_claude.return_value.extract_captions.return_value = mock_extractor.return_value.extract_structure.return_value

        result = main('test.zip', 'config.yaml')

        assert result == expected_json
        mock_extractor.assert_called_once()
        mock_claude.assert_called_once()
        mock_json.assert_called_once()

# In test_main.py
def test_main_invalid_ai_provider(mock_argparse, mock_load_config, mock_zipfile):
    """
    Test the main function's behavior with an invalid AI provider.

    This test verifies that when given an invalid AI provider in the configuration,
    the main function raises a ValueError.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'invalid_provider'}
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with pytest.raises(ValueError, match="Invalid AI provider: invalid_provider"):
            main('test.zip', 'config.yaml')

def test_main_missing_arguments(mock_argparse):
    """
    Test the argument parsing when required command-line arguments are missing.

    This test verifies that when the required --zip and --config arguments are not provided,
    the main function raises a ValueError.
    """
    with pytest.raises(ValueError, match="ZIP path and config path must be provided"):
        main('', '')  # Pass empty strings to simulate missing arguments

def test_main_zip_file_not_found(mock_argparse, mock_load_config):
    """
    Test the main function's behavior when the specified ZIP file is not found.

    This test verifies that when the ZIP file specified in the command-line arguments
    does not exist, the main function raises a FileNotFoundError.
    """
    mock_argparse.return_value = Mock(zip='nonexistent.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}

    with patch('soda_curation.main.Path.is_file', return_value=False), \
         patch('zipfile.ZipFile', side_effect=FileNotFoundError("No such file or directory: 'nonexistent.zip'")):
        with pytest.raises(FileNotFoundError):
            main('nonexistent.zip', 'config.yaml')


def test_main_invalid_zip_file(mock_argparse, mock_load_config, mock_zipfile):
    """
    Test the main function's behavior when the specified ZIP file is invalid.

    This test verifies that when the ZIP file specified in the command-line arguments
    is not a valid ZIP archive, the main function raises a zipfile.BadZipFile exception.
    """
    mock_argparse.return_value = Mock(zip='invalid.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}
    mock_zipfile.side_effect = zipfile.BadZipFile()

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with pytest.raises(zipfile.BadZipFile):
            main('invalid.zip', 'config.yaml')

def test_main_processing_failure(mock_argparse, mock_load_config, mock_zipfile, mock_openai_client, mock_os):
    """
    Test the main function's behavior when ZIP file processing fails.

    This test verifies that when the AI processor fails to process the ZIP file structure
    (returning None), the main function raises a ValueError.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {
        'ai': 'openai',
        'openai': {
            'api_key': 'test',
            'model': 'gpt-4'
        }
    }
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True), \
         patch('soda_curation.main.get_manuscript_structure') as mock_get_structure:
        mock_get_structure.return_value = ZipStructure(errors=["Failed to process ZIP structure"])

        result = main('test.zip', 'config.yaml')
        assert "Failed to process ZIP structure" in result

def test_main_with_caption_extraction_docx_success(mock_argparse, mock_load_config, mock_zipfile, mock_extract_captions, mock_json, mock_openai_client, mock_os, mock_torch_load):
    """
    Test successful caption extraction from a DOCX file in the main function.

    This test verifies that when processing a ZIP file with a DOCX document,
    the main function correctly extracts captions using the OpenAI model and updates the ZipStructure.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml', output='test_output.json')

    mock_load_config.return_value = {
        'ai': 'openai',
        'openai': {
            'api_key': 'test',
            'model': 'gpt-4'
        }
    }
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    mock_gpt, _ = mock_extract_captions
    mock_gpt.return_value.extract_captions.return_value = ZipStructure(
        manuscript_id="test",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[Figure("Figure 1", ["image1.png"], [], "Extracted caption from DOCX", [])]
    )

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.XMLStructureExtractor') as mock_structure_gpt:
            with patch('builtins.open', mock_open(read_data=b"test content")), \
                 patch('os.path.exists', return_value=True), \
                 patch('os.path.join', return_value='/mock/path/test.docx'):

                mock_structure_gpt.return_value.extract_structure.return_value = ZipStructure(
                    manuscript_id="test",
                    xml="test.xml",
                    docx="test.docx",
                    pdf="test.pdf",
                    appendix=[],
                    figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
                )
                with patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock_yolo:
                    mock_yolo.return_value = MagicMock()
                    with patch('os.listdir', return_value=['file1', 'file2']):
                        result = main('test.zip', 'config.yaml')
                        assert isinstance(result, MagicMock)
                        mock_json.assert_called_once()
                        # Check if the mock was called with the expected arguments
                        expected_structure = ZipStructure(
                            manuscript_id="test",
                            xml="test.xml",
                            docx="test.docx",
                            pdf="test.pdf",
                            appendix=[],
                            figures=[Figure("Figure 1", ["image1.png"], [], "Extracted caption from DOCX", [])]
                        )
                        mock_json.assert_called_with(expected_structure, cls=ANY, ensure_ascii=False, indent=2)

def test_main_with_caption_extraction_pdf_fallback(mock_argparse, mock_load_config, mock_zipfile, mock_processors, mock_setup_logging):
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml', output='test_output.json')
    
    mock_gpt, mock_caption_extractor, mock_object_detection = mock_processors

    mock_load_config.return_value = {
        'ai': 'openai',
        'openai': {
            'api_key': 'test',
            'model': 'gpt-4'
        },
        'object_detection': {
            'model_path': 'test_model_path'
        }
    }

    initial_structure = ZipStructure(
        manuscript_id="test",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
    )

    # Define MockPath class
    class MockPath:
        def __init__(self, path):
            self.path = path

        def __truediv__(self, other):
            return MockPath(f"{self.path}/{other}")

        def is_file(self):
            return True

        @property
        def parent(self):
            return MockPath('/mock/extract_dir')

        @property
        def stem(self):
            return 'test'

        def __str__(self):
            return self.path

    with patch('soda_curation.main.extract_figure_captions') as mock_extract_captions:
        mock_extract_captions.side_effect = [
            ZipStructure(
                manuscript_id="test",
                xml="test.xml",
                docx="test.docx",
                pdf="test.pdf",
                appendix=[],
                figures=[Figure("Figure 1", ["image1.png"], [], "Figure caption not found.", [])]
            ),
            ZipStructure(
                manuscript_id="test",
                xml="test.xml",
                docx=None,
                pdf="test.pdf",
                appendix=[],
                figures=[Figure("Figure 1", ["image1.png"], [], "Extracted caption from PDF", [])]
            )
        ]

        with patch('soda_curation.main.Path', MockPath), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.join', side_effect=lambda *args: '/'.join(str(arg) for arg in args)), \
             patch('json.dumps') as mock_json_dumps, \
             patch('os.walk', return_value=[('/mock/extract_dir', [], ['file1', 'file2'])]), \
             patch('os.remove'), \
             patch('os.rmdir'), \
             patch('builtins.open', mock_open(read_data=b"test content")), \
             patch('os.listdir', return_value=['file1', 'file2']), \
             patch('soda_curation.main.XMLStructureExtractor') as mock_extractor, \
             patch('zipfile.ZipFile'), \
             patch('soda_curation.main.get_manuscript_structure', return_value=initial_structure):

            mock_extractor.return_value.extract_structure.return_value = initial_structure

            result = main('test.zip', 'config.yaml')

        assert mock_extract_captions.call_count == 2, f"extract_figure_captions was called {mock_extract_captions.call_count} times, expected 2"

        assert isinstance(result, str)
        parsed_result = json.loads(result)
        assert parsed_result['manuscript_id'] == "test"
        assert parsed_result['docx'] is None, f"Expected docx to be None, but got {parsed_result['docx']}"
        assert parsed_result['pdf'] == "test.pdf"
        assert parsed_result['figures'][0]['figure_caption'] == "Extracted caption from PDF"

    mock_json_dumps.assert_called_with(ANY, cls=ANY, ensure_ascii=False, indent=2)
    
    actual_structure = mock_json_dumps.call_args[0][0]
    assert actual_structure.manuscript_id == "test"
    assert actual_structure.xml == "test.xml"
    assert actual_structure.docx is None, f"Expected docx to be None, but got {actual_structure.docx}"
    assert actual_structure.pdf == "test.pdf"
    assert len(actual_structure.figures) == 1
    assert actual_structure.figures[0].figure_caption == "Extracted caption from PDF"
    
def test_main_with_caption_extraction_docx_only(mock_argparse, mock_load_config, mock_zipfile, mock_extract_captions, mock_json, mock_openai_client, mock_os, mock_torch_load):
    """
    Test caption extraction when only a DOCX file is available.

    This test verifies that the main function correctly handles caption extraction
    when only a DOCX file is present in the ZIP structure (no PDF fallback).
    It ensures that:
    1. The system attempts to extract captions from the DOCX file.
    2. The extraction process is called only once (no fallback to PDF).
    3. The final ZipStructure contains captions extracted from the DOCX file.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml', output='test_output.json')
    mock_load_config.return_value = {
        'ai': 'openai',
        'openai': {
            'api_key': 'test',
            'model': 'gpt-4'
        }
    }
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    mock_gpt, _ = mock_extract_captions
    mock_gpt.return_value.extract_captions.return_value = ZipStructure(
        manuscript_id="test",
        xml="test.xml",
        docx="test.docx",
        pdf=None,
        appendix=[],
        figures=[Figure("Figure 1", ["image1.png"], [], "Extracted caption from DOCX", [])]
    )

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.XMLStructureExtractor') as mock_structure_gpt:
            with patch('builtins.open', mock_open(read_data=b"test content")), \
                 patch('os.path.exists', return_value=True), \
                 patch('os.path.join', return_value='/mock/path/test.docx'):

                mock_structure_gpt.return_value.extract_structure.return_value = ZipStructure(
                    manuscript_id="test",
                    xml="test.xml",
                    docx="test.docx",
                    pdf=None,
                    appendix=[],
                    figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
                )
                with patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock_yolo:
                    mock_yolo.return_value = MagicMock()
                    with patch('os.listdir', return_value=['file1', 'file2']):
                        result = main('test.zip', 'config.yaml')
                        assert isinstance(result, MagicMock)
                        mock_json.assert_called_once()
                        # Check if the mock was called with the expected arguments
                        expected_structure = ZipStructure(
                            manuscript_id="test",
                            xml="test.xml",
                            docx="test.docx",
                            pdf=None,
                            appendix=[],
                            figures=[Figure("Figure 1", ["image1.png"], [], "Extracted caption from DOCX", [])]
                        )
                        mock_json.assert_called_with(expected_structure, cls=ANY, ensure_ascii=False, indent=2)
