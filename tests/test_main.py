"""
This module contains unit tests for the main functionality of the soda_curation package.

It tests various aspects of the main application, including command-line argument parsing,
configuration loading, ZIP file processing, and integration with different AI providers
(OpenAI and Anthropic) for caption extraction and structure analysis.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
import zipfile
from soda_curation.main import main
import json
from soda_curation.pipeline.zip_structure.zip_structure_base import ZipStructure, Figure

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
    with patch('soda_curation.main.StructureZipFileGPT') as mock_gpt, \
         patch('soda_curation.main.FigureCaptionExtractorGpt') as mock_caption_extractor, \
         patch('soda_curation.main.create_object_detection') as mock_object_detection:
        yield mock_gpt, mock_caption_extractor, mock_object_detection

def test_main_success(mock_argparse, mock_load_config, mock_zipfile, mock_json, mock_openai_client, mock_os):
    """
    Test the successful execution of the main function.

    This test verifies that when given valid inputs and configurations,
    the main function correctly processes a ZIP file and outputs the result.
    It checks that:
    1. The correct AI processor (OpenAI in this case) is instantiated.
    2. The ZIP file is correctly processed.
    3. The result is properly serialized to JSON.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {
        'ai': 'openai', 
        'openai': {
            'api_key': 'test',
            'model': 'gpt-4',
            'caption_extraction_assistant_id': 'test_assistant_id'
        }
    }
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.StructureZipFileGPT') as mock_gpt:
            mock_gpt.return_value.process_zip_structure.return_value = ZipStructure(
                manuscript_id="test",
                xml="test.xml",
                docx="test.docx",
                pdf="test.pdf",
                appendix=[],
                figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
            )
            mock_json.return_value = '{"result": "success"}'

            main()

            mock_gpt.assert_called_once()
            mock_gpt.return_value.process_zip_structure.assert_called_once_with(['file1', 'file2'])
            mock_json.assert_called()

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
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {
        'ai': 'anthropic', 
        'anthropic': {
            'api_key': 'test',
            'model': 'claude-v1'
        }
    }
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.StructureZipFileClaude') as mock_claude:
            mock_claude.return_value.process_zip_structure.return_value = ZipStructure(
                manuscript_id="test",
                xml="test.xml",
                docx="test.docx",
                pdf="test.pdf",
                appendix=[],
                figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
            )
            mock_json.return_value = '{"result": "success"}'

            main()

            mock_claude.assert_called_once()
            mock_claude.return_value.process_zip_structure.assert_called_once_with(['file1', 'file2'])
            mock_json.assert_called()

def test_main_invalid_ai_provider(mock_argparse, mock_load_config, mock_zipfile):
    """
    Test the main function's behavior with an invalid AI provider.
    
    This test verifies that when given an invalid AI provider in the configuration,
    the main function exits with a system error.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'invalid_provider'}
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with pytest.raises(SystemExit):
            main()

def test_main_missing_arguments(mock_argparse):
    """
    Test the main function's behavior when required command-line arguments are missing.
    
    This test verifies that when the required --zip and --config arguments are not provided,
    the main function exits with a system error.
    """
    mock_argparse.return_value = Mock(zip=None, config=None)

    with pytest.raises(SystemExit):
        main()

def test_main_zip_file_not_found(mock_argparse, mock_load_config):
    """
    Test the main function's behavior when the specified ZIP file is not found.
    
    This test verifies that when the ZIP file specified in the command-line arguments
    does not exist, the main function exits with a system error.
    """
    mock_argparse.return_value = Mock(zip='nonexistent.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}

    with patch('soda_curation.main.Path.is_file', return_value=False):
        with pytest.raises(SystemExit):
            main()

def test_main_invalid_zip_file(mock_argparse, mock_load_config, mock_zipfile):
    """
    Test the main function's behavior when the specified ZIP file is invalid.
    
    This test verifies that when the ZIP file specified in the command-line arguments
    is not a valid ZIP archive, the main function exits with a system error.
    """
    mock_argparse.return_value = Mock(zip='invalid.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}
    mock_zipfile.side_effect = zipfile.BadZipFile()

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with pytest.raises(SystemExit):
            main()

def test_main_processing_failure(mock_argparse, mock_load_config, mock_zipfile, mock_openai_client, mock_os):
    """
    Test the main function's behavior when ZIP file processing fails.
    
    This test verifies that when the AI processor fails to process the ZIP file structure
    (returning None), the main function exits with a system error.
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

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.StructureZipFileGPT') as mock_gpt:
            mock_gpt.return_value.process_zip_structure.return_value = None

            with pytest.raises(SystemExit):
                main()

def test_main_with_caption_extraction_docx_success(mock_argparse, mock_load_config, mock_zipfile, mock_extract_captions, mock_json, mock_openai_client, mock_os):
    """
    Test successful caption extraction from a DOCX file in the main function.

    This test verifies that when processing a ZIP file with a DOCX document,
    the main function correctly extracts captions using the OpenAI model and updates the ZipStructure.
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
        with patch('soda_curation.main.StructureZipFileGPT') as mock_structure_gpt:
            mock_structure_gpt.return_value.process_zip_structure.return_value = ZipStructure(
                manuscript_id="test",
                xml="test.xml",
                docx="test.docx",
                pdf="test.pdf",
                appendix=[],
                figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
            )
            main()

        mock_structure_gpt.assert_called_once()
        mock_gpt.return_value.extract_captions.assert_called_once()
        assert mock_gpt.return_value.extract_captions.call_args[0][0].endswith('test.docx')

def test_main_with_caption_extraction_pdf_fallback(mock_argparse, mock_load_config, mock_zipfile, mock_processors, mock_setup_logging):
    """
    Test caption extraction fallback to PDF when DOCX extraction fails.

    This test verifies that when caption extraction from a DOCX file fails or produces no results,
    the main function falls back to extracting captions from the PDF file. It ensures that:
    1. The system attempts to extract captions from the DOCX file first.
    2. If DOCX extraction fails, it proceeds to extract captions from the PDF.
    3. The final ZipStructure contains captions extracted from the PDF.
    4. Proper cleanup operations are performed after extraction.

    Args:
        mock_argparse: Mocked command-line argument parser.
        mock_load_config: Mocked configuration loader.
        mock_zipfile: Mocked ZIP file handler.
        mock_processors: Mocked processing components (GPT, caption extractor, object detection).
        mock_setup_logging: Mocked logging setup.
    """
    mock_gpt, mock_caption_extractor, mock_object_detection = mock_processors
    
    # Mock the zip structure
    mock_zip_structure = ZipStructure(
        manuscript_id="test",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
    )
    
    mock_gpt.return_value.process_zip_structure.return_value = mock_zip_structure
    
    # Mock the caption extraction results
    mock_caption_extractor.return_value.extract_captions.side_effect = [
        ZipStructure(  # DOCX extraction result (no captions found)
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[Figure("Figure 1", ["image1.png"], [], "Figure caption not found.", [])]
        ),
        ZipStructure(  # PDF extraction result
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[Figure("Figure 1", ["image1.png"], [], "Extracted caption from PDF", [])]
        )
    ]
    
    # Mock object detection
    mock_object_detection.return_value.detect_panels.return_value = []
    
    with patch('soda_curation.main.Path.is_file', return_value=True), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.join', return_value='/mock/path') as mock_join, \
         patch('json.dumps') as mock_json_dumps, \
         patch('os.walk') as mock_walk, \
         patch('os.remove') as mock_remove, \
         patch('os.rmdir') as mock_rmdir:
        
        # Mock the os.walk to return some fake directory structure
        mock_walk.return_value = [
            ('/mock/path', ['dir1', 'dir2'], ['file1', 'file2']),
            ('/mock/path/dir1', [], ['file3']),
            ('/mock/path/dir2', [], ['file4'])
        ]
        
        main()
    
    # Assert that extract_captions was called twice
    assert mock_caption_extractor.return_value.extract_captions.call_count == 2
    
    # Check the calls to extract_captions
    calls = mock_caption_extractor.return_value.extract_captions.call_args_list
    assert calls[0][0][0] == '/mock/path'  # This is now the mocked path for all files
    assert calls[1][0][0] == '/mock/path'  # This is now the mocked path for all files
    
    # Verify that os.path.join was called with the correct arguments
    mock_join.assert_any_call(ANY, 'test.docx')
    mock_join.assert_any_call(ANY, 'test.pdf')
    
    # Verify that the final result includes the PDF caption
    final_structure = mock_json_dumps.call_args[0][0]
    assert isinstance(final_structure, ZipStructure)
    assert final_structure.figures[0].figure_caption == "Extracted caption from PDF"
    
    # Verify cleanup operations
    assert mock_remove.call_count == 4  # 4 files in our mock directory structure
    assert mock_rmdir.call_count == 3  # 3 directories (including the root) in our mock structure

def test_main_with_caption_extraction_docx_only(mock_argparse, mock_load_config, mock_zipfile, mock_extract_captions, mock_json, mock_openai_client, mock_os):
    """
    Test caption extraction when only a DOCX file is available.

    This test verifies that the main function correctly handles caption extraction
    when only a DOCX file is present in the ZIP structure (no PDF fallback).
    It ensures that:
    1. The system attempts to extract captions from the DOCX file.
    2. The extraction process is called only once (no fallback to PDF).
    3. The final ZipStructure contains captions extracted from the DOCX file.

    Args:
        mock_argparse: Mocked command-line argument parser.
        mock_load_config: Mocked configuration loader.
        mock_zipfile: Mocked ZIP file handler.
        mock_extract_captions: Mocked caption extraction function.
        mock_json: Mocked JSON operations.
        mock_openai_client: Mocked OpenAI client.
        mock_os: Mocked OS operations.
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
        with patch('soda_curation.main.StructureZipFileGPT') as mock_structure_gpt:
            mock_structure_gpt.return_value.process_zip_structure.return_value = ZipStructure(
                manuscript_id="test",
                xml="test.xml",
                docx="test.docx",
                pdf=None,
                appendix=[],
                figures=[Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", [])]
            )
            main()

        mock_structure_gpt.assert_called_once()
        mock_gpt.return_value.extract_captions.assert_called_once()
        assert mock_gpt.return_value.extract_captions.call_args[0][0].endswith('test.docx')
