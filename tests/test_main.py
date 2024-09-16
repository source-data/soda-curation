import pytest
from unittest.mock import patch, Mock, MagicMock
import zipfile
from soda_curation.main import main
import importlib
from soda_curation.pipeline.zip_structure.zip_structure_base import ZipStructure, Figure


@pytest.fixture
def mock_argparse():
    """
    Fixture to mock the argparse.ArgumentParser.parse_args method.
    
    This allows us to simulate command-line arguments in tests without actually
    passing them from the command line.
    """
    with patch('argparse.ArgumentParser.parse_args') as mock_args:
        yield mock_args

@pytest.fixture
def mock_load_config():
    """
    Fixture to mock the load_config function from the soda_curation.main module.
    
    This allows us to control the configuration returned in tests without reading
    from an actual configuration file.
    """
    with patch('soda_curation.main.load_config') as mock_config:
        yield mock_config

@pytest.fixture
def mock_zipfile():
    """
    Fixture to mock the zipfile.ZipFile class.
    
    This allows us to simulate operations on ZIP files without actually creating
    or reading real ZIP files during tests.
    """
    with patch('zipfile.ZipFile') as mock_zip:
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
    with patch('soda_curation.pipeline.extract_captions.extract_captions_openai.openai.OpenAI') as mock_client:
        mock_client.return_value.beta.assistants.update.return_value = MagicMock()
        mock_client.return_value.beta.assistants.create.return_value = MagicMock()
        yield mock_client

@pytest.fixture
def mock_anthropic_client():
    with patch('soda_curation.pipeline.extract_captions.extract_captions_anthropic.Anthropic') as mock_client:
        mock_client.return_value.messages.create.return_value = MagicMock()
        yield mock_client

@pytest.fixture
def mock_os():
    with patch('os.path.exists', return_value=True), \
         patch('os.remove'), \
         patch('os.rmdir'), \
         patch('os.walk', return_value=[]):
        yield

def test_main_success(mock_argparse, mock_load_config, mock_zipfile, mock_json, mock_openai_client, mock_os):
    """
    Test the main function's successful execution path.

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

@pytest.fixture
def mock_extract_captions():
    with patch('soda_curation.main.FigureCaptionExtractorGpt') as mock_gpt:
        with patch('soda_curation.main.FigureCaptionExtractorClaude') as mock_claude:
            yield mock_gpt, mock_claude

def test_main_with_caption_extraction_docx_success(mock_argparse, mock_load_config, mock_zipfile, mock_extract_captions, mock_json, mock_openai_client, mock_os):
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

def test_main_with_caption_extraction_pdf_fallback(mock_argparse, mock_load_config, mock_zipfile, mock_extract_captions, mock_json, mock_openai_client, mock_os):
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
    mock_gpt.return_value.extract_captions.side_effect = [
        ZipStructure(  # DOCX extraction result
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

        assert mock_gpt.return_value.extract_captions.call_count == 2
        assert mock_gpt.return_value.extract_captions.call_args_list[0][0][0].endswith('test.docx')
        assert mock_gpt.return_value.extract_captions.call_args_list[1][0][0].endswith('test.pdf')


def test_main_with_caption_extraction_docx_only(mock_argparse, mock_load_config, mock_zipfile, mock_extract_captions, mock_json, mock_openai_client, mock_os):
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



