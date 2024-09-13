import pytest
from unittest.mock import patch, Mock
import zipfile
from soda_curation.main import main
import importlib

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

def test_main_success(mock_argparse, mock_load_config, mock_zipfile, mock_json):
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
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.StructureZipFileGPT') as mock_gpt:
            mock_gpt.return_value.process_zip_structure.return_value = {'result': 'success'}
            mock_json.return_value = '{"result": "success"}'

            main()

            mock_gpt.assert_called_once()
            mock_gpt.return_value.process_zip_structure.assert_called_once_with(['file1', 'file2'])
            mock_json.assert_called_once()

def test_main_anthropic(mock_argparse, mock_load_config, mock_zipfile, mock_json):
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
    mock_load_config.return_value = {'ai': 'anthropic', 'anthropic': {'api_key': 'test'}}
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.StructureZipFileClaude') as mock_claude:
            mock_claude.return_value.process_zip_structure.return_value = {'result': 'success'}
            mock_json.return_value = '{"result": "success"}'

            main()

            mock_claude.assert_called_once()
            mock_claude.return_value.process_zip_structure.assert_called_once_with(['file1', 'file2'])
            mock_json.assert_called_once()

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

def test_main_processing_failure(mock_argparse, mock_load_config, mock_zipfile):
    """
    Test the main function's behavior when ZIP file processing fails.
    
    This test verifies that when the AI processor fails to process the ZIP file structure
    (returning None), the main function exits with a system error.
    """
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.StructureZipFileGPT') as mock_gpt:
            mock_gpt.return_value.process_zip_structure.return_value = None

            with pytest.raises(SystemExit):
                main()
