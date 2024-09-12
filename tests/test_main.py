import pytest
from unittest.mock import patch, Mock
import zipfile
from soda_curation.main import main

@pytest.fixture
def mock_argparse():
    with patch('argparse.ArgumentParser.parse_args') as mock_args:
        yield mock_args

@pytest.fixture
def mock_load_config():
    with patch('soda_curation.main.load_config') as mock_config:
        yield mock_config

@pytest.fixture
def mock_zipfile():
    with patch('zipfile.ZipFile') as mock_zip:
        yield mock_zip

@pytest.fixture
def mock_json():
    with patch('json.dumps') as mock_dumps:
        yield mock_dumps

def test_main_success(mock_argparse, mock_load_config, mock_zipfile, mock_json):
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
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'invalid_provider'}
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with pytest.raises(SystemExit):
            main()

def test_main_missing_arguments(mock_argparse):
    mock_argparse.return_value = Mock(zip=None, config=None)

    with pytest.raises(SystemExit):
        main()

def test_main_zip_file_not_found(mock_argparse, mock_load_config):
    mock_argparse.return_value = Mock(zip='nonexistent.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}

    with patch('soda_curation.main.Path.is_file', return_value=False):
        with pytest.raises(SystemExit):
            main()

def test_main_invalid_zip_file(mock_argparse, mock_load_config, mock_zipfile):
    mock_argparse.return_value = Mock(zip='invalid.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}
    mock_zipfile.side_effect = zipfile.BadZipFile()

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with pytest.raises(SystemExit):
            main()

def test_main_processing_failure(mock_argparse, mock_load_config, mock_zipfile):
    mock_argparse.return_value = Mock(zip='test.zip', config='config.yaml')
    mock_load_config.return_value = {'ai': 'openai', 'openai': {'api_key': 'test'}}
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['file1', 'file2']

    with patch('soda_curation.main.Path.is_file', return_value=True):
        with patch('soda_curation.main.StructureZipFileGPT') as mock_gpt:
            mock_gpt.return_value.process_zip_structure.return_value = None

            with pytest.raises(SystemExit):
                main()