import pytest
from unittest.mock import Mock, patch
import json
from soda_curation.pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from soda_curation.pipeline.zip_structure.zip_structure_base import ZipStructure, Figure

@pytest.fixture
def mock_openai_client():
    with patch('soda_curation.pipeline.extract_captions.extract_captions_openai.openai.OpenAI') as mock_client:
        yield mock_client

@pytest.fixture
def sample_zip_structure():
    return ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[
            Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "TO BE ADDED IN LATER STEP", []),
            Figure("Figure 2", ["image2.png"], ["data2.xlsx"], "TO BE ADDED IN LATER STEP", [])
        ]
    )

@pytest.fixture
def sample_config():
    return {
        'api_key': 'test_key',
        'model': 'gpt-4',
        'caption_extraction_assistant_id': 'test_assistant_id'
    }

def test_gpt_extract_captions_success(mock_openai_client, sample_zip_structure, sample_config):
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = Mock(status='completed')
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[
        Mock(content=[Mock(text=Mock(value=json.dumps({
            "Figure 1": "This is caption for Figure 1",
            "Figure 2": "This is caption for Figure 2"
        })))])
    ])

    extractor = FigureCaptionExtractorGpt(sample_config)

    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1"
    assert result.figures[1].figure_caption == "This is caption for Figure 2"

def test_gpt_extract_captions_api_error(mock_openai_client, sample_zip_structure, sample_config):
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.side_effect = Exception("API Error")

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result == sample_zip_structure  # Should return original structure on error

def test_gpt_extract_captions_invalid_json(mock_openai_client, sample_zip_structure, sample_config):
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value="Invalid JSON"))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_gpt_extract_captions_missing_figures(mock_openai_client, sample_zip_structure, sample_config):
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value=json.dumps({
        "Figure 1": "This is caption for Figure 1"
    })))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1"
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_gpt_extract_captions_pdf_fallback(mock_openai_client, sample_zip_structure, sample_config):
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value=json.dumps({
        "Figure 1": "This is caption for Figure 1 from PDF",
        "Figure 2": "This is caption for Figure 2 from PDF"
    })))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.pdf", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1 from PDF"
    assert result.figures[1].figure_caption == "This is caption for Figure 2 from PDF"
