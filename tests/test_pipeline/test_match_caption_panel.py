import pytest
from unittest.mock import Mock, patch, MagicMock
from soda_curation.pipeline.match_caption_panel.match_caption_panel_base import MatchPanelCaption
from soda_curation.pipeline.match_caption_panel.match_caption_panel_openai import MatchPanelCaptionOpenAI
from soda_curation.pipeline.match_caption_panel.match_caption_panel_anthropic import MatchPanelCaptionClaude
from soda_curation.pipeline.zip_structure.zip_structure_base import ZipStructure, Figure

class TestMatchPanelCaptionBase:
    def test_extract_panel_image(self):
        class ConcreteMatchPanelCaption(MatchPanelCaption):
            def match_captions(self, zip_structure):
                pass

        matcher = ConcreteMatchPanelCaption()
        
        with patch('PIL.Image.open') as mock_open, \
             patch('io.BytesIO') as mock_bytesio:
            mock_image = MagicMock()
            mock_image.size = (100, 100)
            mock_open.return_value.__enter__.return_value = mock_image
            
            result = matcher._extract_panel_image('test.jpg', [0.1, 0.1, 0.9, 0.9])
            
            assert isinstance(result, str)
            mock_image.crop.assert_called_once_with((10, 10, 90, 90))

class TestMatchPanelCaptionOpenAI:
    @pytest.fixture
    def mock_openai_client(self):
        with patch('soda_curation.pipeline.match_caption_panel.match_caption_panel_openai.openai.OpenAI') as mock_client:
            yield mock_client

    def test_initialization(self, mock_openai_client):
        config = {
            'openai': {
                'api_key': 'test_key',
                'model': 'gpt-4-vision-preview'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        assert matcher.client == mock_openai_client.return_value

    def test_match_captions(self, mock_openai_client):
        config = {
            'openai': {
                'api_key': 'test_key',
                'model': 'gpt-4-vision-preview'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        zip_structure = ZipStructure(
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[
                Figure("Figure 1", ["image1.png"], [], "Test caption", [
                    {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
                ])
            ]
        )
        
        with patch.object(matcher, '_process_figure', return_value=[{"panel_label": "A", "panel_caption": "Matched caption"}]):
            result = matcher.match_captions(zip_structure)
            
            assert isinstance(result, ZipStructure)
            assert result.figures[0].panels[0]["panel_caption"] == "Matched caption"
    def test_call_openai_api(self, mock_openai_client):
        config = {
            'openai': {
                'api_key': 'test_key',
                'model': 'gpt-4-vision-preview'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = "PANEL_A: Test caption"
        
        with patch('soda_curation.pipeline.match_caption_panel.match_caption_panel_openai.get_match_panel_caption_prompt', return_value="Mock prompt"):
            result = matcher._call_openai_api("encoded_image", "Test figure caption")
        
        assert result == "PANEL_A: Test caption"
        mock_openai_client.return_value.chat.completions.create.assert_called_once()

    def test_parse_response(self, mock_openai_client):
        config = {
            'openai': {
                'api_key': 'test_key',
                'model': 'gpt-4-vision-preview'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        response = "PANEL_A: Test caption"
        label, caption = matcher._parse_response(response)
        
        assert label == "A"
        assert caption == "Test caption"


class TestMatchPanelCaptionClaude:
    @pytest.fixture
    def mock_anthropic_client(self):
        with patch('soda_curation.pipeline.match_caption_panel.match_caption_panel_anthropic.Anthropic') as mock_client:
            yield mock_client

    def test_initialization(self, mock_anthropic_client):
        config = {
            'anthropic': {
                'api_key': 'test_key',
                'model': 'claude-3-opus-20240229'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        assert matcher.client == mock_anthropic_client.return_value

    def test_match_captions(self, mock_anthropic_client):
        config = {
            'anthropic': {
                'api_key': 'test_key',
                'model': 'claude-3-opus-20240229'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        
        zip_structure = ZipStructure(
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[
                Figure("Figure 1", ["image1.png"], [], "Test caption", [
                    {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
                ])
            ]
        )
        
        with patch.object(matcher, '_process_figure', return_value=[{"panel_label": "A", "panel_caption": "Matched caption"}]):
            result = matcher.match_captions(zip_structure)
            
            assert isinstance(result, ZipStructure)
            assert result.figures[0].panels[0]["panel_caption"] == "Matched caption"

    def test_call_anthropic_api(self, mock_anthropic_client):
        config = {
            'anthropic': {
                'api_key': 'test_key',
                'model': 'claude-3-opus-20240229',
                'max_tokens_to_sample': 1000,
                'temperature': 0.7
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        
        mock_response = MagicMock()
        mock_response.content = [Mock(text="PANEL_A: Test caption")]
        mock_anthropic_client.return_value.messages.create.return_value = mock_response
        
        with patch('soda_curation.pipeline.match_caption_panel.match_caption_panel_anthropic.get_match_panel_caption_prompt', return_value="Mock prompt"):
            result = matcher._call_anthropic_api("encoded_image", "Test figure caption")
        
        assert result == "PANEL_A: Test caption"
        mock_anthropic_client.return_value.messages.create.assert_called_once()

    def test_parse_response(self, mock_anthropic_client):
        config = {
            'anthropic': {
                'api_key': 'test_key',
                'model': 'claude-3-opus-20240229'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        
        response = "PANEL_A: Test caption"
        label, caption = matcher._parse_response(response)
        
        assert label == "A"
        assert caption == "Test caption"

    def test_process_figure(self, mock_anthropic_client):
        config = {
            'anthropic': {
                'api_key': 'test_key',
                'model': 'claude-3-opus-20240229'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        
        figure = Figure("Figure 1", ["image1.png"], [], "Test caption", [
            {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
        ])
        
        with patch.object(matcher, '_extract_panel_image', return_value="encoded_image"), \
             patch.object(matcher, '_call_anthropic_api', return_value="PANEL_A: Matched caption"), \
             patch.object(matcher, '_parse_response', return_value=("A", "Matched caption")):
            
            result = matcher._process_figure(figure)
            
            assert len(result) == 1
            assert result[0]["panel_label"] == "A"
            assert result[0]["panel_caption"] == "Matched caption"

    def test_save_debug_image(self, mock_anthropic_client, tmp_path):
        config = {
            'anthropic': {
                'api_key': 'test_key',
                'model': 'claude-3-opus-20240229'
            },
            'extract_dir': '/mock/extract_dir',
            'debug': {
                'enabled': True,
                'debug_dir': str(tmp_path)
            }
        }
        matcher = MatchPanelCaptionClaude(config)
        
        encoded_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        filename = "test_debug_image.png"
        
        matcher._save_debug_image(encoded_image, filename)
        
        assert (tmp_path / filename).exists()

    def test_initialization_missing_config(self, mock_anthropic_client):
        with pytest.raises(ValueError, match="Anthropic configuration is missing from the main configuration"):
            MatchPanelCaptionClaude({})

    def test_initialization_missing_api_key(self, mock_anthropic_client):
        with pytest.raises(ValueError, match="Anthropic configuration is missing from the main configuration"):
            MatchPanelCaptionClaude({'anthropic': {}})


    def test_initialization_missing_extract_dir(self, mock_anthropic_client):
        with pytest.raises(ValueError, match="extract_dir is not set in the configuration"):
            MatchPanelCaptionClaude({'anthropic': {'api_key': 'test'}, 'debug': {'enabled': True}})

    def test_debug_mode(self, mock_anthropic_client, tmp_path):
        config = {
            'anthropic': {'api_key': 'test', 'model': 'claude-3-opus-20240229'},
            'extract_dir': str(tmp_path),
            'debug': {'enabled': True, 'debug_dir': str(tmp_path), 'process_first_figure_only': True}
        }
        matcher = MatchPanelCaptionClaude(config)
        assert matcher.debug_enabled
        assert matcher.debug_dir == str(tmp_path)
        assert matcher.process_first_figure_only

    def test_process_figure_error(self, mock_anthropic_client):
        config = {
            'anthropic': {'api_key': 'test', 'model': 'claude-3-opus-20240229'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        
        figure = Figure("Figure 1", ["image1.png"], [], "Test caption", [
            {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
        ])
        
        with patch.object(matcher, '_extract_panel_image', side_effect=Exception("Test error")):
            result = matcher._process_figure(figure)
            assert result == []
            
    def test_call_anthropic_api_error(self, mock_anthropic_client):
        config = {
            'anthropic': {'api_key': 'test', 'model': 'claude-3-opus-20240229'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        
        mock_anthropic_client.return_value.messages.create.side_effect = Exception("API Error")
        
        result = matcher._call_anthropic_api("encoded_image", "Test figure caption")
        assert result == ""

    def test_parse_response_error(self, mock_anthropic_client):
        config = {
            'anthropic': {'api_key': 'test', 'model': 'claude-3-opus-20240229'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionClaude(config)
        
        response = "Invalid response"
        label, caption = matcher._parse_response(response)
        assert label == ""
        assert caption == ""

class TestMatchPanelCaptionOpenAI:
    @pytest.fixture
    def mock_openai_client(self):
        with patch('soda_curation.pipeline.match_caption_panel.match_caption_panel_openai.openai.OpenAI') as mock_client:
            yield mock_client

    def test_initialization(self, mock_openai_client):
        config = {
            'openai': {
                'api_key': 'test_key',
                'model': 'gpt-4-vision-preview'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        assert matcher.client == mock_openai_client.return_value

    def test_initialization_missing_config(self, mock_openai_client):
        with pytest.raises(ValueError, match="OpenAI configuration is missing from the main configuration"):
            MatchPanelCaptionOpenAI({})

    def test_initialization_missing_api_key(self, mock_openai_client):
        with pytest.raises(ValueError, match="OpenAI configuration is missing from the main configuration"):
            MatchPanelCaptionOpenAI({'openai': {}})

    def test_initialization_missing_extract_dir(self, mock_openai_client):
        with pytest.raises(ValueError, match="extract_dir is not set in the configuration"):
            MatchPanelCaptionOpenAI({'openai': {'api_key': 'test'}, 'debug': {'enabled': True}})

    def test_initialization_with_debug_options(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir',
            'debug': {'enabled': True, 'debug_dir': '/mock/debug_dir', 'process_first_figure_only': True}
        }
        matcher = MatchPanelCaptionOpenAI(config)
        assert matcher.debug_enabled
        assert matcher.debug_dir == '/mock/debug_dir'
        assert matcher.process_first_figure_only

    def test_match_captions(self, mock_openai_client):
        config = {
            'openai': {
                'api_key': 'test_key',
                'model': 'gpt-4-vision-preview'
            },
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        zip_structure = ZipStructure(
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[
                Figure("Figure 1", ["image1.png"], [], "Test caption", [
                    {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
                ])
            ]
        )
        
        with patch.object(matcher, '_process_figure', return_value=[{"panel_label": "A", "panel_caption": "Matched caption"}]):
            result = matcher.match_captions(zip_structure)
            
            assert isinstance(result, ZipStructure)
            assert result.figures[0].panels[0]["panel_caption"] == "Matched caption"

    def test_match_captions_with_debug(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir',
            'debug': {'enabled': True, 'debug_dir': '/mock/debug_dir', 'process_first_figure_only': True}
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        zip_structure = ZipStructure(
            manuscript_id="test",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            appendix=[],
            figures=[
                Figure("Figure 1", ["image1.png"], [], "Test caption", [
                    {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
                ]),
                Figure("Figure 2", ["image2.png"], [], "Test caption 2", [
                    {"panel_label": "B", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
                ])
            ]
        )
        
        with patch.object(matcher, '_process_figure', return_value=[{"panel_label": "A", "panel_caption": "Matched caption"}]):
            result = matcher.match_captions(zip_structure)
            
            assert isinstance(result, ZipStructure)
            assert len(result.figures) == 2
            assert result.figures[0].panels[0]["panel_caption"] == "Matched caption"
            assert result.figures[1].panels == []  # Second figure should not be processed due to debug mode

    def test_process_figure(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        figure = Figure("Figure 1", ["image1.png"], [], "Test caption", [
            {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
        ])
        
        with patch.object(matcher, '_extract_panel_image', return_value="encoded_image"), \
             patch.object(matcher, '_call_openai_api', return_value="PANEL_A: Matched caption"), \
             patch.object(matcher, '_parse_response', return_value=("A", "Matched caption")):
            
            result = matcher._process_figure(figure)
            
            assert len(result) == 1
            assert result[0]["panel_label"] == "A"
            assert result[0]["panel_caption"] == "Matched caption"

    def test_process_figure_error(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        figure = Figure("Figure 1", ["image1.png"], [], "Test caption", [
            {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
        ])
        
        with patch.object(matcher, '_extract_panel_image', side_effect=Exception("Test error")):
            result = matcher._process_figure(figure)
            assert result == []

    def test_call_openai_api(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        mock_openai_client.return_value.chat.completions.create.return_value.choices[0].message.content = "PANEL_A: Test caption"
        
        with patch('soda_curation.pipeline.match_caption_panel.match_caption_panel_openai.get_match_panel_caption_prompt', return_value="Mock prompt"):
            result = matcher._call_openai_api("encoded_image", "Test figure caption")
        
        assert result == "PANEL_A: Test caption"
        mock_openai_client.return_value.chat.completions.create.assert_called_once()

    def test_call_openai_api_error(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API Error")
        
        result = matcher._call_openai_api("encoded_image", "Test figure caption")
        assert result == ""

    def test_parse_response(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        response = "PANEL_A: Test caption"
        label, caption = matcher._parse_response(response)
        
        assert label == "A"
        assert caption == "Test caption"

    def test_parse_response_error(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        response = "Invalid response"
        label, caption = matcher._parse_response(response)
        assert label == ""
        assert caption == ""

    def test_save_debug_image(self, mock_openai_client, tmp_path):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir',
            'debug': {'enabled': True, 'debug_dir': str(tmp_path)}
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        encoded_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        filename = "test_debug_image.png"
        
        matcher._save_debug_image(encoded_image, filename)
        
        assert (tmp_path / filename).exists()

    def test_save_debug_image_error(self, mock_openai_client, tmp_path):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir',
            'debug': {'enabled': True, 'debug_dir': str(tmp_path)}
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        with patch('PIL.Image.open', side_effect=Exception("Image Error")):
            matcher._save_debug_image("invalid_image_data", "test.png")
            # This should not raise an exception, but log an error instead

    def test_process_figure_error(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        figure = Figure("Figure 1", ["image1.png"], [], "Test caption", [
            {"panel_label": "A", "panel_bbox": [0.1, 0.1, 0.9, 0.9]}
        ])
        
        with patch.object(matcher, '_extract_panel_image', side_effect=Exception("Test error")):
            result = matcher._process_figure(figure)
            assert result == []

    def test_call_openai_api_error(self, mock_openai_client):
        config = {
            'openai': {'api_key': 'test', 'model': 'gpt-4-vision-preview'},
            'extract_dir': '/mock/extract_dir'
        }
        matcher = MatchPanelCaptionOpenAI(config)
        
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API Error")
        
        result = matcher._call_openai_api("encoded_image", "Test figure caption")
        assert result == ""
