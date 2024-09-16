import pytest
from unittest.mock import Mock, patch, mock_open
import numpy as np
from PIL import Image
from pathlib import Path

from soda_curation.pipeline.object_detection.object_detection import (
    ObjectDetection,
    convert_to_pil_image,
    convert_and_resize_image,
    create_object_detection
)

@pytest.fixture
def mock_yolov10():
    with patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock:
        yield mock

@pytest.fixture
def mock_image():
    with patch('soda_curation.pipeline.object_detection.object_detection.Image') as mock:
        mock.open.return_value = Mock(mode='RGB')
        yield mock

@pytest.fixture
def mock_pdf2image():
    with patch('soda_curation.pipeline.object_detection.object_detection.pdf2image') as mock:
        yield mock

@pytest.fixture
def mock_subprocess():
    with patch('soda_curation.pipeline.object_detection.object_detection.subprocess') as mock:
        yield mock

@pytest.fixture
def mock_os():
    with patch('soda_curation.pipeline.object_detection.object_detection.os') as mock:
        mock.path.exists.return_value = True
        mock.path.abspath.side_effect = lambda x: f"/app/{x}"
        mock.path.splitext.side_effect = lambda x: (x.rsplit('.', 1)[0], '.' + x.rsplit('.', 1)[1])
        yield mock

def test_convert_to_pil_image_jpg(mock_image, mock_os):
    """
    Test that convert_to_pil_image correctly handles JPG files.
    """
    result = convert_to_pil_image('test.jpg')
    mock_image.open.assert_called_once_with('/app/test.jpg')
    assert isinstance(result, mock_image.open.return_value.__class__)

def test_convert_to_pil_image_pdf(mock_pdf2image, mock_image, mock_os):
    """
    Test that convert_to_pil_image correctly handles PDF files.
    """
    mock_page = Mock(spec=Image.Image)
    mock_pdf2image.convert_from_path.return_value = [mock_page]
    
    with patch('soda_curation.pipeline.object_detection.object_detection.convert_and_resize_image', return_value=mock_page) as mock_convert:
        result = convert_to_pil_image('test.pdf')
    
    mock_pdf2image.convert_from_path.assert_called_once_with('/app/test.pdf', dpi=300)
    mock_convert.assert_called_once_with(mock_page)
    assert result == mock_page

def test_convert_to_pil_image_eps(mock_subprocess, mock_image, mock_os):
    """
    Test that convert_to_pil_image correctly handles EPS files.
    """
    result = convert_to_pil_image('test.eps')
    mock_subprocess.run.assert_called_once()
    mock_image.open.assert_called_once()
    assert isinstance(result, mock_image.open.return_value.__class__)

def test_convert_to_pil_image_unsupported(mock_os):
    """
    Test that convert_to_pil_image raises a ValueError for unsupported file formats.
    """
    with pytest.raises(ValueError, match="Unsupported file format: .unsupported"):
        convert_to_pil_image('test.unsupported')

def test_convert_and_resize_image():
    """
    Test that convert_and_resize_image correctly resizes an image.
    """
    mock_image = Mock(mode='RGB')
    result = convert_and_resize_image(mock_image)
    mock_image.thumbnail.assert_called_once()
    assert result == mock_image

def test_object_detection_initialization(mock_yolov10):
    """
    Test the initialization of the ObjectDetection class.
    """
    with patch('soda_curation.pipeline.object_detection.object_detection.Path.exists', return_value=True):
        od = ObjectDetection("test_model.pt")
    assert od.model_path == "test_model.pt"
    mock_yolov10.assert_called_once_with("test_model.pt")

def test_detect_panels(mock_yolov10):
    """
    Test the detect_panels method of the ObjectDetection class.
    """
    od = ObjectDetection("test_model.pt")
    
    # Mock YOLO results
    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_results.boxes.conf = [0.95]
    od.model.return_value = [mock_results]
    
    with patch('soda_curation.pipeline.object_detection.object_detection.convert_to_pil_image') as mock_convert:
        mock_convert.return_value = Mock()
        result = od.detect_panels("test_image.jpg")
    
    assert len(result) == 1
    assert result[0]["panel_label"] == "A"
    assert result[0]["panel_bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert result[0]["panel_caption"] == "TO BE ADDED IN LATER STEP"

def test_create_object_detection():
    """
    Test the create_object_detection function with a custom model path.
    """
    config = {
        "object_detection": {
            "model_path": "custom_model.pt"
        }
    }
    with patch('soda_curation.pipeline.object_detection.object_detection.Path.exists', return_value=True), \
         patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock_yolo:
        od = create_object_detection(config)
    assert isinstance(od, ObjectDetection)
    assert od.model_path == "/app/custom_model.pt"
    mock_yolo.assert_called_once_with("/app/custom_model.pt")

def test_create_object_detection_default_path():
    """
    Test the create_object_detection function with the default model path.
    """
    config = {}
    with patch('soda_curation.pipeline.object_detection.object_detection.Path.exists', return_value=True), \
         patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock_yolo:
        od = create_object_detection(config)
    assert isinstance(od, ObjectDetection)
    assert od.model_path == "/app/data/models/panel_detection_model_no_labels.pt"
    mock_yolo.assert_called_once_with("/app/data/models/panel_detection_model_no_labels.pt")

def test_create_object_detection_file_not_found():
    """
    Test that create_object_detection raises a FileNotFoundError when the model file doesn't exist.
    """
    config = {}
    with patch('soda_curation.pipeline.object_detection.object_detection.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            create_object_detection(config)
