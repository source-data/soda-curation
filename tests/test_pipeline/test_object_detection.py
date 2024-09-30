"""
This module contains unit tests for the object detection functionality in the soda_curation package.

It tests various aspects of image conversion, resizing, and panel detection using the YOLOv10 model.
The tests use mock objects to simulate file operations and model predictions.
"""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
from PIL import Image

from soda_curation.pipeline.object_detection.object_detection import (
    ObjectDetection,
    convert_and_resize_image,
    convert_to_pil_image,
    create_object_detection,
)


@pytest.fixture
def mock_yolov10():
    """
    Fixture to mock the YOLOv10 class.

    This fixture patches the YOLOv10 import to return a mock object,
    allowing tests to run without an actual YOLOv10 model.
    """
    with patch('soda_curation.pipeline.object_detection.object_detection.YOLOv10') as mock:
        yield mock

@pytest.fixture
def mock_image():
    """
    Fixture to mock the PIL Image module.

    This fixture patches the Image module to return mock objects for image operations,
    allowing tests to run without actual image files.
    """
    with patch('soda_curation.pipeline.object_detection.object_detection.Image') as mock:
        mock.open.return_value = Mock(mode='RGB')
        yield mock

@pytest.fixture
def mock_pdf2image():
    """
    Fixture to mock the pdf2image module.

    This fixture patches the pdf2image module to return mock objects for PDF conversion,
    allowing tests to simulate PDF to image conversion without actual PDF files.
    """
    with patch('soda_curation.pipeline.object_detection.object_detection.pdf2image') as mock:
        yield mock

@pytest.fixture
def mock_subprocess():
    """
    Fixture to mock the subprocess module.

    This fixture patches the subprocess module to simulate system command execution,
    particularly for EPS file conversion.
    """
    with patch('soda_curation.pipeline.object_detection.object_detection.subprocess') as mock:
        yield mock

@pytest.fixture
def mock_os():
    """
    Fixture to mock various os module functions.

    This fixture patches os.path functions to simulate file system operations
    without requiring actual files or directories.
    """
    with patch('soda_curation.pipeline.object_detection.object_detection.os') as mock:
        mock.path.exists.return_value = True
        mock.path.abspath.side_effect = lambda x: f"/app/{x}"
        mock.path.splitext.side_effect = lambda x: (x.rsplit('.', 1)[0], '.' + x.rsplit('.', 1)[1])
        yield mock

def test_convert_to_pil_image_jpg(mock_image, mock_os):
    """
    Test the conversion of a JPG file to a PIL Image.

    This test verifies that the convert_to_pil_image function correctly handles JPG files
    by opening them with PIL and returning the resulting Image object.
    """
    result, _ = convert_to_pil_image('test.jpg')
    mock_image.open.assert_called_once_with('/app/test.jpg')
    assert isinstance(result, mock_image.open.return_value.__class__)

def test_convert_to_pil_image_pdf(mock_pdf2image, mock_image, mock_os):
    """
    Test the conversion of a PDF file to a PIL Image.

    This test checks if the convert_to_pil_image function correctly uses pdf2image
    to convert a PDF file to an image, and then processes it with convert_and_resize_image.
    """
    mock_page = Mock(spec=Image.Image)
    mock_pdf2image.convert_from_path.return_value = [mock_page]
    
    with patch('soda_curation.pipeline.object_detection.object_detection.convert_and_resize_image', return_value=mock_page) as mock_convert:
        result, new_file_path = convert_to_pil_image('test.pdf')
    
    mock_pdf2image.convert_from_path.assert_called_once_with('/app/test.pdf', dpi=300)
    mock_convert.assert_called_once_with(mock_page)
    assert result == mock_page
    assert new_file_path == '/app/test.png'

def test_convert_to_pil_image_eps(mock_subprocess, mock_image, mock_os):
    """
    Test the conversion of an EPS file to a PIL Image.

    This test verifies that the convert_to_pil_image function correctly handles EPS files
    by using subprocess to convert them to PNG and then opening the result with PIL.
    """
    result, _ = convert_to_pil_image('test.eps')
    mock_subprocess.run.assert_called_once()
    mock_image.open.assert_called_once()
    assert isinstance(result, mock_image.open.return_value.__class__)

def test_convert_to_pil_image_unsupported(mock_os):
    """
    Test the handling of unsupported file formats in convert_to_pil_image.

    This test checks if the function raises a ValueError when given an unsupported file format.
    """
    with pytest.raises(ValueError, match="Unsupported file format: .unsupported"):
        convert_to_pil_image('test.unsupported')

def test_convert_and_resize_image():
    """
    Test the image conversion and resizing functionality.

    This test verifies that the convert_and_resize_image function correctly resizes
    an input image while maintaining its mode.
    """
    mock_image = Mock(mode='RGB')
    result = convert_and_resize_image(mock_image)
    mock_image.thumbnail.assert_called_once()
    assert result == mock_image

def test_object_detection_initialization(mock_yolov10):
    """
    Test the initialization of the ObjectDetection class.

    This test checks if the ObjectDetection class is correctly initialized
    with the given model path and if it creates a YOLOv10 model instance.
    """
    with patch('soda_curation.pipeline.object_detection.object_detection.Path.exists', return_value=True):
        od = ObjectDetection("test_model.pt")
    assert od.model_path == "test_model.pt"
    mock_yolov10.assert_called_once_with("test_model.pt")

def test_detect_panels(mock_yolov10):
    """
    Test the panel detection functionality of the ObjectDetection class.

    This test simulates the detection of panels in an image using a mocked YOLOv10 model,
    and verifies that the detect_panels method correctly processes the model's output.
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
    Test the creation of an ObjectDetection instance with a custom model path.

    This test verifies that the create_object_detection function correctly creates
    an ObjectDetection instance using a custom model path from the configuration.
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
    Test the creation of an ObjectDetection instance with the default model path.

    This test checks if the create_object_detection function uses the default model path
    when no custom path is provided in the configuration.
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
    Test the handling of a non-existent model file in create_object_detection.

    This test verifies that the create_object_detection function raises a FileNotFoundError
    when the specified model file does not exist.
    """
    config = {}
    with patch('soda_curation.pipeline.object_detection.object_detection.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            create_object_detection(config)
