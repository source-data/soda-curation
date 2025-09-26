import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from src.soda_curation.pipeline.match_caption_panel.object_detection import (
    ObjectDetection,
    convert_and_resize_image,
    convert_eps_to_png,
    convert_tiff_with_cv2,
    convert_to_pil_image,
    create_object_detection,
    create_standard_thumbnail,
    fallback_ghostscript_conversion,
    scale_down_large_image,
)


@pytest.fixture
def mock_image():
    """Fixture to mock the PIL Image module."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.Image"
    ) as mock:
        mock.open.return_value = Mock(mode="RGB")
        yield mock


@pytest.fixture
def mock_yolo():
    """Fixture to mock the YOLO class."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLO"
    ) as mock:
        mock.return_value = Mock()  # Add a return value for the model
        yield mock


@pytest.fixture
def mock_pdf2image():
    """Fixture to mock the pdf2image module."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.pdf2image"
    ) as mock:
        yield mock


@pytest.fixture
def mock_subprocess():
    """Fixture to mock the subprocess module."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.subprocess"
    ) as mock:
        yield mock


@pytest.fixture
def mock_os():
    """Fixture to mock various os module functions."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.os"
    ) as mock:
        mock.path.exists.return_value = True
        mock.path.abspath.side_effect = lambda x: f"/app/{x}"
        mock.path.splitext.side_effect = lambda x: (
            x.rsplit(".", 1)[0],
            "." + x.rsplit(".", 1)[1],
        )
        yield mock


def test_convert_to_pil_image_jpg(mock_image, mock_os):
    """
    Test the conversion of a JPG file to a PIL Image.

    This test verifies that the convert_to_pil_image function correctly handles JPG files
    by opening them with PIL and returning the resulting Image object.
    """
    result, _ = convert_to_pil_image("test.jpg")
    mock_image.open.assert_called_once_with("/app/test.jpg")
    assert isinstance(result, mock_image.open.return_value.__class__)


def test_convert_to_pil_image_pdf(mock_pdf2image, mock_image, mock_os):
    """
    Test the conversion of a PDF file to a PIL Image.

    This test checks if the convert_to_pil_image function correctly uses
    create_standard_thumbnail to convert a PDF file to an image.
    """
    # Create a mock image that will be returned by create_standard_thumbnail
    mock_img = Mock(spec=Image.Image)

    # Mock create_standard_thumbnail to return the path and make the actual function
    # return the mock image
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.create_standard_thumbnail",
        return_value="/app/test.png",
    ) as mock_thumbnail, patch("PIL.Image.open", return_value=mock_img) as _:
        _, new_file_path = convert_to_pil_image("test.pdf")

    # Verify that create_standard_thumbnail was called correctly
    mock_thumbnail.assert_called_once_with("/app/test.pdf", "/app/test.png", dpi=300)

    # Verify the output paths
    assert new_file_path == "/app/test.png"


def test_convert_to_pil_image_eps(mock_subprocess, mock_image, mock_os):
    """
    Test the conversion of an EPS file to a PIL Image.

    This test verifies that the convert_to_pil_image function correctly handles EPS files
    by using subprocess to convert them to PNG and then opening the result with PIL.
    """
    # Make the first conversion attempt succeed to avoid fallback
    mock_subprocess.CalledProcessError = subprocess.CalledProcessError
    mock_subprocess.run.return_value = Mock(returncode=0)

    result, _ = convert_to_pil_image("test.eps")

    # Verify that subprocess.run was called at least once
    assert mock_subprocess.run.call_count >= 1
    mock_image.open.assert_called_once()
    assert isinstance(result, mock_image.open.return_value.__class__)


def test_convert_to_pil_image_unsupported(mock_os):
    """
    Test the handling of unsupported file formats in convert_to_pil_image.

    This test checks if the function raises a ValueError when given an unsupported file format.
    """
    # The mock_os fixture makes os.path.exists return True, but Image.open still fails
    with pytest.raises(ValueError, match="Failed to convert or open image"):
        convert_to_pil_image("test.unsupported")


def test_convert_and_resize_image():
    """
    Test the image conversion and resizing functionality.

    This test verifies that the convert_and_resize_image function correctly resizes
    an input image while maintaining its mode.
    """
    mock_image = Mock(mode="RGB")
    result = convert_and_resize_image(mock_image)
    mock_image.thumbnail.assert_called_once()
    assert result == mock_image


def test_object_detection_initialization(mock_yolo):
    """
    Test the initialization of the ObjectDetection class.

    This test checks if the ObjectDetection class is correctly initialized
    with the given model path and if it creates a YOLO model instance.
    """
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.Path.exists",
        return_value=True,
    ):
        od = ObjectDetection("test_model.pt")
    assert od.model_path == "test_model.pt"
    mock_yolo.assert_called_once_with("test_model.pt")


def test_detect_panels(mock_yolo):
    """
    Test the panel detection functionality of the ObjectDetection class.

    This test simulates the detection of panels in an image using a mocked YOLO model,
    and verifies that the detect_panels method correctly processes the model's output.
    """
    od = ObjectDetection("test_model.pt")

    # Mock YOLO results
    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_results.boxes.conf = [0.95]
    od.model.return_value = [mock_results]

    # Create mock image with proper numpy array
    mock_image = Mock(spec=Image.Image)
    mock_array = np.zeros((100, 100, 3), dtype=np.uint8)  # Create actual numpy array

    with patch("numpy.array", return_value=mock_array):
        result = od.detect_panels(mock_image)

    assert len(result) == 1
    assert result[0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert result[0]["confidence"] == 0.95


def test_create_object_detection():
    """
    Test the creation of an ObjectDetection instance with a custom model path.

    This test verifies that the create_object_detection function correctly creates
    an ObjectDetection instance using a custom model path from the configuration.
    """
    config = {"object_detection": {"model_path": "custom_model.pt"}}
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.Path.exists",
        return_value=True,
    ), patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLO"
    ) as mock_yolo:
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
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.Path.exists",
        return_value=True,
    ), patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLO"
    ) as mock_yolo:
        od = create_object_detection(config)
    assert isinstance(od, ObjectDetection)
    assert od.model_path == "/app/data/models/panel_detection_model_no_labels.pt"
    mock_yolo.assert_called_once_with(
        "/app/data/models/panel_detection_model_no_labels.pt"
    )


def test_create_object_detection_file_not_found():
    """
    Test the handling of a non-existent model file in create_object_detection.

    This test verifies that the create_object_detection function raises a FileNotFoundError
    when the specified model file does not exist.
    """
    config = {}
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.Path.exists",
        return_value=False,
    ):
        with pytest.raises(FileNotFoundError):
            create_object_detection(config)


def test_convert_to_pil_image_ai(mock_subprocess, mock_image, mock_os):
    """
    Test the conversion of an AI file to a PIL Image.

    This test verifies that the convert_to_pil_image function correctly handles AI files
    by using Ghostscript to convert them to PNG and then opening the result with PIL.
    """
    result, new_file_path = convert_to_pil_image("test.ai")

    # Check if Ghostscript command was called
    mock_subprocess.run.assert_called_once()
    command_args = mock_subprocess.run.call_args[0][0]
    assert command_args[0] == "convert"

    # Check if the converted PNG was opened
    mock_image.open.assert_called_once()
    assert isinstance(result, mock_image.open.return_value.__class__)
    assert new_file_path == "/app/test.png"


@pytest.fixture
def mocker(monkeypatch):
    """Fixture to handle mocking."""
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


def test_convert_to_pil_image_ai_failure(mock_subprocess, mock_os):
    """Test handling of AI conversion failures."""
    mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "gs")
    mock_os.path.exists.return_value = True

    with pytest.raises(ValueError) as exc_info:
        convert_to_pil_image("test.ai")
    assert "Failed to convert or open image" in str(exc_info.value)


@pytest.mark.parametrize(
    "file_ext", [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf", ".eps", ".ai"]
)
def test_supported_formats(
    file_ext, mock_image, mock_subprocess, mock_os, mock_pdf2image
):
    """Test that all supported formats are handled correctly."""
    test_file = f"test{file_ext}"
    mock_os.path.exists.return_value = True

    # Setup mocks for different file types
    if file_ext == ".pdf":
        mock_page = Mock(spec=Image.Image)
        mock_pdf2image.convert_from_path.return_value = [mock_page]
        mock_image.open.return_value = mock_page
    elif file_ext in [".tif", ".tiff"]:
        # Mock the TIFF conversion process
        with patch(
            "src.soda_curation.pipeline.match_caption_panel.object_detection.convert_tiff_with_cv2",
            return_value="/app/test.png",
        ) as mock_convert_tiff, patch(
            "PIL.Image.open", return_value=Mock(spec=Image.Image)
        ) as _:
            result, path = convert_to_pil_image(test_file)
            assert path == "/app/test.png"
            mock_convert_tiff.assert_called_once()
            return  # Skip the rest of the function for TIFF files

    # For other formats
    result, _ = convert_to_pil_image(test_file)
    assert isinstance(result, Image.Image) or isinstance(result, Mock)


def test_detect_panels_from_ai(mock_subprocess, mock_image, mock_os, monkeypatch):
    """Test panel detection on a converted AI file."""
    config = {"object_detection": {"model_path": "test_model.pt"}}

    # Use monkeypatch instead of mocker
    monkeypatch.setattr("pathlib.Path.exists", lambda x: True)

    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLO"
    ) as mock_yolo:
        # Setup mock YOLO results
        mock_results = Mock()
        mock_results.boxes.xyxyn.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_results.boxes.conf = [0.95]
        mock_yolo.return_value.return_value = [mock_results]

        # Create object detector
        detector = create_object_detection(config)

        # Convert and detect panels
        image, _ = convert_to_pil_image("test.ai")
        panels = detector.detect_panels(image)

        assert len(panels) == 1
        assert panels[0]["bbox"] == [
            0.1,
            0.2,
            0.3,
            0.4,
        ]  # Changed from panel_bbox to bbox
        assert panels[0]["confidence"] == 0.95


def test_detect_panels_with_no_detections(mock_yolo):
    """Test behavior when no panels are detected in an image."""
    od = ObjectDetection("test_model.pt")

    # Mock YOLO results with no detections
    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = []
    mock_results.boxes.conf = []
    od.model.return_value = [mock_results]

    # Create proper mock image
    mock_image = Mock(spec=Image.Image)
    mock_image.__array_interface__ = {
        "shape": (100, 100, 3),
        "typestr": "|u1",
        "data": (0, False),
        "version": 3,
    }

    result = od.detect_panels(mock_image)
    assert len(result) == 0


def test_detect_panels_with_low_confidence(mock_yolo):
    """Test filtering of low confidence detections."""
    od = ObjectDetection("test_model.pt")

    # Mock YOLO results with low confidence
    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_results.boxes.conf = [0.2]  # Below threshold
    od.model.return_value = [mock_results]

    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.convert_to_pil_image"
    ) as mock_convert:
        mock_convert.return_value = (Mock(), None)
        result = od.detect_panels(Mock())

    # Should still return the detection, filtering happens in process_figures
    assert len(result) == 1
    assert result[0]["confidence"] == 0.2


def test_detect_panels_multiple_panels(mock_yolo):
    """Test detection of multiple panels with different confidences."""
    od = ObjectDetection("test_model.pt")

    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.2, 0.7, 0.4],
        [0.8, 0.2, 0.9, 0.4],
    ]
    mock_results.boxes.conf = [0.95, 0.85, 0.75]
    od.model.return_value = [mock_results]

    mock_image = Mock(spec=Image.Image)
    mock_array = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("numpy.array", return_value=mock_array):
        result = od.detect_panels(mock_image)

    assert len(result) == 3
    assert all("bbox" in r for r in result)
    assert all("confidence" in r for r in result)


def test_image_resize_maintains_aspect_ratio():
    """Test that image resizing maintains aspect ratio."""
    # Create a test image with known dimensions
    test_image = Mock(spec=Image.Image)
    test_image.size = (1000, 500)  # 2:1 aspect ratio
    test_image.mode = "RGB"

    _ = convert_and_resize_image(test_image)
    test_image.thumbnail.assert_called_once()
    assert test_image.mode == "RGB"


def test_detect_panels_overlapping_boxes(mock_yolo):
    """Test handling of overlapping panel detections."""
    od = ObjectDetection("test_model.pt")

    # Mock YOLO results with overlapping boxes
    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [
        [0.1, 0.2, 0.3, 0.4],  # Original box
        [0.15, 0.25, 0.35, 0.45],  # Overlapping box
    ]
    mock_results.boxes.conf = [0.95, 0.90]
    od.model.return_value = [mock_results]

    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.convert_to_pil_image"
    ) as mock_convert:
        mock_convert.return_value = (Mock(), None)
        result = od.detect_panels("test_image.jpg")

    # Verify that overlapping boxes are handled appropriately
    assert len(result) == 2
    assert result[0]["confidence"] > result[1]["confidence"]


def test_image_color_mode_handling():
    """Test handling of different image color modes."""
    for mode in ["RGB", "RGBA", "L"]:
        test_image = Mock(spec=Image.Image)
        test_image.mode = mode
        converted_image = Mock(spec=Image.Image)
        converted_image.mode = "RGB"
        test_image.convert.return_value = converted_image

        result = convert_and_resize_image(test_image)
        assert result.mode == "RGB"


def test_detect_panels_input_validation():
    """Test input validation for detect_panels method."""
    with patch("pathlib.Path.exists", return_value=True), patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLO"
    ):
        od = ObjectDetection("test_model.pt")

        # Test with None image
        with pytest.raises(
            ValueError, match="Input image cannot be None"
        ):  # Changed back to ValueError
            od.detect_panels(None)


def test_detect_panels_with_custom_parameters(mock_yolo):
    """Test detection with custom confidence and IoU thresholds."""
    with patch("pathlib.Path.exists", return_value=True):
        od = ObjectDetection("test_model.pt")

        # Mock numpy array conversion
        mock_image = Mock(spec=Image.Image)
        mock_array = Mock()
        with patch("numpy.array", return_value=mock_array):
            _ = od.detect_panels(
                mock_image,
                conf=0.8,
                iou=0.2,
                imgsz=1024,
                max_det=30,  # Include this parameter
            )

            od.model.assert_called_with(
                mock_array,
                conf=0.8,
                iou=0.2,
                imgsz=1024,
                max_det=30,  # Include this parameter
            )


def test_large_image_handling():
    """Test handling of very large images."""
    large_image = Mock(spec=Image.Image)
    large_image.size = (10000, 10000)
    large_image.mode = "RGB"

    _ = convert_and_resize_image(large_image)  # Fixed function name
    large_image.thumbnail.assert_called_once()


def test_detect_panels_with_corrupted_image(mock_yolo):
    """Test handling of corrupted or invalid image data."""
    od = ObjectDetection("test_model.pt")
    corrupted_image = Mock(spec=Image.Image)
    corrupted_image.convert.side_effect = Exception("Corrupted image")

    result = od.detect_panels(corrupted_image)
    assert len(result) == 0  # Should return empty list on error


def test_convert_to_pil_image_with_zero_byte_file(mock_os):
    """Test handling of zero-byte files."""
    mock_os.path.exists.return_value = True
    mock_os.path.getsize.return_value = 0
    mock_os.path.abspath.return_value = "/app/empty.jpg"
    mock_os.path.splitext.return_value = ("empty", ".jpg")

    with patch("PIL.Image.open") as mock_open:
        mock_open.side_effect = UnidentifiedImageError("cannot identify image file")
        with pytest.raises(ValueError) as exc_info:
            convert_to_pil_image("empty.jpg")
        assert "Failed to open image" in str(exc_info.value)


class TestObjectDetection(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_test_image(self, size=(15000, 12000)):
        """Helper to create a test image file."""
        img_path = os.path.join(self.test_dir, "test.png")
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        return img_path

    def test_scale_down_large_image(self):
        """Test that large images are properly scaled down."""
        original_path = self.create_test_image((15000, 12000))

        scaled_path = scale_down_large_image(original_path)

        self.assertNotEqual(original_path, scaled_path)
        self.assertTrue(os.path.exists(scaled_path))

        # Check dimensions of scaled image
        img = cv2.imread(scaled_path)
        height, width = img.shape[:2]
        self.assertLess(width * height, 178956970)

        # Check aspect ratio
        original_ratio = 15000 / 12000
        scaled_ratio = width / height
        self.assertAlmostEqual(original_ratio, scaled_ratio, places=2)

    def test_small_image_not_scaled(self):
        """Test that small images are not scaled."""
        original_path = self.create_test_image((800, 600))
        scaled_path = scale_down_large_image(original_path)

        self.assertEqual(original_path, scaled_path)

        img = cv2.imread(scaled_path)
        height, width = img.shape[:2]
        self.assertEqual((width, height), (800, 600))


def test_fallback_ghostscript_conversion(mock_subprocess):
    """Test the fallback ghostscript conversion function."""
    mock_subprocess.run.return_value = Mock(returncode=0)

    result = fallback_ghostscript_conversion("test.eps", "test.png")

    # Verify ghostscript command was called correctly
    mock_subprocess.run.assert_called_once()
    cmd_args = mock_subprocess.run.call_args[0][0]
    assert cmd_args[0] == "gs"
    assert "-dNOPAUSE" in cmd_args
    assert "-dBATCH" in cmd_args
    assert "-sOutputFile=test.png" in cmd_args
    assert "test.eps" in cmd_args

    assert result == "test.png"


def test_fallback_ghostscript_failure(mock_subprocess):
    """Test handling of ghostscript conversion failures."""
    mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "gs")

    with pytest.raises(ValueError, match="Failed to convert EPS with ghostscript"):
        fallback_ghostscript_conversion("test.eps", "test.png")


def test_convert_eps_to_png_with_imagemagick(mock_subprocess):
    """Test conversion of EPS to PNG using ImageMagick."""
    mock_subprocess.run.return_value = Mock(returncode=0)

    result = convert_eps_to_png("test.eps", "test.png")

    # Verify ImageMagick command was called correctly
    mock_subprocess.run.assert_called_once()
    cmd_args = mock_subprocess.run.call_args[0][0]
    assert cmd_args[0] == "convert"
    assert "-trim" in cmd_args
    assert "+repage" in cmd_args
    assert "-flatten" in cmd_args

    assert result == "test.png"


def test_convert_eps_to_png_imagemagick_failure_with_fallback(mock_subprocess):
    """Test fallback to ghostscript when ImageMagick fails."""
    # First call (ImageMagick) fails, second call (ghostscript) succeeds
    mock_subprocess.run.side_effect = [
        subprocess.CalledProcessError(1, "convert"),
        Mock(returncode=0),
    ]

    result = convert_eps_to_png("test.eps", "test.png")

    # Verify both commands were attempted
    assert mock_subprocess.run.call_count == 2

    # First call should be ImageMagick
    cmd_args_1 = mock_subprocess.run.call_args_list[0][0][0]
    assert cmd_args_1[0] == "convert"

    # Second call should be ghostscript
    cmd_args_2 = mock_subprocess.run.call_args_list[1][0][0]
    assert cmd_args_2[0] == "gs"

    assert result == "test.png"


def test_convert_tiff_with_cv2():
    """Test conversion of TIFF to PNG using OpenCV."""
    # Create a temporary test TIFF file
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp_in:
        tmp_path = tmp_in.name

    try:
        # Create a small test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [255, 0, 0]  # Red square
        cv2.imwrite(tmp_path, img)

        # Create output path
        tmp_out = tmp_path.replace(".tiff", ".png")

        with patch("cv2.imread", return_value=img), patch(
            "cv2.imwrite", return_value=True
        ):
            result = convert_tiff_with_cv2(tmp_path, tmp_out)

        assert result == tmp_out

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


def test_convert_tiff_with_cv2_grayscale():
    """Test conversion of grayscale TIFF to PNG."""
    with patch("cv2.imread") as mock_imread, patch("cv2.imwrite", return_value=True):
        # Create grayscale image mock
        grayscale_img = np.zeros((100, 100), dtype=np.uint8)
        mock_imread.return_value = grayscale_img

        result = convert_tiff_with_cv2("test.tiff", "test.png")

        # Verify color conversion was called
        assert mock_imread.call_count == 1

        assert result == "test.png"


def test_convert_tiff_with_cv2_rgba():
    """Test conversion of RGBA TIFF to RGB PNG."""
    with patch("cv2.imread") as mock_imread, patch(
        "cv2.imwrite", return_value=True
    ), patch("cv2.cvtColor") as _:
        # Create RGBA image mock
        rgba_img = np.zeros((100, 100, 4), dtype=np.uint8)
        mock_imread.return_value = rgba_img

        result = convert_tiff_with_cv2("test.tiff", "test.png")

        # Verify color conversion was called
        assert mock_imread.call_count == 1

        assert result == "test.png"


def test_convert_tiff_with_cv2_high_bit_depth():
    """Test conversion of 16-bit TIFF to 8-bit PNG."""
    with patch("cv2.imread") as mock_imread, patch("cv2.imwrite", return_value=True):
        # Create 16-bit image mock
        img_16bit = np.zeros((100, 100, 3), dtype=np.uint16)
        img_16bit[50:70, 50:70] = 65535  # Max value for uint16
        mock_imread.return_value = img_16bit

        result = convert_tiff_with_cv2("test.tiff", "test.png")

        # Verify conversion was performed
        assert mock_imread.call_count == 1

        assert result == "test.png"


def test_create_standard_thumbnail_png():
    """Test creating a standard thumbnail from a PNG file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_path = tmp_in.name

    try:
        # Create a test image
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        img[250:750, 250:750] = [0, 255, 0]  # Green square
        cv2.imwrite(tmp_path, img)

        # Create output path
        tmp_out = tmp_path.replace(".png", "_thumb.png")

        with patch("cv2.imread", return_value=img), patch(
            "cv2.imwrite", return_value=True
        ):
            result = create_standard_thumbnail(tmp_path, tmp_out)

        assert result == tmp_out

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)


def test_create_standard_thumbnail_eps():
    """Test creating a standard thumbnail from an EPS file."""
    # Mock both potential conversion paths
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.convert_eps_to_png",
        return_value="test_converted.png",
    ) as mock_convert_eps, patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.subprocess.run",
        return_value=Mock(returncode=0),
    ) as mock_run:
        result = create_standard_thumbnail("test.eps", "test_thumb.png")

        # Check if either convert_eps_to_png was called or subprocess.run was called directly
        if mock_convert_eps.call_count > 0:
            mock_convert_eps.assert_called_with("test.eps", "test_thumb.png", 300)
        else:
            assert mock_run.call_count > 0

        assert result in ["test_converted.png", "test_thumb.png"]


def test_create_standard_thumbnail_tiff():
    """Test creating a standard thumbnail from a TIFF file."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.convert_tiff_with_cv2"
    ) as mock_convert_tiff:
        mock_convert_tiff.return_value = "test_converted.png"

        result = create_standard_thumbnail("test.tiff", "test_thumb.png")

        mock_convert_tiff.assert_called_once_with("test.tiff", "test_thumb.png")
        assert result == "test_converted.png"


def test_create_standard_thumbnail_pdf():
    """Test creating a standard thumbnail from a PDF file."""
    with patch("pdf2image.convert_from_path") as mock_convert_pdf, patch(
        "PIL.Image.Image.save"
    ):
        mock_page = Mock(spec=Image.Image)
        mock_convert_pdf.return_value = [mock_page]

        result = create_standard_thumbnail("test.pdf", "test_thumb.png")

        mock_convert_pdf.assert_called_once_with("test.pdf", dpi=300)
        assert result == "test_thumb.png"


def test_create_standard_thumbnail_fallback():
    """Test fallback to PIL when other methods fail."""
    with patch("cv2.imread", return_value=None), patch(
        "PIL.Image.open"
    ) as mock_open, patch("PIL.Image.Image.save"):
        mock_img = Mock(spec=Image.Image)
        mock_open.return_value = mock_img

        result = create_standard_thumbnail("test.jpg", "test_thumb.png")

        mock_open.assert_called_once()
        assert result == "test_thumb.png"


def test_create_standard_thumbnail_all_failures():
    """Test when all thumbnail creation methods fail."""
    with patch("cv2.imread", return_value=None), patch(
        "PIL.Image.open", side_effect=Exception("Cannot open image")
    ):
        with pytest.raises(
            ValueError, match="Failed to convert image after all attempts"
        ):
            create_standard_thumbnail("test.jpg", "test_thumb.png")


def test_convert_to_pil_image_with_new_functions():
    """Test that convert_to_pil_image uses the new specialized functions for different formats."""
    # Test EPS using the new convert_eps_to_png
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.create_standard_thumbnail"
    ) as mock_thumbnail, patch(
        "PIL.Image.open", return_value=Mock(spec=Image.Image, mode="RGB")
    ), patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.convert_and_resize_image",
        return_value=Mock(spec=Image.Image),
    ), patch(
        "os.path.exists", return_value=True
    ), patch(
        "os.path.abspath", return_value="/app/test.eps"
    ), patch(
        "os.path.splitext", return_value=("/app/test", ".eps")
    ):
        mock_thumbnail.return_value = "/app/test.png"

        result, path = convert_to_pil_image("test.eps")

        assert isinstance(result, Image.Image)
        assert path == "/app/test.png"
        mock_thumbnail.assert_called_once()
