"""Tests for object detection and mmqc_utils-based image conversion."""

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.soda_curation.pipeline.match_caption_panel.object_detection import (
    MAX_IMAGE_DIMENSION,
    ObjectDetection,
    convert_to_pil_image,
    create_object_detection,
)


def _jpeg_bytes(size=(64, 48), color=(255, 0, 0)) -> bytes:
    """Create in-memory JPEG bytes for mocking convert_to_bounded_jpeg."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_yolo():
    """Fixture to mock the YOLOv10 class."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLOv10"
    ) as mock:
        mock.return_value = Mock()
        yield mock


@pytest.fixture
def mock_bounded_jpeg():
    """Fixture to mock mmqc_utils.convert_to_bounded_jpeg in the module namespace."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.convert_to_bounded_jpeg"
    ) as mock:
        mock.return_value = _jpeg_bytes()
        yield mock


# ---------------------------------------------------------------------------
# convert_to_pil_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "file_ext", [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf", ".eps", ".ai"]
)
def test_convert_to_pil_image_supported_formats(file_ext, tmp_path, mock_bounded_jpeg):
    """All supported formats are converted through convert_to_bounded_jpeg."""
    source = tmp_path / f"test{file_ext}"
    source.write_bytes(b"placeholder")

    image, new_file_path = convert_to_pil_image(str(source))

    mock_bounded_jpeg.assert_called_once_with(
        str(source),
        rasterization_dpi=300,
        max_dimension=MAX_IMAGE_DIMENSION,
    )
    assert image.mode == "RGB"
    assert new_file_path == str(tmp_path / "test.jpg")


def test_convert_to_pil_image_writes_jpeg_next_to_source(tmp_path, mock_bounded_jpeg):
    """The produced JPEG bytes are persisted next to the source file."""
    source = tmp_path / "figure.eps"
    source.write_bytes(b"placeholder")

    _, new_file_path = convert_to_pil_image(str(source))

    out = tmp_path / "figure.jpg"
    assert str(out) == new_file_path
    assert out.is_file()
    assert out.read_bytes() == mock_bounded_jpeg.return_value
    # The written file is a valid JPEG
    reopened = Image.open(out)
    assert reopened.format == "JPEG"


def test_convert_to_pil_image_custom_dpi(tmp_path, mock_bounded_jpeg):
    """The dpi argument is forwarded as rasterization_dpi."""
    source = tmp_path / "figure.pdf"
    source.write_bytes(b"placeholder")

    convert_to_pil_image(str(source), dpi=150)

    mock_bounded_jpeg.assert_called_once_with(
        str(source),
        rasterization_dpi=150,
        max_dimension=MAX_IMAGE_DIMENSION,
    )


def test_convert_to_pil_image_file_not_found():
    """A missing source file raises FileNotFoundError before any conversion."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        convert_to_pil_image("/nonexistent/path/test.png")


def test_convert_to_pil_image_conversion_failure(tmp_path, mock_bounded_jpeg):
    """Conversion errors surface as ValueError."""
    source = tmp_path / "broken.tiff"
    source.write_bytes(b"placeholder")
    mock_bounded_jpeg.side_effect = Exception("wand exploded")

    with pytest.raises(ValueError, match="Failed to convert or open image"):
        convert_to_pil_image(str(source))


def test_convert_to_pil_image_real_image_roundtrip(tmp_path, mock_bounded_jpeg):
    """A real PIL image is returned with the dimensions of the JPEG payload."""
    mock_bounded_jpeg.return_value = _jpeg_bytes(size=(120, 80))
    source = tmp_path / "test.png"
    source.write_bytes(b"placeholder")

    image, _ = convert_to_pil_image(str(source))

    assert image.size == (120, 80)
    assert image.mode == "RGB"
    # Conversion result is usable as a numpy array (needed by YOLO)
    assert np.array(image).shape == (80, 120, 3)


# ---------------------------------------------------------------------------
# ObjectDetection
# ---------------------------------------------------------------------------


def test_object_detection_initialization(mock_yolo):
    """ObjectDetection wires the model path into YOLOv10."""
    od = ObjectDetection("test_model.pt")
    assert od.model_path == "test_model.pt"
    mock_yolo.assert_called_once_with("test_model.pt")


def test_detect_panels(mock_yolo):
    """detect_panels converts YOLO output into bbox/confidence dicts."""
    od = ObjectDetection("test_model.pt")

    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_results.boxes.conf = [0.95]
    od.model.return_value = [mock_results]

    mock_image = Mock(spec=Image.Image)
    mock_array = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("numpy.array", return_value=mock_array):
        result = od.detect_panels(mock_image)

    assert len(result) == 1
    assert result[0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert result[0]["confidence"] == 0.95


def test_detect_panels_with_no_detections(mock_yolo):
    """No detections yields an empty list."""
    od = ObjectDetection("test_model.pt")

    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = []
    mock_results.boxes.conf = []
    od.model.return_value = [mock_results]

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
    """Low-confidence detections are returned; filtering happens downstream."""
    od = ObjectDetection("test_model.pt")

    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_results.boxes.conf = [0.2]
    od.model.return_value = [mock_results]

    mock_image = Mock(spec=Image.Image)
    mock_array = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("numpy.array", return_value=mock_array):
        result = od.detect_panels(mock_image)

    assert len(result) == 1
    assert result[0]["confidence"] == 0.2


def test_detect_panels_multiple_panels(mock_yolo):
    """Multiple detections are all returned with bbox and confidence."""
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


def test_detect_panels_overlapping_boxes(mock_yolo):
    """Overlapping boxes are returned as-is (NMS happens in the model)."""
    od = ObjectDetection("test_model.pt")

    mock_results = Mock()
    mock_results.boxes.xyxyn.tolist.return_value = [
        [0.1, 0.2, 0.3, 0.4],
        [0.15, 0.25, 0.35, 0.45],
    ]
    mock_results.boxes.conf = [0.95, 0.90]
    od.model.return_value = [mock_results]

    test_image = Image.new("RGB", (100, 100), color="red")

    result = od.detect_panels(test_image)

    assert len(result) == 2
    assert result[0]["confidence"] > result[1]["confidence"]


def test_detect_panels_input_validation():
    """detect_panels rejects None and non-PIL inputs."""
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLOv10"
    ):
        od = ObjectDetection("test_model.pt")

        with pytest.raises(ValueError, match="Input image cannot be None"):
            od.detect_panels(None)

        with pytest.raises(TypeError, match="received dict"):
            od.detect_panels({"bbox": [0, 0, 1, 1]})


def test_detect_panels_with_custom_parameters(mock_yolo):
    """Custom detection parameters are forwarded to the model call."""
    od = ObjectDetection("test_model.pt")

    mock_image = Mock(spec=Image.Image)
    mock_array = Mock()
    with patch("numpy.array", return_value=mock_array):
        od.detect_panels(mock_image, conf=0.8, iou=0.2, imgsz=1024, max_det=30)

        od.model.assert_called_with(
            mock_array, conf=0.8, iou=0.2, imgsz=1024, max_det=30
        )


def test_detect_panels_with_corrupted_image(mock_yolo):
    """Errors during detection return an empty list."""
    od = ObjectDetection("test_model.pt")
    od.model.side_effect = Exception("model failure")

    corrupted_image = Mock(spec=Image.Image)
    result = od.detect_panels(corrupted_image)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# create_object_detection
# ---------------------------------------------------------------------------


def test_create_object_detection():
    """create_object_detection uses the configured model path."""
    config = {"object_detection": {"model_path": "custom_model.pt"}}
    with (
        patch(
            "src.soda_curation.pipeline.match_caption_panel.object_detection.Path.exists",
            return_value=True,
        ),
        patch(
            "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLOv10"
        ) as mock_yolo,
    ):
        od = create_object_detection(config)
    assert isinstance(od, ObjectDetection)
    assert od.model_path in ["custom_model.pt", "/app/custom_model.pt"]
    assert mock_yolo.called


def test_create_object_detection_default_path():
    """Default model path is used when config has none."""
    config = {}
    with (
        patch(
            "src.soda_curation.pipeline.match_caption_panel.object_detection.Path.exists",
            return_value=True,
        ),
        patch(
            "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLOv10"
        ) as mock_yolo,
    ):
        od = create_object_detection(config)
    assert isinstance(od, ObjectDetection)
    assert "panel_detection_model_no_labels.pt" in od.model_path
    assert mock_yolo.called


def test_create_object_detection_file_not_found():
    """A missing model file raises FileNotFoundError."""
    config = {}
    with patch(
        "src.soda_curation.pipeline.match_caption_panel.object_detection.Path.exists",
        return_value=False,
    ):
        with pytest.raises(FileNotFoundError):
            create_object_detection(config)
