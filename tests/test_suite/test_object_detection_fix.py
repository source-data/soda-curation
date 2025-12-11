"""
Test suite to verify the object detection fix for the 'dict' object has no attribute 'shape' error.
"""
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.soda_curation.pipeline.match_caption_panel.object_detection import (
    ObjectDetection,
    convert_to_pil_image,
    create_object_detection,
)


class TestObjectDetectionFix:
    """Test the object detection fix for dictionary vs PIL Image issues."""

    def test_convert_to_pil_image_returns_pil_image(self, tmp_path):
        """Test that convert_to_pil_image always returns a PIL Image."""
        # Create a test image file
        test_image = Image.new("RGB", (100, 100), color="red")
        test_file = tmp_path / "test.png"
        test_image.save(test_file)

        # Test the function
        image, file_path = convert_to_pil_image(str(test_file))

        # Verify we get a PIL Image
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (100, 100)

    def test_convert_to_pil_image_handles_eps_fallback(self, tmp_path):
        """Test that EPS conversion fallback returns a PIL Image, not a file path."""
        # Create a mock EPS file that will fail conversion
        eps_file = tmp_path / "test.eps"
        eps_file.write_text("fake eps content")

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.object_detection.create_standard_thumbnail"
        ) as mock_thumbnail:
            # Mock the thumbnail creation to fail and return the original file path
            mock_thumbnail.side_effect = Exception("Conversion failed")

            # This should raise an error instead of returning a file path
            with pytest.raises(ValueError, match="Failed to convert or open image"):
                convert_to_pil_image(str(eps_file))

    def test_detect_panels_with_valid_pil_image(self):
        """Test that detect_panels works with a valid PIL Image."""
        # Create a mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.xyxyn = np.array([[0.1, 0.1, 0.9, 0.9]])
        mock_result.boxes.conf = np.array([0.8])
        mock_model.return_value = [mock_result]

        # Create ObjectDetection with mocked model
        detector = ObjectDetection.__new__(ObjectDetection)
        detector.model = mock_model

        # Create a test PIL Image
        test_image = Image.new("RGB", (100, 100), color="red")

        # Test detect_panels
        detections = detector.detect_panels(test_image)

        # Verify results
        assert len(detections) == 1
        assert detections[0]["confidence"] == 0.8
        assert detections[0]["bbox"] == [0.1, 0.1, 0.9, 0.9]

    def test_detect_panels_raises_error_with_dict(self):
        """Test that detect_panels raises an error when given a dictionary instead of PIL Image."""
        # Create a mock YOLO model
        mock_model = Mock()

        detector = ObjectDetection.__new__(ObjectDetection)
        detector.model = mock_model

        # Test with a dictionary - should now raise TypeError with our validation
        with pytest.raises(TypeError, match="Expected PIL Image, but received dict"):
            detector.detect_panels({"not": "an image"})

        # Model should NOT be called because validation happens before
        mock_model.assert_not_called()

    def test_detect_panels_raises_error_with_none(self):
        """Test that detect_panels raises an error when given None."""
        mock_model = Mock()
        detector = ObjectDetection.__new__(ObjectDetection)
        detector.model = mock_model

        with pytest.raises(ValueError, match="Input image cannot be None"):
            detector.detect_panels(None)

    def test_create_object_detection_with_mock_config(self, tmp_path):
        """Test create_object_detection with a mock model file."""
        # Create a mock model file
        model_file = tmp_path / "test_model.pt"
        model_file.write_bytes(b"fake model data")

        # Mock the Path.exists method to return True for our test file
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "src.soda_curation.pipeline.match_caption_panel.object_detection.YOLOv10"
            ) as mock_yolo:
                mock_yolo_instance = Mock()
                mock_yolo.return_value = mock_yolo_instance

                config = {"object_detection": {"model_path": str(model_file)}}

                detector = create_object_detection(config)

                # Verify the detector was created
                assert isinstance(detector, ObjectDetection)
                assert detector.model_path == str(model_file)
                mock_yolo.assert_called_once_with(str(model_file))

    def test_integration_convert_and_detect(self, tmp_path):
        """Integration test: convert_to_pil_image + detect_panels."""
        # Create a test image file
        test_image = Image.new("RGB", (200, 200), color="blue")
        test_file = tmp_path / "test.png"
        test_image.save(test_file)

        # Convert to PIL Image
        image, file_path = convert_to_pil_image(str(test_file))
        assert isinstance(image, Image.Image)

        # Mock YOLO model for detection
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes.xyxyn = np.array([[0.2, 0.2, 0.8, 0.8]])
        mock_result.boxes.conf = np.array([0.9])
        mock_model.return_value = [mock_result]

        # Create detector and test
        detector = ObjectDetection.__new__(ObjectDetection)
        detector.model = mock_model

        detections = detector.detect_panels(image)

        # Verify the full pipeline works
        assert len(detections) == 1
        assert detections[0]["confidence"] == 0.9
        assert detections[0]["bbox"] == [0.2, 0.2, 0.8, 0.8]
