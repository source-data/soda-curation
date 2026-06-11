"""
This module provides functionality for object detection in scientific figures,
particularly for identifying panels within figure images.

Image conversion is delegated to ``mmqc_utils`` (wand/ImageMagick based), which
produces bounded JPEGs for all supported formats (EPS, AI, PDF, TIFF, PNG, JPG).
Panel detection is performed with the YOLOv10 model.
"""

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from mmqc_utils import convert_to_bounded_jpeg
from PIL import Image

try:
    # ultralytics renamed/flattened model entrypoints across versions.
    # Prefer YOLOv10 when available, otherwise fall back to YOLO.
    # Keep the name ``YOLOv10`` at module scope so tests can patch it.
    from ultralytics import YOLOv10
except ImportError:
    from ultralytics import YOLO as YOLOv10

logger = logging.getLogger(__name__)

MAX_IMAGE_DIMENSION = 2048
"""Maximum width/height (in pixels) for converted figure images."""


def convert_to_pil_image(file_path: str, dpi: int = 300) -> Tuple[Image.Image, str]:
    """
    Convert various image formats (PDF, EPS, AI, TIFF, JPG, PNG) to a PIL image.

    Conversion is delegated to ``mmqc_utils.convert_to_bounded_jpeg``, which
    rasterizes vector formats, flattens transparency onto a white background,
    and downscales so neither dimension exceeds ``MAX_IMAGE_DIMENSION``.
    The resulting JPEG is written next to the source file.

    Args:
        file_path (str): The path to the image file.
        dpi (int): Dots per inch used when rasterizing vector formats. Default is 300.

    Returns:
        Tuple[PIL.Image, str]: The converted PIL image and the path to the JPEG file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be converted.
    """
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        jpeg_bytes = convert_to_bounded_jpeg(
            file_path,
            rasterization_dpi=dpi,
            max_dimension=MAX_IMAGE_DIMENSION,
        )

        new_file_path = os.path.splitext(file_path)[0] + ".jpg"
        with open(new_file_path, "wb") as f:
            f.write(jpeg_bytes)

        image = Image.open(BytesIO(jpeg_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image, new_file_path

    except Exception as e:
        logger.error(f"Image conversion failed: {str(e)}")
        raise ValueError(f"Failed to convert or open image: {str(e)}")


class ObjectDetection:
    """
    A class for performing object detection on images using the YOLOv10 model.

    This class provides methods to load a YOLOv10 model and use it to detect
    panels within figure images.

    Attributes:
        model_path (str): Path to the YOLOv10 model file.
        model (YOLOv10): The loaded YOLOv10 model.
    """

    def __init__(self, model_path: str):
        """
        Initialize the ObjectDetection class.

        Args:
            model_path (str): Path to the YOLOv10 model file.
        """
        self.model_path = model_path
        self.model = YOLOv10(self.model_path)
        logger.info(f"Initialized ObjectDetection with model: {self.model_path}")

    def detect_panels(
        self,
        image: Image.Image,
        conf: float = 0.25,
        iou: float = 0.1,
        imgsz: int = 512,
        max_det: int = 30,
    ) -> List[Dict[str, float]]:
        """
        Detect panels in the given image using YOLOv10.

        This method processes an image, detects panels within it using the YOLOv10 model,
        and returns a list of detected panels with their properties.

        Args:
            image (Image.Image): The input PIL Image object.
            conf (float): Confidence threshold for detection. Default is 0.25.
            iou (float): IoU threshold for non-max suppression. Default is 0.1.
            imgsz (int): Inference size for the model. Default is 512.
            max_det (int): Maximum number of detections. Default is 30.

        Returns:
            List[Dict[str, float]]: List of detected panels with bbox and confidence

        Raises:
            Exception: If there's an error during the detection process.
        """
        if image is None:
            raise ValueError("Input image cannot be None")

        # Validate that image is a PIL Image, not a dict or other type
        if isinstance(image, dict):
            logger.error(f"detect_panels received a dict instead of PIL Image: {image}")
            raise TypeError(
                f"Expected PIL Image, but received dict with keys: {list(image.keys())}. "
                "This usually means detect_panels was called with detection results instead of an image."
            )

        # Check for PIL Image attributes
        if not (
            hasattr(image, "mode")
            and hasattr(image, "size")
            and hasattr(image, "convert")
        ):
            logger.error(
                f"detect_panels received invalid object: type={type(image).__name__}, "
                f"has_mode={hasattr(image, 'mode')}, has_size={hasattr(image, 'size')}, "
                f"has_convert={hasattr(image, 'convert')}, repr={repr(image)[:200]}"
            )
            raise TypeError(
                f"Expected PIL Image, but received {type(image).__name__}. "
                f"Object does not have required PIL Image attributes (mode, size, convert). "
                f"Type: {type(image)}, Module: {type(image).__module__}"
            )

        logger.info(
            f"Detecting panels in image - type: {type(image).__name__}, mode: {image.mode}, size: {image.size}"
        )

        try:
            np_image = np.array(image)
            results = self.model(
                np_image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det
            )

            detections = []
            for i, box in enumerate(results[0].boxes.xyxyn.tolist()):
                x1, y1, x2, y2 = box
                confidence = float(results[0].boxes.conf[i])

                detection_info = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                }
                detections.append(detection_info)

            logger.info(f"Detected {len(detections)} panels")
            return detections

        except Exception as e:
            logger.error(f"Error detecting panels: {str(e)}")
            return []


def create_object_detection(config: Dict[str, Any]) -> ObjectDetection:
    """
    Create an instance of ObjectDetection based on the configuration.

    This function reads the configuration to determine the path of the YOLOv10 model
    and creates an ObjectDetection instance with that model.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model path information.

    Returns:
        ObjectDetection: An instance of the ObjectDetection class.

    Raises:
        FileNotFoundError: If the specified model file is not found.
    """
    relative_model_path = config.get("object_detection", {}).get(
        "model_path", "data/models/panel_detection_model_no_labels.pt"
    )

    # Docker images use /app as the project root; locally, resolve from cwd.
    docker_model_path = Path("/app") / relative_model_path
    local_model_path = Path(relative_model_path)
    absolute_model_path = (
        docker_model_path if docker_model_path.exists() else local_model_path
    )

    logger.info(f"Loading model from: {absolute_model_path}")

    if not absolute_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {absolute_model_path}")

    return ObjectDetection(str(absolute_model_path))
