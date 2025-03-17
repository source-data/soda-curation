"""
This module provides functionality for object detection in scientific figures,
particularly for identifying panels within figure images.

It includes utilities for image conversion and resizing, as well as a class for
performing object detection using the YOLOv10 model.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pdf2image
from PIL import Image
from PIL.Image import DecompressionBombError
from ultralytics import YOLOv10

logger = logging.getLogger(__name__)


def scale_down_large_image(file_path: str, max_pixels: int = 178956970) -> str:
    """
    Scale down an image file if it exceeds the maximum pixel limit.
    Uses OpenCV for memory-efficient processing of large images.

    Args:
        file_path (str): Path to the image file
        max_pixels (int): Maximum number of pixels allowed (default: 178956970)

    Returns:
        str: Path to the scaled down image
    """
    try:
        # Read image dimensions first
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to open image: {file_path}")

        height, width = img.shape[:2]
        total_pixels = width * height

        if total_pixels <= max_pixels:
            return file_path

        # Calculate scaling factor
        scale_factor = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Create scaled image path
        scaled_path = (
            os.path.splitext(file_path)[0] + "_scaled" + os.path.splitext(file_path)[1]
        )

        # Resize using OpenCV
        img_scaled = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )
        cv2.imwrite(scaled_path, img_scaled)

        return scaled_path
    except Exception as e:
        raise ValueError(f"Failed to scale image: {str(e)}")


def convert_to_pil_image(file_path: str, dpi: int = 300) -> Tuple[Image.Image, str]:
    """
    Convert various image formats (PDF, EPS, TIFF, JPG, PNG) to a PIL image.
    Large images are automatically scaled down if they exceed the pixel limit.

    Args:
        file_path (str): The path to the image file.
        dpi (int): Dots per inch for high-resolution conversion. Default is 300.

    Returns:
        Tuple[PIL.Image, str]: The converted PIL image and the path to the new image file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is unsupported.
    """
    file_path = os.path.abspath(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    new_file_path = file_path

    if file_ext == ".pdf":
        pages = pdf2image.convert_from_path(file_path, dpi=dpi)
        if pages:
            new_file_path = file_path.replace(".pdf", ".png")
            pages[0].save(new_file_path, "PNG")
            image = pages[0]
        else:
            raise ValueError("PDF conversion failed: no pages found")
    elif file_ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        try:
            image = Image.open(file_path)
        except DecompressionBombError:
            # Scale down the image and try again
            scaled_path = scale_down_large_image(file_path)
            try:
                image = Image.open(scaled_path)
                new_file_path = scaled_path
            except Exception as e:
                raise ValueError(f"Failed to process scaled image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to open image: {str(e)}")
    elif file_ext in [".eps", ".ai"]:
        try:
            new_file_path = file_path.replace(file_ext, ".png")
            command = [
                "gs",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=pngalpha",
                f"-r{dpi}",
                f"-sOutputFile={new_file_path}",
                file_path,
            ]
            subprocess.run(command, check=True)
            image = Image.open(new_file_path)
        except DecompressionBombError:
            # Scale down the converted PNG and try again
            scaled_path = scale_down_large_image(new_file_path)
            try:
                image = Image.open(scaled_path)
                new_file_path = scaled_path
            except Exception as e:
                raise ValueError(f"Failed to process scaled image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to convert or open image: {str(e)}")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    image = convert_and_resize_image(image)
    return image, new_file_path


def convert_and_resize_image(image: Image.Image, max_size: int = 2048) -> Image.Image:
    """
    Convert the image to RGB format if needed and resize it to have a maximum dimension of max_size.

    This function ensures that the image is in RGB format and resizes it while maintaining
    the aspect ratio, so that the largest dimension does not exceed max_size.

    Args:
        image (PIL.Image): The input image.
        max_size (int): The maximum size for the image's width or height. Default is 1024.

    Returns:
        PIL.Image: The converted and resized PIL image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to maintain aspect ratio
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


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

        logger.info("Detecting panels in image")

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

    # Construct the absolute path to the model
    absolute_model_path = Path("/app") / relative_model_path

    logger.info(f"Loading model from: {absolute_model_path}")

    if not absolute_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {absolute_model_path}")

    return ObjectDetection(str(absolute_model_path))
