"""
This module provides functionality for detecting figure panels in scientific images.

It includes utilities for converting various image formats to PIL images,
as well as a class for performing object detection using the YOLOv10 model.
The module can handle different image formats including JPG, PNG, PDF, and EPS.
"""

import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
import os
import base64
import io
import subprocess

import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLOv10
import fitz  # PyMuPDF
import pdf2image

logger = logging.getLogger(__name__)

def convert_to_pil_image(file_path: str, dpi: int = 300) -> Image.Image:
    """
    Convert various image formats (PDF, EPS, TIFF, JPG, PNG) to a PIL image.

    This function handles different file formats and converts them to a PIL Image object.
    For PDF files, it converts the first page to an image. For EPS files, it uses
    Ghostscript for conversion.

    Args:
        file_path (str): The path to the image file.
        dpi (int): Dots per inch for high-resolution EPS conversion. Default is 300.

    Returns:
        PIL.Image: The converted PIL image.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is unsupported.
    """
    file_path = os.path.abspath(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        image = Image.open(file_path)
        image = convert_and_resize_image(image)
        return image
    elif file_ext == '.pdf':
        pages = pdf2image.convert_from_path(file_path, dpi=dpi)
        if pages:
            image = convert_and_resize_image(pages[0])
            return image
        else:
            raise ValueError("PDF conversion failed: no pages found")
    elif file_ext == '.eps':
        output_path = file_path.replace('.eps', '.png')
        command = [
            'gs',
            '-dNOPAUSE',
            '-dBATCH',
            '-sDEVICE=pngalpha',
            f'-r{dpi}',
            f'-sOutputFile={output_path}',
            file_path
        ]
        subprocess.run(command, check=True)
        image = Image.open(output_path)
        image = convert_and_resize_image(image)
        os.remove(output_path)  # Clean up the temporary PNG file
        return image
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def convert_and_resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
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
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
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

    def detect_panels(self, image_path: str, conf: float = 0.25, iou: float = 0.1, imgsz: int = 512, max_det: int = 30) -> List[Dict[str, Any]]:
        """
        Detect panels in the given image using YOLOv10.

        This method processes an image file, detects panels within it using the YOLOv10 model,
        and returns a list of detected panels with their properties.

        Args:
            image_path (str): Path to the input image file.
            conf (float): Confidence threshold for detection. Default is 0.25.
            iou (float): IoU threshold for non-max suppression. Default is 0.1.
            imgsz (int): Inference size for the model. Default is 512.
            max_det (int): Maximum number of detections. Default is 30.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing detected panel information.
                Each dictionary includes 'panel_label', 'panel_caption', and 'panel_bbox'.

        Raises:
            Exception: If there's an error during the detection process.
        """
        logger.info(f"Detecting panels in image: {image_path}")
        
        try:
            pil_image = convert_to_pil_image(image_path)
            # Convert PIL Image to numpy array
            np_image = np.array(pil_image)
            
            results = self.model(np_image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)
            
            panels = []
            for i, box in enumerate(results[0].boxes.xyxyn.tolist()):
                x1, y1, x2, y2 = box
                confidence = float(results[0].boxes.conf[i])
                
                panel_info = {
                    "panel_label": chr(65 + i),
                    "panel_caption": "TO BE ADDED IN LATER STEP",
                    "panel_bbox": [x1, y1, x2, y2]
                }
                panels.append(panel_info)
            
            logger.info(f"Detected {len(panels)} panels in {image_path}")
            return panels
        
        except Exception as e:
            logger.error(f"Error detecting panels in {image_path}: {str(e)}")
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
    relative_model_path = config.get('object_detection', {}).get('model_path', 'data/models/panel_detection_model_no_labels.pt')
    
    # Construct the absolute path to the model
    absolute_model_path = Path('/app') / relative_model_path
    
    logger.info(f"Loading model from: {absolute_model_path}")
    
    if not absolute_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {absolute_model_path}")
    
    return ObjectDetection(str(absolute_model_path))
