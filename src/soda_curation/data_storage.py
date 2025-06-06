"""Utilities for storing and loading data objects for development and testing."""

import json
import logging
import pickle
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def save_figure_data(figure_data: List[Tuple[str, str, str]], output_file: str) -> None:
    """
    Save figure data to a JSON file.

    Args:
        figure_data: List of tuples containing (figure_label, base64_encoded_image, figure_caption)
        output_file: Path to save the data
    """
    try:
        # Convert to a list of dictionaries for better JSON serialization
        data_to_save = [
            {"figure_label": label, "image_data": image_data, "figure_caption": caption}
            for label, image_data, caption in figure_data
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f)

        logger.info(
            f"Saved figure data with {len(figure_data)} figures to {output_file}"
        )
    except Exception as e:
        logger.error(f"Error saving figure data: {str(e)}")


def load_figure_data(input_file: str) -> List[Tuple[str, str, str]]:
    """
    Load figure data from a JSON file.

    Args:
        input_file: Path to the saved data

    Returns:
        List of tuples containing (figure_label, base64_encoded_image, figure_caption)
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert back to list of tuples
        figure_data = [
            (item["figure_label"], item["image_data"], item["figure_caption"])
            for item in data
        ]

        logger.info(
            f"Loaded figure data with {len(figure_data)} figures from {input_file}"
        )
        return figure_data
    except Exception as e:
        logger.error(f"Error loading figure data: {str(e)}")
        return []


def save_zip_structure(zip_structure: Any, output_file: str) -> None:
    """
    Save zip structure object using pickle.

    Args:
        zip_structure: The ZipStructure object
        output_file: Path to save the data
    """
    try:
        with open(output_file, "wb") as f:
            pickle.dump(zip_structure, f)

        logger.info(f"Saved zip structure to {output_file}")
    except Exception as e:
        logger.error(f"Error saving zip structure: {str(e)}")


def load_zip_structure(input_file: str) -> Optional[Any]:
    """
    Load zip structure object from pickle file.

    Args:
        input_file: Path to the saved data

    Returns:
        ZipStructure object or None if loading failed
    """
    try:
        with open(input_file, "rb") as f:
            zip_structure = pickle.load(f)

        logger.info(f"Loaded zip structure from {input_file}")
        return zip_structure
    except Exception as e:
        logger.error(f"Error loading zip structure: {str(e)}")
        return None
