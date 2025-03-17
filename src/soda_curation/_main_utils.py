"""Utility functions supporting the main entry point."""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def validate_paths(
    zip_path: str, config_path: str, output_path: Optional[str] = None
) -> None:
    """
    Validate input paths and ensure output directory exists.

    Args:
        zip_path: Path to input ZIP file
        config_path: Path to configuration file
        output_path: Optional path to output JSON file

    Raises:
        ValueError: If required paths are not provided
        FileNotFoundError: If input files don't exist
    """
    if not zip_path:
        raise ValueError("ZIP path must be provided")
    if not config_path:
        raise ValueError("config path must be provided")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file {zip_path} does not exist")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")


def setup_extract_dir() -> Path:
    """
    Create and return a temporary extraction directory.

    Returns:
        Path to temporary extraction directory

    Note:
        The caller is responsible for cleaning up this directory using cleanup_extract_dir
    """
    temp_dir = tempfile.mkdtemp(prefix="soda_curation_")
    extract_dir = Path(temp_dir)
    logger.info(f"Created temporary extraction directory: {extract_dir}")
    return extract_dir


def write_output(output_json: str, output_path: str) -> None:
    """
    Write JSON output to file.

    Args:
        output_json: JSON string to write
        output_path: Path to output file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info(f"Output written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write output: {str(e)}")
        raise


def cleanup_extract_dir(extract_dir: Path) -> None:
    """
    Clean up temporary extraction directory.

    Args:
        extract_dir: Path to temporary extraction directory
    """
    if extract_dir and extract_dir.exists():
        try:
            shutil.rmtree(extract_dir)
            logger.info(f"Cleaned up temporary directory: {extract_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {str(e)}")


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text while preserving content.

    Args:
        text: HTML text to strip

    Returns:
        str: Text with HTML tags removed
    """
    if not text:
        return ""

    # Use BeautifulSoup to strip HTML tags
    try:
        # Only use BeautifulSoup if the text contains HTML-like content
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            # Get text while preserving spaces between elements
            return soup.get_text(" ", strip=True)
        return text
    except Exception:
        # Fallback to regex if BeautifulSoup fails
        return re.sub(r"<[^>]+>", " ", text)


def normalize_text(text: str, strip_html: bool = True) -> str:
    """
    Basic normalization:
      - optionally removes HTML tags
      - lowercases all text
      - removes HTML line breaks
      - collapses multiple whitespace into a single space
      - strips leading/trailing whitespace
    """
    if not text:
        return ""

    # (0) optionally strip HTML tags
    if strip_html:
        text = strip_html_tags(text)

    # (1) lowercase
    text = text.lower()

    # (2) remove line breaks or convert them into spaces
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # (3) remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    # (4) Special character normalization for sup tags (+/+, -/-)
    text = text.replace("+/+", "+/+")  # Preserve as is
    text = text.replace("-/-", "-/-")  # Preserve as is

    # (5) trim
    text = text.strip()

    return text


def exact_match_check(extracted_text: str, source_text: str) -> bool:
    """
    Check if normalized extracted text exists within normalized source text.

    Args:
        extracted_text: Text to check for hallucination
        source_text: Original source text to compare against

    Returns:
        bool: True if extract is found in source, False otherwise
    """
    if not extracted_text or not source_text:
        return False

    # Create normalized versions of both texts
    norm_extracted = normalize_text(extracted_text, strip_html=True)
    norm_source = normalize_text(source_text, strip_html=True)

    # Check if the normalized extracted text is in the normalized source
    return norm_extracted in norm_source


def fuzzy_match_score(extracted_text: str, source_text: str) -> float:
    """
    Calculate fuzzy match similarity score between extracted text and source text.

    Args:
        extracted_text: Text to check for hallucination
        source_text: Original source text to compare against

    Returns:
        float: Similarity score between 0-100
    """
    if not extracted_text or not source_text:
        return 0.0

    # Create normalized versions for fuzzy matching
    norm_extracted = normalize_text(extracted_text, strip_html=True)
    norm_source = normalize_text(source_text, strip_html=True)

    # Get the best partial ratio score
    return fuzz.partial_ratio(norm_extracted, norm_source)


def calculate_hallucination_score(extracted_text: str, source_text: str) -> float:
    """
    Calculate a 0-1 hallucination possibility score.
    0 = definitely not hallucinated, 1 = likely hallucinated

    Args:
        extracted_text: Text to check for hallucination
        source_text: Original source text to compare against

    Returns:
        float: Hallucination possibility score (0-1)
    """
    # Handle empty strings
    if not extracted_text or not source_text:
        return 1.0

    # First try exact match
    if exact_match_check(extracted_text, source_text):
        return 0.0

    # If not exact match, use fuzzy matching
    similarity = fuzzy_match_score(extracted_text, source_text)

    # Convert similarity (0-100) to hallucination score (0-1)
    # Higher similarity = lower hallucination score
    # If similarity is very high (≥98), treat as not hallucinated
    # if similarity >= 98.0:
    #     return 0.0

    return 1.0 - (similarity / 100.0)
