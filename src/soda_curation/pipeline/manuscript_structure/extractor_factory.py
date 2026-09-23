"""Pick the structure extractor that matches the input archive's format."""

import logging
import zipfile
from typing import Iterable, Union

from .manuscript_meca_parser import MANIFEST_FILE_NAME, MecaStructureExtractor
from .manuscript_xml_parser import XMLStructureExtractor

logger = logging.getLogger(__name__)

StructureExtractor = Union[XMLStructureExtractor, MecaStructureExtractor]


def is_meca_archive(namelist: Iterable[str]) -> bool:
    """Return True if the archive is a MECA package.

    A manifest at the root is unambiguous: legacy eJP archives carry exactly one root
    XML, named after the manuscript.
    """
    return MANIFEST_FILE_NAME in set(namelist)


def create_structure_extractor(zip_path: str, extract_dir: str) -> StructureExtractor:
    """Build the extractor matching the archive format at `zip_path`."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        meca = is_meca_archive(zip_ref.namelist())

    if meca:
        logger.info(f"Detected MECA archive: {zip_path}")
        return MecaStructureExtractor(zip_path, extract_dir)

    logger.info(f"Detected legacy eJP archive: {zip_path}")
    return XMLStructureExtractor(zip_path, extract_dir)
