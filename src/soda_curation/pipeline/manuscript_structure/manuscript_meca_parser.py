"""
This module extracts manuscript structure from MECA archives.

MECA (Manuscript Exchange Common Approach) packages are what eJournalPress transfers
for revision-stage manuscripts. They differ from the legacy eJP archives in two ways
that matter here:

* they are flat -- every member sits at the archive root, with no directories;
* their ``article.xml`` is plain JATS and carries none of the ``object-type`` /
  ``object_id`` elements the legacy parser relies on.

``manifest.xml`` is therefore the only file map, and ``<item-title>`` the only reliable
figure label -- file names themselves are not parseable to extract figure labels.

Manifests can reference files that are not in the archive. Every href is resolved
against the archive namelist and dropped, with an entry in ``ZipStructure.errors``,
when it is missing.
"""

import logging
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

from lxml import etree
from mmqc_utils import document_to_html

from .exceptions import NoManuscriptFileError, NoXMLFileFoundError
from .manuscript_structure import Figure, ZipStructure

logger = logging.getLogger(__name__)

MANIFEST_FILE_NAME = "manifest.xml"
ARTICLE_FILE_NAME = "article.xml"

NS = {
    "m": "https://manuscriptexchange.org/schema/manifest",
    "xlink": "http://www.w3.org/1999/xlink",
}
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

# manifest item-types we map onto ZipStructure fields.
ITEM_TYPE_ARTICLE = "article"
ITEM_TYPE_FIGURE = "figure"
ITEM_TYPE_SOURCE_DATA = "additional_figure_data"
ITEM_TYPE_MERGED_PDF = "merged_pdf"
ITEM_TYPE_SUPPLEMENTAL = "supplemental"

# "Figure 1", "Figure 12" -- but not "Figure EV1" or "Figure + Figure Legend".
FIGURE_LABEL_RE = re.compile(r"^Figure\s+(\d+)$")
# "Figure 1 Source Data" -- the leading figure number is what attributes it.
SOURCE_DATA_FIGURE_RE = re.compile(r"^Figure\s+(\d+)\b")


class MecaStructureExtractor:
    """Extract manuscript structure from a MECA archive.

    Mirrors the public surface of
    :class:`~.manuscript_xml_parser.XMLStructureExtractor` so ``main.py`` can use
    either without knowing which archive format it was handed.
    """

    def __init__(self, zip_path: str, extract_dir: str):
        """Read the manifest, resolve it against the archive, and extract what it names.

        Args:
            zip_path: Path to the MECA ZIP file.
            extract_dir: Directory to extract contents to.
        """
        self.zip_path = zip_path
        self.extract_dir = Path(extract_dir)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        self.errors: List[str] = []

        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            self._names = set(zip_ref.namelist())

            if MANIFEST_FILE_NAME not in self._names:
                raise NoXMLFileFoundError(
                    f"No {MANIFEST_FILE_NAME} found in the root of the archive"
                )

            self.manifest = etree.fromstring(zip_ref.read(MANIFEST_FILE_NAME))
            self.manuscript_id = self._get_manuscript_id(zip_ref)

            self.manuscript_extract_dir = self.extract_dir / self.manuscript_id
            self.manuscript_extract_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created manuscript directory: {self.manuscript_extract_dir}")

            self._structure = self._build_structure()
            self._extract_members(zip_ref, self._referenced_members(self._structure))

    # -- construction helpers -------------------------------------------------

    def _get_manuscript_id(self, zip_ref: zipfile.ZipFile) -> str:
        """Read the manuscript id from article.xml, falling back to the archive name."""
        if ARTICLE_FILE_NAME in self._names:
            try:
                article = etree.fromstring(zip_ref.read(ARTICLE_FILE_NAME))
                msid = article.xpath("//article-id[@pub-id-type='manuscript']")
                if msid and msid[0].text:
                    return msid[0].text.strip()
            except etree.XMLSyntaxError:
                logger.warning(
                    "Could not parse %s; falling back to the archive name",
                    ARTICLE_FILE_NAME,
                    exc_info=True,
                )

        fallback = re.sub(r"-meca$", "", Path(self.zip_path).stem)
        logger.warning(
            "No manuscript id in %s, falling back to archive name %s",
            ARTICLE_FILE_NAME,
            fallback,
        )
        return fallback

    def _resolve(self, href: Optional[str]) -> Optional[str]:
        """Map a manifest href onto an actual archive member, or None if it is absent.

        File names can carry significant whitespace (``"Figure 2 .tif"``), so the exact
        href is tried first and percent-decoding only as a fallback.
        """
        if not href:
            return None
        if href in self._names:
            return href
        decoded = unquote(href)
        if decoded in self._names:
            return decoded
        return None

    def _items(self, item_type: str) -> List[Tuple[str, Optional[str]]]:
        """Return (title, resolved href) for every item of the given type.

        Items whose file is missing from the archive yield a None href and are
        recorded in ``errors``.
        """
        items = []
        for item in self.manifest.xpath(
            f"//m:item[@item-type='{item_type}']", namespaces=NS
        ):
            title = (item.findtext("m:item-title", "", NS) or "").strip()
            resolved = None
            for instance in item.xpath("m:instance", namespaces=NS):
                resolved = self._resolve(instance.get(XLINK_HREF))
                if resolved is not None:
                    break
            if resolved is None:
                hrefs = [
                    i.get(XLINK_HREF) for i in item.xpath("m:instance", namespaces=NS)
                ]
                message = (
                    f"Manifest item '{title or item_type}' references "
                    f"{hrefs} which is not in the archive"
                )
                logger.warning(message)
                self.errors.append(message)
            items.append((title, resolved))
        return items

    def _build_structure(self) -> ZipStructure:
        """Assemble the ZipStructure from the manifest."""
        docx = self._get_docx()
        pdf = self._get_single(ITEM_TYPE_MERGED_PDF)
        figures = self._get_figures()
        appendix = self._get_appendix()
        non_associated = self._assign_source_data(figures)

        return ZipStructure(
            manuscript_id=self.manuscript_id,
            xml=ARTICLE_FILE_NAME,
            docx=docx,
            pdf=pdf or "",
            appendix=appendix,
            figures=figures,
            _full_appendix=[],
            non_associated_sd_files=non_associated,
            errors=list(self.errors),
        )

    def _get_single(self, item_type: str) -> Optional[str]:
        """Return the resolved href of the first item of a type, if any."""
        for _, href in self._items(item_type):
            if href is not None:
                return href
        return None

    def _get_docx(self) -> str:
        """Return the manuscript document, which the pipeline cannot run without."""
        docx = self._get_single(ITEM_TYPE_ARTICLE)
        if docx is None:
            raise NoManuscriptFileError(
                "No manuscript file found in the MECA manifest "
                f"(item-type='{ITEM_TYPE_ARTICLE}')"
            )
        return docx

    def _get_appendix(self) -> List[str]:
        """Return appendix files.

        Mirrors the legacy parser's ``label='Appendix'`` filter, which keeps
        supplementary tables and datasets out of the appendix.
        """
        return [
            href
            for title, href in self._items(ITEM_TYPE_SUPPLEMENTAL)
            if href is not None and title.lower().startswith("appendix")
        ]

    def _get_figures(self) -> List[Figure]:
        """Return the figures named by the manifest, skipping EV figures."""
        figures = []
        for title, href in self._items(ITEM_TYPE_FIGURE):
            if href is None:
                continue

            # Check for EV before normalising: normalisation keeps only the digits,
            # so "Figure EV3" would silently become "Figure 3".
            if "EV" in title:
                logger.info(f"Skipping EV figure: {title}")
                continue

            match = FIGURE_LABEL_RE.match(title)
            if match:
                label = f"Figure {int(match.group(1))}"
            else:
                # Some packages ship every figure combined in one file, titled e.g.
                # "Figure + Figure Legend". Keep it as a single figure under its own
                # title rather than guessing at figure boundaries.
                label = title
                message = (
                    f"Figure item title '{title}' ({href}) is not of the form "
                    "'Figure N'; emitting it as a single figure"
                )
                logger.warning(message)
                self.errors.append(message)

            logger.info(f"Processing figure: {label}")
            figures.append(
                Figure(
                    figure_label=label,
                    img_files=[href],
                    sd_files=[],
                    figure_caption="",
                    panels=[],
                )
            )
        return figures

    def _assign_source_data(self, figures: List[Figure]) -> List[str]:
        """Attach source data to figures by title, returning what stays unattached.

        Aggregate files ("Figure Source Data", "Figure EV1-6 Source Data") name no
        single figure and become ``non_associated_sd_files``.
        """
        by_label: Dict[str, Figure] = {fig.figure_label: fig for fig in figures}
        non_associated = []

        for title, href in self._items(ITEM_TYPE_SOURCE_DATA):
            if href is None:
                continue

            match = SOURCE_DATA_FIGURE_RE.match(title)
            figure = by_label.get(f"Figure {int(match.group(1))}") if match else None
            if figure is not None:
                figure.sd_files.append(href)
            else:
                non_associated.append(href)

            if href.lower().endswith(".rar"):
                # RAR cannot be opened by the stdlib, so the file is still handed to
                # data4rev-flow but never gets panel-level assignment.
                message = (
                    f"Source data '{href}' is a RAR archive; its contents cannot be "
                    "inspected, so no panel-level source data will be assigned"
                )
                logger.warning(message)
                self.errors.append(message)

        return non_associated

    def _referenced_members(self, structure: ZipStructure) -> List[str]:
        """Return the archive members the structure actually points at."""
        members = [ARTICLE_FILE_NAME, MANIFEST_FILE_NAME, structure.docx]
        if structure.pdf:
            members.append(structure.pdf)
        members.extend(structure.appendix)
        members.extend(structure.non_associated_sd_files)
        for figure in structure.figures:
            members.extend(figure.img_files)
            members.extend(figure.sd_files)
        return [m for m in dict.fromkeys(members) if m in self._names]

    def _extract_members(self, zip_ref: zipfile.ZipFile, members: List[str]) -> None:
        """Extract the given members flat into the manuscript directory.

        MECA archives have no directories, so members are written under their own
        names with no prefix stripping.
        """
        for member in members:
            target_path = self.manuscript_extract_dir / member
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
        logger.info(f"Extracted {len(members)} files to {self.manuscript_extract_dir}")

    # -- public surface -------------------------------------------------------

    def get_full_path(self, relative_path: str) -> Path:
        """Get full path in extraction directory."""
        return self.manuscript_extract_dir / relative_path

    def extract_structure(self) -> ZipStructure:
        """Return the manuscript structure read from the manifest."""
        return self._structure

    def extract_docx_content(self, docx_path: str) -> str:
        """Extract content from the manuscript file as cleaned HTML.

        Args:
            docx_path: Path to the manuscript file, relative to the extract dir.

        Returns:
            str: Cleaned HTML extracted from the file.

        Raises:
            NoManuscriptFileError: If the file is not found or extraction fails.
        """
        try:
            full_path = self.manuscript_extract_dir / docx_path
            if not full_path.exists():
                raise NoManuscriptFileError(f"Manuscript file not found at {full_path}")

            logger.info(
                f"Extracting content from {full_path.suffix.lower()} file "
                "using mmqc_utils.document_to_html"
            )
            return document_to_html(full_path)

        except NoManuscriptFileError:
            raise
        except Exception as e:
            logger.error(f"Error extracting content from {docx_path}: {str(e)}")
            raise NoManuscriptFileError(
                f"Failed to extract content from {docx_path}: {str(e)}"
            )
