"""Tests for MecaStructureExtractor and archive-format dispatch."""

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.soda_curation.pipeline.manuscript_structure.exceptions import (
    NoManuscriptFileError,
    NoXMLFileFoundError,
)
from src.soda_curation.pipeline.manuscript_structure.extractor_factory import (
    create_structure_extractor,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_meca_parser import (
    MecaStructureExtractor,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)

MSID = "EMBOJ-2026-124104R"

ARTICLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="doi">10.15252/embj.2026124104</article-id>
      <article-id pub-id-type="manuscript">{msid}</article-id>
    </article-meta>
  </front>
</article>
"""

ARTICLE_XML_NO_MSID = """<?xml version="1.0" encoding="UTF-8"?>
<article><front><article-meta/></front></article>
"""

MANIFEST_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<manifest manifest-version="1.0" '
    'xmlns="https://manuscriptexchange.org/schema/manifest" '
    'xmlns:xlink="http://www.w3.org/1999/xlink">\n'
)


def _manifest(items):
    """Build a MECA manifest from (item_type, title, href) triples."""
    parts = [MANIFEST_HEAD]
    for idx, (item_type, title, href) in enumerate(items):
        parts.append(
            f'  <item id="ejp-{idx:03d}" item-type="{item_type}" item-version="0">\n'
            f"    <item-title>{title}</item-title>\n"
            f'    <instance media-type="application/octet-stream" '
            f'xlink:href="{href}" />\n'
            f"  </item>\n"
        )
    parts.append("</manifest>\n")
    return "".join(parts)


@pytest.fixture
def make_meca_zip(tmp_path):
    """Create a flat MECA archive from a list of manifest items.

    `omit` names hrefs that appear in the manifest but are left out of the archive --
    which is what real eJP packages do for most of their source data.
    """

    def _make(items, omit=(), article_xml=ARTICLE_XML, name="EMBOJ-2026-124104-meca"):
        zip_path = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("transfer.xml", "<transfer/>")
            if article_xml is not None:
                zf.writestr("article.xml", article_xml.format(msid=MSID))
            zf.writestr("manifest.xml", _manifest(items))
            for _, _, href in items:
                if href not in omit:
                    zf.writestr(href, "x")
        return str(zip_path)

    return _make


@pytest.fixture
def extract_dir(tmp_path):
    d = tmp_path / "extract"
    d.mkdir()
    return str(d)


BASE_ITEMS = [
    ("article", "Manuscript Text", "Manuscript_Text.docx"),
    ("merged_pdf", "Merged PDF", "167196_1_merged.pdf"),
    ("figure", "Figure 1", "Figure 1.tif"),
]


def test_dispatch_picks_meca_extractor(make_meca_zip, extract_dir):
    """A manifest-bearing archive is routed to the MECA extractor."""
    zip_path = make_meca_zip(BASE_ITEMS)
    extractor = create_structure_extractor(zip_path, extract_dir)
    assert isinstance(extractor, MecaStructureExtractor)


def test_dispatch_picks_legacy_extractor(tmp_path, extract_dir):
    """An archive without a manifest still goes to the legacy extractor."""
    zip_path = tmp_path / "EMBOJ-DUMMY.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "EMBOJ-DUMMY.xml",
            """<?xml version="1.0"?><article><notes>
            <doc object-type="Manuscript Text"><object_id>Doc/ms.docx</object_id></doc>
            </notes></article>""",
        )
        zf.writestr("Doc/ms.docx", "x")
    extractor = create_structure_extractor(str(zip_path), extract_dir)
    assert isinstance(extractor, XMLStructureExtractor)


def test_manuscript_id_from_article_xml(make_meca_zip, extract_dir):
    """The msid comes from article.xml, not from the archive name."""
    zip_path = make_meca_zip(BASE_ITEMS)
    extractor = create_structure_extractor(zip_path, extract_dir)
    # The archive stem is EMBOJ-2026-124104-meca; the JATS id carries the revision.
    assert extractor.manuscript_id == MSID
    assert extractor.extract_structure().manuscript_id == MSID


def test_manuscript_id_falls_back_to_archive_name(make_meca_zip, extract_dir):
    """Without a JATS manuscript id, the archive stem is used minus `-meca`."""
    zip_path = make_meca_zip(BASE_ITEMS, article_xml=ARTICLE_XML_NO_MSID)
    extractor = create_structure_extractor(zip_path, extract_dir)
    assert extractor.manuscript_id == "EMBOJ-2026-124104"


def test_missing_manifest_raises(tmp_path, extract_dir):
    """MecaStructureExtractor refuses an archive with no manifest."""
    zip_path = tmp_path / "no-manifest.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("article.xml", ARTICLE_XML.format(msid=MSID))
    with pytest.raises(NoXMLFileFoundError):
        MecaStructureExtractor(str(zip_path), extract_dir)


def test_missing_manuscript_raises(make_meca_zip, extract_dir):
    """A manifest with no article item is a hard failure."""
    zip_path = make_meca_zip([("figure", "Figure 1", "Figure 1.tif")])
    with pytest.raises(NoManuscriptFileError):
        create_structure_extractor(zip_path, extract_dir)


def test_flat_paths_are_emitted_verbatim(make_meca_zip, extract_dir):
    """Paths are bare file names -- no directory prefix, no normalisation.

    Includes a name with significant trailing whitespace, which real packages ship.
    """
    items = BASE_ITEMS + [("figure", "Figure 2", "Figure 2 .tif")]
    zip_path = make_meca_zip(items)
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    assert structure.xml == "article.xml"
    assert structure.docx == "Manuscript_Text.docx"
    assert structure.pdf == "167196_1_merged.pdf"
    assert structure.figures[0].img_files == ["Figure 1.tif"]
    assert structure.figures[1].img_files == ["Figure 2 .tif"]

    emitted = [structure.docx, structure.pdf, *structure.appendix]
    for figure in structure.figures:
        emitted.extend(figure.img_files + figure.sd_files)
    assert all("/" not in path for path in emitted)


def test_figure_labels_come_from_item_titles(make_meca_zip, extract_dir):
    """Labels are read from item-title, since file names are not parseable."""
    items = [
        ("article", "Manuscript Text", "PUL1-MainText-Revision.docx"),
        ("figure", "Figure 1", "Roberts-Revision-Fig-1.eps"),
        ("figure", "Figure 2", "Roberts-Revision-Fig-2.eps"),
    ]
    zip_path = make_meca_zip(items)
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    assert [f.figure_label for f in structure.figures] == ["Figure 1", "Figure 2"]
    assert structure.figures[0].img_files == ["Roberts-Revision-Fig-1.eps"]


def test_ev_figures_are_skipped(make_meca_zip, extract_dir):
    """EV figures are excluded, and never collapse onto a main figure's label."""
    items = [
        ("article", "Manuscript Text", "Manuscript_Text.docx"),
        ("figure", "Figure 1", "Figure 1.tif"),
        ("figure", "Figure 2", "Figure 2.tif"),
        ("figure", "Figure 3", "Figure 3.tif"),
        ("figure", "Figure EV1", "Figure EV1.tif"),
        ("figure", "Figure EV3", "Figure EV3.tif"),
    ]
    zip_path = make_meca_zip(items)
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    # Regression guard: normalising "Figure EV3" would yield "Figure 3".
    assert [f.figure_label for f in structure.figures] == [
        "Figure 1",
        "Figure 2",
        "Figure 3",
    ]
    assert len(structure.figures) == 3


def test_source_data_attribution(make_meca_zip, extract_dir):
    """Per-figure source data attaches by title; aggregates stay unassociated."""
    items = [
        ("article", "Manuscript Text", "Manuscript_Text.docx"),
        ("figure", "Figure 1", "Figure 1.tif"),
        ("figure", "Figure 2", "Figure 2.tif"),
        ("additional_figure_data", "Figure 1 Source Data", "Source Data Fig. 1.zip"),
        ("additional_figure_data", "Figure 2 Source Data", "Source Data Fig. 2.zip"),
        ("additional_figure_data", "Figure EV1-6 Source Data", "EV Source Data.zip"),
        (
            "additional_figure_data",
            "Appendix Figure S1-S6 Source Data",
            "Appendix Source Data.zip",
        ),
        ("additional_figure_data", "Figure Source Data", "Roberts-SourceData.zip"),
    ]
    zip_path = make_meca_zip(items)
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    assert structure.figures[0].sd_files == ["Source Data Fig. 1.zip"]
    assert structure.figures[1].sd_files == ["Source Data Fig. 2.zip"]
    assert structure.non_associated_sd_files == [
        "EV Source Data.zip",
        "Appendix Source Data.zip",
        "Roberts-SourceData.zip",
    ]

    # Nothing is claimed twice.
    attached = [f for fig in structure.figures for f in fig.sd_files]
    assert not set(attached) & set(structure.non_associated_sd_files)


def test_missing_files_are_dropped_and_recorded(make_meca_zip, extract_dir):
    """Manifest entries with no matching archive member never reach the output.

    Three of the four reference packages declare source data they do not ship;
    emitting those paths would fail data4rev-flow's file validation.
    """
    items = [
        ("article", "Manuscript Text", "Manuscript_Text.docx"),
        ("figure", "Figure 1", "Figure 1.tif"),
        ("additional_figure_data", "Figure 1 Source Data", "Source Data Figure 1.zip"),
        ("additional_figure_data", "Figure Source Data", "Roberts-SourceData.zip"),
    ]
    zip_path = make_meca_zip(
        items, omit=("Source Data Figure 1.zip", "Roberts-SourceData.zip")
    )
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    assert structure.figures[0].sd_files == []
    assert structure.non_associated_sd_files == []
    assert any("Source Data Figure 1.zip" in e for e in structure.errors)
    assert any("Roberts-SourceData.zip" in e for e in structure.errors)


def test_combined_figure_emitted_as_single_entry(make_meca_zip, extract_dir):
    """A package shipping all figures in one file yields one figure, plus an error."""
    items = [
        ("article", "Manuscript Text", "Manuscript.docx"),
        ("figure", "Figure + Figure Legend", "Figure.pdf"),
    ]
    zip_path = make_meca_zip(items)
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    assert len(structure.figures) == 1
    assert structure.figures[0].figure_label == "Figure + Figure Legend"
    assert structure.figures[0].img_files == ["Figure.pdf"]
    assert any("Figure + Figure Legend" in e for e in structure.errors)


def test_rar_source_data_is_kept_and_flagged(make_meca_zip, extract_dir):
    """RAR source data still reaches the output, with its limitation recorded."""
    items = [
        ("article", "Manuscript Text", "Manuscript_Text.docx"),
        ("figure", "Figure 1", "Figure 1.tif"),
        ("additional_figure_data", "Figure 1 Source Data", "Source Data Fig. 1.rar"),
    ]
    zip_path = make_meca_zip(items)
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    assert structure.figures[0].sd_files == ["Source Data Fig. 1.rar"]
    assert any("RAR" in e for e in structure.errors)


def test_appendix_filter(make_meca_zip, extract_dir):
    """Only supplemental items titled as an appendix become the appendix."""
    items = [
        ("article", "Manuscript Text", "Manuscript_Text.docx"),
        ("supplemental", "Appendix Figures S1-S6", "Appendix.docx"),
        ("supplemental", "Supplemental Table", "DataSetEV1-4.zip"),
    ]
    zip_path = make_meca_zip(items)
    structure = create_structure_extractor(zip_path, extract_dir).extract_structure()

    assert structure.appendix == ["Appendix.docx"]


def test_extraction_is_flat_and_selective(make_meca_zip, extract_dir):
    """Referenced files land flat; unreferenced artwork is never written to disk."""
    items = [
        ("article", "Manuscript Text", "Manuscript_Text.docx"),
        ("figure", "Figure 1", "Figure 1.tif"),
        ("figure", "Figure EV1", "Figure EV1.tif"),
        ("data_set", "Dataset EV1", "Dataset EV1.xlsx"),
    ]
    zip_path = make_meca_zip(items)
    extractor = create_structure_extractor(zip_path, extract_dir)
    ms_dir = extractor.manuscript_extract_dir

    assert ms_dir == Path(extract_dir) / MSID
    assert (ms_dir / "Manuscript_Text.docx").exists()
    assert (ms_dir / "Figure 1.tif").exists()
    assert extractor.get_full_path("Figure 1.tif") == ms_dir / "Figure 1.tif"

    # EV figures and unmapped datasets are skipped, which keeps a 1.26 GB archive
    # from being unpacked in full.
    assert not (ms_dir / "Figure EV1.tif").exists()
    assert not (ms_dir / "Dataset EV1.xlsx").exists()


def test_extract_docx_content(make_meca_zip, extract_dir):
    """Manuscript conversion is delegated to mmqc_utils with the full path."""
    zip_path = make_meca_zip(BASE_ITEMS)
    extractor = create_structure_extractor(zip_path, extract_dir)

    target = "src.soda_curation.pipeline.manuscript_structure.manuscript_meca_parser.document_to_html"
    with patch(target, return_value="<p>text</p>") as mock_convert:
        assert extractor.extract_docx_content("Manuscript_Text.docx") == "<p>text</p>"
    mock_convert.assert_called_once_with(
        extractor.manuscript_extract_dir / "Manuscript_Text.docx"
    )


def test_extract_docx_content_missing_file(make_meca_zip, extract_dir):
    """A missing manuscript file surfaces as NoManuscriptFileError."""
    zip_path = make_meca_zip(BASE_ITEMS)
    extractor = create_structure_extractor(zip_path, extract_dir)
    with pytest.raises(NoManuscriptFileError):
        extractor.extract_docx_content("absent.docx")
