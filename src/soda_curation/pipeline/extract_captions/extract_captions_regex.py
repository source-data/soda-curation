"""Regex-based extraction of data from manuscript text."""

import re
from typing import Dict, List, Optional

import pypandoc
from bs4 import BeautifulSoup

from ..manuscript_structure.manuscript_structure import ZipStructure
from .extract_captions_base import FigureCaptionExtractor


def extract_sections(html_content: str, sections: List[str] = None) -> Dict[str, str]:
    """
    Extract sections from HTML-formatted manuscript text.

    Known sections include:
    - materials_methods
    - results
    - figure_legends
    - acknowledgments
    - references

    If text for a requested section is not found, None is returned for that section.

    Args:
        html_content (str): HTML-formatted manuscript text.
        sections (List[str], optional): List of sections to extract. If None, all known sections are extracted. Defaults to None.
    Returns:
        sections (Dict[str, str]): Dictionary of extracted sections, with section names as keys and extracted text as values.
    """
    # extraction logic is based on the code at https://github.com/source-data/curation-backend/blob/630d31db2cc24a82f37bab90f892746024c269b8/paper.php#L2448
    section_patterns = {
        "materials_methods": [
            r"^(mat(erial|erials|\.)?\s(and|&|,\s)?)?(met(hod|hods|\.))?[\.:]?$"
        ],
        "results": [r"^results?(\sand\sdiscussion[\.:]?)?$"],
        "figure_legends": [r"^fig(\.|ure|ures)?(\slegends?)?[\.:]?$"],
        "acknowledgments": [r"^acknowledge?ments?[\.:]?$"],
        "references": [r"^references?[\.:]?$"],
    }

    soup = BeautifulSoup(html_content, "html.parser")

    # If no specific sections are requested, attempt to extract all known sections.
    if not sections:
        sections = list(section_patterns.keys())

    extracted_sections = {}

    for section in sections:
        if section not in section_patterns:
            raise ValueError(f"Section '{section}' is not recognized.")

        patterns = section_patterns[section]
        return_text = None

        # Strategy 1: Title-based search
        titles = soup.find_all("title")
        for title in titles:
            for pattern in patterns:
                if re.search(pattern, title.get_text(), re.I):
                    return_text = title.parent.get_text()
                    break

        # Strategy 2: Sec-based search
        if return_text is None:
            secs = soup.find_all("sec", attrs={"level": "1"})
            for sec in secs:
                sectitle = sec.find("sectitle")
                if sectitle:
                    for pattern in patterns:
                        if re.search(pattern, sectitle.get_text(), re.I):
                            return_text = sec.get_text()
                            break

        # Strategy 3: Paragraph-based search
        if return_text is None:
            start = False
            result_text = []
            paragraphs = soup.find_all("p")
            if paragraphs:
                for paragraph in paragraphs:
                    text = paragraph.get_text(strip=True)

                    # Detect section start (only if matching patterns for current section)
                    for pattern in patterns:
                        if re.match(pattern, text, re.I) and text:
                            start = True
                            result_text = []

                    # Detect section end (use generic end patterns)
                    if len(text) < 30 and (
                        any(
                            re.match(end_pattern, text, re.I)
                            for end_patterns in section_patterns.values()
                            for end_pattern in end_patterns
                            if end_patterns != patterns  # Skip current section patterns
                        )
                    ):
                        start = False

                    if start:
                        result_text.append(paragraph.get_text())

            if result_text:
                return_text = "\n".join(result_text)

        extracted_sections[section] = return_text

    return extracted_sections


def extract_figures(
    figure_legend: str, expected_figure_labels: List[str]
) -> List[Dict[str, str]]:
    """
    Extract figure titles and captions from figure legend text.

    Fuzzily matches based on the expected figure labels, which are assumed to be unique. E.g. "Fig. A" and "Figure A" are considered equivalent.
    Args:
        figure_legend (str): Text of the figure legend that should contain the figure labels, titles, and captions.
        expected_figure_labels (List[str]): List of the labels of the figures expected to be in the figure legend text.
    Returns:
        figures (List[Dict[str, str]]): List of dictionaries, each containing the label, title, and caption of a figure. Contains one figure for each expected figure label. If no title or caption is found for a figure, the corresponding value in the dictionary is an empty string.
    """
    # extraction logic based on the code at https://github.com/source-data/curation-backend/blob/630d31db2cc24a82f37bab90f892746024c269b8/paper.php#L2321
    figures = {
        label: {
            "caption": "",
            "title": "",
        }
        for label in expected_figure_labels
    }

    label_re = re.compile(r"(Fig\.|Figure)\s*([A-Z0-9])+[:.]?")  # Fig./Figure A/B/1/2
    matches = label_re.finditer(figure_legend)
    match_idxes = list(sorted([match.start() for match in matches]))
    for i in range(len(match_idxes)):
        start = match_idxes[i]
        if i == len(match_idxes) - 1:
            end = len(figure_legend)
        else:
            end = match_idxes[i + 1]

        label_title_caption = figure_legend[start:end].strip()
        label_match = label_re.match(label_title_caption)
        label_indicator = label_match.group(2)
        figure = figures.get(
            f"Fig. {label_indicator}", figures.get(f"Figure {label_indicator}", None)
        )
        if figure:
            title_caption = (
                label_title_caption[label_match.end() :].strip().split("\n", 1)
            )
            figure["title"] = title_caption[0].strip() if len(title_caption) > 1 else ""
            figure["caption"] = (
                title_caption[1].strip()
                if len(title_caption) > 1
                else title_caption[0].strip()
            )
    return [
        {
            "label": label,
            "title": figure["title"],
            "caption": figure["caption"],
        }
        for label, figure in figures.items()
    ]


class FigureCaptionExtractorRegex(FigureCaptionExtractor):

    def _locate_figure_captions(
        self, doc_string: str, expected_figure_count: int, expected_figure_labels: str
    ) -> Optional[str]:
        """
        Locate figure captions in the document using AI.

        Args:
            doc_string (str): Document text content
            expected_figure_count (int): Expected number of figures
            expected_figure_labels (str): Expected figure labels

        Returns:
            Optional[str]: Located captions text or None
        """

        sections = extract_sections(doc_string, ["figure_legends"])
        figure_legend = sections.get("figure_legends", None)
        return figure_legend or ""

    def _extract_individual_captions(
        self, all_captions: str, expected_figure_count: int, expected_figure_labels: str
    ) -> Dict[str, str]:
        """
        Extract individual figure captions from the located captions text.

        Args:
            all_captions (str): Text containing all captions
            expected_figure_count (int): Expected number of figures
            expected_figure_labels (str): Expected figure labels

        Returns:
            Dict[str, str]: Dictionary of figure labels and their captions
        """
        figures = extract_figures(all_captions, expected_figure_labels)
        return {
            f["label"]: {
                "caption": f["caption"],
                "title": f["title"],
            }
            for f in figures
        }

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse JSON response containing figure captions.

        Args:
            response_text (str): The response text to parse.

        Returns:
            Dict[str, str]: Dictionary of figure labels and their captions.
        """
        return response_text

    def extract_captions(
        self,
        docx_path: str,
        zip_structure: ZipStructure,
        expected_figure_count: int,
        expected_figure_labels: str,
    ) -> ZipStructure:
        """
        Extract captions from the document and update ZipStructure.

        Args:
            docx_path (str): Path to DOCX file
            zip_structure (ZipStructure): Current ZIP structure
            expected_figure_count (int): Expected number of figures
            expected_figure_labels (str): Expected figure labels

        Returns:
            ZipStructure: Updated ZIP structure with extracted captions
        """
        manuscript_html = self._extract_docx_content(docx_path)
        figure_legend = self._locate_figure_captions(
            manuscript_html, expected_figure_count, expected_figure_labels
        )
        extracted_figures = self._extract_individual_captions(
            figure_legend, expected_figure_count, expected_figure_labels
        )
        for figure in zip_structure.figures:
            if figure.figure_label in extracted_figures:
                extracted_figure = extracted_figures[figure.figure_label]
                caption_text = extracted_figure["caption"]
                is_valid, rouge_score = self._validate_caption(docx_path, caption_text)

                figure.figure_caption = caption_text
                figure.caption_title = extracted_figure["title"]
                figure.rouge_l_score = rouge_score
                figure.possible_hallucination = not is_valid
        return zip_structure
