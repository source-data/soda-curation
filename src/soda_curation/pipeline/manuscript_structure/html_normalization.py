"""Turn pandoc HTML (and similar wrappers) into plain text for prompts and storage."""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup


def pandoc_html_to_plain_text(html: Optional[str]) -> str:
    """
    Strip tags and normalize whitespace from HTML produced by pandoc or PyPDF2 wrappers.

    Preserves paragraph breaks (block-level separation becomes newlines); collapses
    runs of blank lines to at most one empty line.
    """
    if html is None:
        return ""
    raw = str(html).strip()
    if not raw:
        return ""

    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    joined = "\n".join(out).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined
