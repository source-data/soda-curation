"""Regression: Panel objects built in _main_utils must serialize with CustomJSONEncoder.

Imports use the ``soda_curation`` namespace (as the deployed image does via
PYTHONPATH=/app/src), not ``src.soda_curation``. A ``src.``-prefixed import in
``_main_utils`` creates a second ``Panel`` class the encoder does not recognise.
"""

import json

from soda_curation._main_utils import ensure_single_panel_figure
from soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    CustomJSONEncoder,
    Figure,
    Panel,
    ZipStructure,
)


def test_ensure_single_panel_figure_creates_serializable_panel():
    figure = Figure(figure_label="Figure 1", img_files=[], sd_files=[], figure_caption="Caption")
    figure.panels = []

    ensure_single_panel_figure(figure)

    assert isinstance(figure.panels[0], Panel)
    zip_structure = ZipStructure()
    zip_structure.figures = [figure]
    output = json.loads(json.dumps(zip_structure, cls=CustomJSONEncoder))
    assert output["figures"][0]["panels"][0]["panel_label"] == "A"
