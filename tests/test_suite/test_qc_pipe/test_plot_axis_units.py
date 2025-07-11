import pytest

from soda_curation.qc.data_types import (
    AxisDefinition,
    AxisJustification,
    AxisUnit,
    PanelPlotAxisUnits,
    PlotAxisUnitsResult,
)
from soda_curation.qc.qc_tests.plot_axis_units import PlotAxisUnitsAnalyzer


class DummyModelAPI:
    def __init__(self, config):
        pass

    def generate_response(self, **kwargs):
        # Return a valid PlotAxisUnitsResult JSON string
        return PlotAxisUnitsResult(
            outputs=[
                PanelPlotAxisUnits(
                    panel_label="A",
                    is_a_plot="yes",
                    units_provided=[
                        AxisUnit(axis="x", answer="yes"),
                        AxisUnit(axis="y", answer="no"),
                    ],
                    justify_why_units_are_missing=[
                        AxisJustification(axis="y", justification="Not provided")
                    ],
                    unit_definition_as_provided=[
                        AxisDefinition(axis="x", definition="seconds")
                    ],
                )
            ]
        ).model_dump_json()


def test_plot_axis_units_pass_and_needed(monkeypatch):
    config = {"pipeline": {"plot_axis_units": {"openai": {}}}}
    analyzer = PlotAxisUnitsAnalyzer(config)
    # Patch the model_api to use dummy
    analyzer.model_api = DummyModelAPI(config)
    passed, result = analyzer.analyze_figure("Fig1", "", "")
    assert isinstance(result, PlotAxisUnitsResult)
    assert isinstance(result.outputs, list)
    panel = result.outputs[0]
    assert panel.is_a_plot == "yes"
    assert any(u.answer == "no" for u in panel.units_provided)
    # The test should be needed and not passed (since y axis is missing units)
    test_needed = any(u.answer == "no" for u in panel.units_provided)
    test_passed = all(u.answer in ("yes", "not needed") for u in panel.units_provided)
    assert test_needed is True
    assert test_passed is False
