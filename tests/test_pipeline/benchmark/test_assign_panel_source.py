"""Panel source assignment benchmark tests.

Runs from ground truth only, no manuscript package: figures and panel labels come from
the ground truth, source-data zips are stubs with the ground truth's file names (see
panel_source_gt.py). Manuscripts without source-data ground truth are skipped.
"""

import dataclasses
import logging
import time
import uuid
from typing import Any, Dict

import yaml

from src.soda_curation.pipeline.assign_panel_source.assign_panel_source_openai import (
    PanelSourceAssignerOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler
from src.soda_curation.pipeline.step_config import resolve_step_config

from .base_runner import BaseBenchmarkRunner
from .panel_source_gt import (
    figure_listing,
    figures_from_ground_truth,
    has_sd_ground_truth,
    score_figure,
    write_stub_package,
)

logger = logging.getLogger(__name__)

SCORES = ("precision", "recall", "shared_recall", "off_panel")


class PanelSourceBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for panel source assignment benchmark tests."""

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run panel source assignment test."""
        ground_truth = self._load_ground_truth(test_case["msid"])
        if not has_sd_ground_truth(ground_truth):
            logger.info(f"No source-data ground truth for {test_case['msid']}, skipped")
            return {"msid": test_case["msid"], "skipped": "no source-data ground truth"}

        cache_path = self.get_cache_path(test_case)
        cached_result = self.get_cached_result(cache_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {test_case['msid']}")
            return cached_result

        result = self._run_panel_source_test(test_case, ground_truth)
        self.cache_result(cache_path, result)
        return result

    def _assigner(
        self, test_case: Dict[str, Any], extract_dir
    ) -> PanelSourceAssignerOpenAI:
        if test_case["provider"] != "openai":
            raise NotImplementedError(f"Unsupported provider: {test_case['provider']}")
        prompts_source = test_case.get("prompts_source", self.config["prompts_source"])
        with open(prompts_source) as f:
            pipeline_config = yaml.safe_load(f).get("default", {}).get("pipeline", {})
        step = resolve_step_config(pipeline_config["assign_panel_source"])
        # Per-test-case nonce: runs can't read each other's prompt cache (as in
        # the caption benchmark), so token usage matches prod.
        nonce = f"Request {uuid.uuid4().hex}.\n\n"
        config = {
            "pipeline": {
                "assign_panel_source": {
                    "model": test_case["model"],
                    "prompts": {
                        **step["prompts"],
                        "system": nonce + step["prompts"]["system"],
                    },
                }
            }
        }
        return PanelSourceAssignerOpenAI(
            config, PromptHandler(config["pipeline"]), extract_dir=extract_dir
        )

    def _run_panel_source_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run panel source assignment test."""
        extract_dir = self.create_temp_extract_dir(test_case["msid"])
        try:
            start_time = time.time()
            write_stub_package(ground_truth, extract_dir)
            zip_structure = ZipStructure(
                manuscript_id=test_case["msid"],
                figures=figures_from_ground_truth(ground_truth),
            )
            figures = self._assigner(test_case, extract_dir).assign_panel_source(
                zip_structure
            )
            duration_ms = (time.time() - start_time) * 1000

            extracted = {f.figure_label: f for f in figures}
            rows, per_figure = [], {}
            totals: Dict[str, int] = {}
            for gt_fig in ground_truth["figures"]:
                label = gt_fig["figure_label"]
                listing = figure_listing(ground_truth, gt_fig)
                if not listing:
                    continue
                expected = {
                    p["panel_label"]: p.get("sd_files", []) for p in gt_fig["panels"]
                }
                ext = extracted.get(label)
                actual = {p.panel_label: p.sd_files for p in ext.panels} if ext else {}
                score = score_figure(expected, actual, listing)
                per_figure[label] = {
                    **score,
                    "actual": {k: sorted(v) for k, v in actual.items()},
                    "unassigned": sorted(ext.unassigned_sd_files) if ext else [],
                }
                for k, v in score["counts"].items():
                    totals[k] = totals.get(k, 0) + v
                for name in SCORES:
                    if score[name] is None:
                        continue
                    rows.append(
                        self.fill_results_bag(
                            test_case={**test_case, "duration_ms": duration_ms},
                            result={"input": f"{len(listing)} files"},
                            actual_output=str(score["counts"]),
                            expected_output="",
                            figure_label=label,
                            task=f"panel_source_{name}",
                            score=score[name],
                        )
                    )

            def ratio(n, d):
                return totals.get(n, 0) / totals[d] if totals.get(d) else None

            return {
                "msid": test_case["msid"],
                "input": {f.figure_label: f.sd_files for f in zip_structure.figures},
                "scores": {
                    "precision": ratio("correct", "assigned"),
                    "recall": ratio("correct", "expected"),
                    "shared_recall": ratio("shared_correct", "shared_expected"),
                    "off_panel": ratio("off_panel_kept", "off_panel_expected"),
                    "counts": totals,
                },
                "figures": per_figure,
                "results": rows,
                "cost": dataclasses.asdict(zip_structure.cost.assign_panel_source),
            }
        except Exception as e:
            logger.error(
                f"Error running panel source assignment test: {str(e)}", exc_info=True
            )
            raise
        finally:
            self.cleanup_temp_dir(extract_dir)
