"""Metrics for evaluating model performance on different tasks."""

import ast
import logging
from typing import List, Set

import numpy as np
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


class RougeMetric(BaseMetric):
    def __init__(self, score_type: str = "rouge1", threshold: float = 0.95):
        self.score_type = score_type
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type=self.score_type,
        )
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def name(self):
        return self.score_type

    @property
    def __name__(self):
        return f"{self.score_type} metric"


class BleuMetric(BaseMetric):
    def __init__(self, bleu_type: str = "bleu1", threshold: float = 0.95):
        self.bleu_type = bleu_type
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.sentence_bleu_score(
            prediction=test_case.actual_output,
            references=test_case.expected_output,
            bleu_type=self.bleu_type,
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def name(self):
        return self.bleu_type

    @property
    def __name__(self):
        return f"{self.bleu_type} metric"


class PanelSourceMatchMetric(BaseMetric):
    """Metric for evaluating panel source data assignment accuracy."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold=threshold)
        self.exact_match_score = 0.0
        self.jaccard_score = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            import json

            expected = json.loads(test_case.expected_output)
            actual = json.loads(test_case.actual_output)

            total_panels = 0
            exact_matches = 0
            jaccard_scores = []

            for exp_figure in expected:
                figure_label = exp_figure["figure_label"]
                act_figure = next(
                    (f for f in actual if f["figure_label"] == figure_label), None
                )

                if not act_figure:
                    continue

                # Process panels
                for exp_panel in exp_figure.get("panels", []):
                    total_panels += 1
                    exp_files = set(exp_panel.get("sd_files", []))

                    # Find matching panel in actual
                    act_panel = next(
                        (
                            p
                            for p in act_figure.get("panels", [])
                            if p["panel_label"] == exp_panel["panel_label"]
                        ),
                        None,
                    )

                    if act_panel:
                        act_files = set(act_panel.get("sd_files", []))
                        if exp_files == act_files:
                            exact_matches += 1
                        jaccard_scores.append(
                            calculate_jaccard_similarity(exp_files, act_files)
                        )
                    else:
                        jaccard_scores.append(0.0)

            # Calculate scores
            self.exact_match_score = (
                exact_matches / total_panels if total_panels > 0 else 0.0
            )
            self.jaccard_score = np.mean(jaccard_scores) if jaccard_scores else 0.0

            # Combined score
            self.score = (self.exact_match_score + self.jaccard_score) / 2
            self.success = self.score >= self.threshold

            return self.score

        except Exception as e:
            logger.error(f"Error measuring panel source matches: {str(e)}")
            self.score = 0.0
            self.exact_match_score = 0.0
            self.jaccard_score = 0.0
            self.success = False
            return self.score

    @property
    def name(self) -> str:
        return "panel_source_accuracy"


class DataSourceAccuracyMetric(BaseMetric):
    """Metric for measuring data source extraction accuracy."""

    def __init__(self, threshold: float = 0.8):
        """Initialize metric with threshold."""
        super().__init__()  # Call parent's init without arguments
        self.threshold = threshold  # Set threshold after parent init
        self.last_measured_value = 0.0  # Initialize last measured value

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure accuracy of data source extraction."""
        try:
            # Parse the actual and expected outputs from strings to lists
            actual_sources = ast.literal_eval(test_case.actual_output)
            expected_sources = ast.literal_eval(test_case.expected_output)

            # Calculate individual scores
            database_matches = 0
            accession_matches = 0
            url_matches = 0

            for expected in expected_sources:
                for actual in actual_sources:
                    # Database partial match
                    if (
                        expected["database"].lower() in actual["database"].lower()
                        or actual["database"].lower() in expected["database"].lower()
                    ):
                        database_matches += 1

                    # Exact matches for accession and URL
                    if expected["accession_number"] == actual["accession_number"]:
                        accession_matches += 1
                    if expected["url"] == actual["url"]:
                        url_matches += 1

            total_sources = len(expected_sources) if expected_sources else 1

            # Calculate combined score
            combined_score = (
                database_matches / total_sources
                + accession_matches / total_sources
                + url_matches / total_sources
            ) / 3

            self.last_measured_value = combined_score  # Store the score
            return combined_score

        except Exception as e:
            logger.error(f"Error measuring data source accuracy: {str(e)}")
            self.last_measured_value = 0.0
            return 0.0

    def is_successful(self) -> bool:
        """Check if the last measured value meets the threshold."""
        return self.last_measured_value >= self.threshold

    @property
    def name(self) -> str:
        """Get the name of the metric."""
        return "data_source_accuracy"


def get_metrics_for_task(task_name: str) -> List[BaseMetric]:
    """Get metrics for a specific task."""
    base_metrics = []

    if task_name == "extract_sections":
        base_metrics.extend(
            [
                RougeMetric(
                    score_type="rouge1"
                ),  # Changed from rouge_type to score_type
                RougeMetric(score_type="rouge2"),
                RougeMetric(score_type="rougeL"),
                BleuMetric(bleu_type="bleu1"),  # Changed from n_gram to bleu_type
                BleuMetric(bleu_type="bleu2"),
                BleuMetric(bleu_type="bleu3"),
                BleuMetric(bleu_type="bleu4"),
            ]
        )
    elif task_name == "extract_individual_captions":
        base_metrics.extend(
            [
                RougeMetric(score_type="rouge1"),
                RougeMetric(score_type="rouge2"),
                RougeMetric(score_type="rougeL"),
                BleuMetric(bleu_type="bleu1"),
                BleuMetric(bleu_type="bleu2"),
                BleuMetric(bleu_type="bleu3"),
                BleuMetric(bleu_type="bleu4"),
            ]
        )
    elif task_name == "extract_data_availability":
        base_metrics.append(DataSourceAccuracyMetric())

    return base_metrics
