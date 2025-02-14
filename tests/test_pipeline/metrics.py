"""Metrics for evaluating model performance on different tasks."""

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
    """Metric for evaluating data source extraction accuracy."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold=threshold)
        self.exact_match_score = 0.0
        self.jaccard_score = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            import json

            expected = json.loads(test_case.expected_output)
            actual = json.loads(test_case.actual_output)

            if not expected:
                self.score = 1.0 if not actual else 0.0
                self.exact_match_score = self.score
                self.jaccard_score = self.score
                self.success = self.score >= self.threshold
                return self.score

            # Count exact matches
            exact_matches = 0
            for exp_source in expected:
                for act_source in actual:
                    if (
                        exp_source.get("database", "").lower().strip()
                        == act_source.get("database", "").lower().strip()
                        and exp_source.get("accession_number", "").lower().strip()
                        == act_source.get("accession_number", "").lower().strip()
                    ):
                        exact_matches += 1
                        break

            # Calculate Jaccard similarities
            jaccard_scores = []
            for exp_source in expected:
                source_scores = []
                for act_source in actual:
                    # Compare database names
                    db_jaccard = calculate_jaccard_similarity(
                        set(exp_source.get("database", "").lower().split()),
                        set(act_source.get("database", "").lower().split()),
                    )
                    # Compare accession numbers
                    acc_jaccard = calculate_jaccard_similarity(
                        set(exp_source.get("accession_number", "").lower().split()),
                        set(act_source.get("accession_number", "").lower().split()),
                    )
                    source_scores.append((db_jaccard + acc_jaccard) / 2)
                jaccard_scores.append(max(source_scores) if source_scores else 0.0)

            # Calculate final scores
            self.exact_match_score = exact_matches / len(expected)
            self.jaccard_score = np.mean(jaccard_scores)
            self.score = (self.exact_match_score + self.jaccard_score) / 2
            self.success = self.score >= self.threshold

            return self.score

        except Exception as e:
            logger.error(f"Error measuring data source accuracy: {str(e)}")
            self.score = 0.0
            self.exact_match_score = 0.0
            self.jaccard_score = 0.0
            self.success = False
            return self.score

    @property
    def name(self) -> str:
        return "data_source_accuracy"


def get_metrics_for_task(task_name: str) -> List[BaseMetric]:
    """Get appropriate metrics for a given task."""
    base_metrics = [
        RougeMetric(score_type=rouge_type)
        for rouge_type in ["rouge1", "rouge2", "rougeL"]
    ] + [
        BleuMetric(bleu_type=bleu_type)
        for bleu_type in ["bleu1", "bleu2", "bleu3", "bleu4"]
    ]

    # Add task-specific metrics
    if task_name == "assign_panel_source":
        base_metrics.append(PanelSourceMatchMetric())
    elif task_name == "extract_data_availability":
        base_metrics.append(DataSourceAccuracyMetric())

    return base_metrics
