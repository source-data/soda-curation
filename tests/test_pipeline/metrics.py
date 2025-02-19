"""Metrics for evaluating model performance on different tasks."""

import ast
import json
import logging
import re
from typing import List, Set

from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


def normalize_text(text: str, is_data_availability: bool = False) -> str:
    """Normalize text for consistent comparison."""
    if not isinstance(text, str):
        text = str(text)

    # Remove "Data Availability" prefix if specified
    if is_data_availability:
        text = re.sub(r"^data\s+availability\s*[:.]?\s*", "", text, flags=re.IGNORECASE)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove punctuation except periods and hyphens
    text = re.sub(r"[^\w\s.-]", "", text)

    # Normalize numbers
    text = re.sub(r"(\d+)", lambda m: str(float(m.group(1))), text)

    return text.strip()


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

    def measure(self, test_case: LLMTestCase) -> float:
        # Normalize both texts
        prediction = normalize_text(test_case.actual_output)
        target = normalize_text(test_case.expected_output)

        # Skip empty texts
        if not prediction or not target:
            return 0.0

        self.score = self.scorer.sentence_bleu_score(
            prediction=prediction,
            references=target,
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

    def __init__(self, threshold: float = 1.0):
        super().__init__(threshold=threshold)
        self.panel_scores = {}
        self.score = 0.0

    def calculate_panel_score(
        self, expected_files: List[str], actual_files: List[str]
    ) -> float:
        """Calculate score for a single panel."""
        # If no files are expected and none provided, that's correct
        if not expected_files and not actual_files:
            return 1.0

        # If no files are expected but some provided, that's wrong
        if not expected_files and actual_files:
            return 0.0

        # If files are expected but none provided, that's wrong
        if expected_files and not actual_files:
            return 0.0

        # Calculate exact matches
        expected_set = set(expected_files)
        actual_set = set(actual_files)
        correct_matches = len(expected_set.intersection(actual_set))

        # Score is based on exact matches over expected files
        return correct_matches / len(expected_files)

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            expected = json.loads(test_case.expected_output)
            actual = json.loads(test_case.actual_output)

            # Reset scores
            self.panel_scores = {}
            total_score = 0.0
            panel_count = 0

            # Calculate score for each panel
            for panel_id, data in expected.items():
                expected_files = data.get("files", [])
                actual_files = actual.get(panel_id, {}).get("files", [])

                panel_score = self.calculate_panel_score(expected_files, actual_files)
                self.panel_scores[panel_id] = panel_score
                total_score += panel_score
                panel_count += 1

            # Calculate average score
            self.score = total_score / panel_count if panel_count > 0 else 0.0
            self.success = self.score >= self.threshold

            return self.score

        except Exception as e:
            logger.error(f"Error measuring panel source matches: {str(e)}")
            self.score = 0.0
            self.panel_scores = {}
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
    # For most tasks, we'll use BLEU1
    if task_name in [
        "extract_sections",
        "extract_individual_captions",
        "locate_figure_captions",
    ]:
        return [BleuMetric(bleu_type="bleu1")]
    # For data availability, we keep the specific metric
    elif task_name == "extract_data_availability":
        return [DataSourceAccuracyMetric()]
    # For panel source assignment, we use the panel source metric
    elif task_name == "panel_source_assignment":
        return [PanelSourceMatchMetric()]
    return []
