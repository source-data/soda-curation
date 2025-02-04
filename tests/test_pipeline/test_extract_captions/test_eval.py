import logging
from datetime import datetime
from functools import lru_cache
from json import dump, dumps, load, loads
from os import getenv
from pathlib import Path
from typing import Dict, Set, Tuple
from zipfile import ZipFile

import nltk
import numpy as np
import papermill as pm
import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase
from tabulate import tabulate

from soda_curation.config import load_config
from soda_curation.data_availability.data_availability_openai import (
    DataAvailabilityExtractorGPT,
)
from soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorGpt,
)
from src.soda_curation.pipeline.assign_panel_source.assign_panel_source import (
    PanelSourceAssigner,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    full_path,
)

nltk.download("punkt_tab")

########################################################################################
# Scoring task results
########################################################################################

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


def _get_metrics():
    return [
        RougeMetric(score_type=rouge_type)
        for rouge_type in ["rouge1", "rouge2", "rougeL"]
    ] + [
        BleuMetric(bleu_type=bleu_type)
        for bleu_type in ["bleu1", "bleu2", "bleu3", "bleu4"]
    ]


########################################################################################
# Fixtures
########################################################################################

ground_truth_dir = Path("data/ground_truth")
manuscript_dir = Path("data/archives")


@lru_cache
def _get_ground_truth(msid):
    ground_truth_file = ground_truth_dir / f"{msid}.json"
    with open(ground_truth_file, "r") as f:
        return load(f)


def _expected_figure_labels(msid):
    return [f["figure_label"] for f in _get_ground_truth(msid)["figures"]]


def _expected_figure_legends(msid):
    return _get_ground_truth(msid)["all_captions"]


@lru_cache
def _get_manuscript_path(msid):
    ground_truth = _get_ground_truth(msid)
    archive_path = manuscript_dir / f"{msid}.zip"
    extracted_archive_path = manuscript_dir / msid
    if not extracted_archive_path.exists():
        extracted_archive_path.mkdir(exist_ok=True, parents=True)
        with ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extracted_archive_path)
    return extracted_archive_path / ground_truth["docx"]


def _parse_env_list(env_var, all_value, all_indicator="all", delimiter=","):
    val = getenv(env_var, "")
    if len(val) == 0:
        return []
    if val == all_indicator:
        return all_value
    try:
        val = int(val)
        return all_value[:val]
    except ValueError:
        pass
    if val.find(delimiter) == -1:
        return [val]
    return val.split(delimiter)


def _check_config(config: dict) -> dict:
    """Validate and configure OpenAI settings."""
    if "openai" not in config:
        raise ValueError("OpenAI configuration missing")

    required_fields = [
        "api_key",
        "caption_extraction_assistant_id",
        "panel_source_data_assistant_id",
        "caption_location_assistant_id",
    ]

    for field in required_fields:
        if field not in config["openai"]:
            raise ValueError(f"Missing required OpenAI config: {field}")

    # Set other defaults if not present
    config["openai"].setdefault("max_tokens", 96000)
    config["openai"].setdefault("top_p", 1.0)
    config["openai"].setdefault("temperature", 0.5)

    return config


# Modify _get_base_config
@lru_cache(maxsize=1)
def _get_base_config():
    config_path = "config.yaml"
    config = load_config(config_path)
    if "openai" not in config:
        raise ValueError("Missing openai configuration")
    return config


def _configure_openai_settings(
    config: dict, model: str, config_type: str, config_value: float
) -> dict:
    """Configure OpenAI settings in config."""
    config["openai"]["model"] = model
    if config_type == "temp":
        config["openai"]["temperature"] = config_value
    else:
        config["openai"]["top_p"] = config_value
    return config


VALID_MODELS = {
    "openai": {"gpt-4o": "gpt-4o-2024-08-06", "gpt-4o-mini": "gpt-4o-mini-2024-07-18"}
}


def _strategies():
    """Return list of OpenAI model strategies to test."""

    def model_strats(models, temps=["0.1", "0.5"], top_ps=["0", "1."]):
        strats = []
        for model in models:
            strats.extend([f"openai_{model}_temp={temp}" for temp in temps])
            strats.extend([f"openai_{model}_top_p={top_p}" for top_p in top_ps])
        return strats

    all_strategies = model_strats(["gpt-4o", "gpt-4o-mini"])
    return _parse_env_list("STRATEGIES", all_strategies)


def _validate_strategy(strategy: str) -> tuple:
    """Validate and parse OpenAI strategy string."""
    try:
        provider, model_alias, config = strategy.split("_")
        config_type, config_value = config.split("=")

        if provider != "openai":
            raise ValueError(
                f"Invalid provider: {provider}. Only 'openai' is supported."
            )

        if model_alias not in VALID_MODELS["openai"]:
            raise ValueError(f"Invalid model alias: {model_alias}")

        if config_type not in ["temp", "top_p"]:
            raise ValueError(f"Invalid config type: {config_type}")

        actual_model = VALID_MODELS["openai"][model_alias]
        return actual_model, config_type, float(config_value)

    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid strategy format: {strategy}. Error: {str(e)}")


def _manuscripts():
    msids_with_ground_truth = set([f.stem for f in ground_truth_dir.glob("*.json")])
    msids_with_manuscript_archive = set([f.stem for f in manuscript_dir.glob("*.zip")])
    print("msids_with_ground_truth", msids_with_ground_truth)
    print("msids_with_manuscript_archive", msids_with_manuscript_archive)
    msids_with_both = msids_with_ground_truth.intersection(
        msids_with_manuscript_archive
    )
    return _parse_env_list("MANUSCRIPTS", list(sorted(msids_with_both)))


def _runs():
    runs = getenv("RUNS", "")
    try:
        n_runs = int(runs)
    except ValueError:
        print("Invalid RUNS value, using default of 2")
        n_runs = 2
    return range(n_runs)


def manuscript_fixtures():
    return [
        {
            "strategy": strategy,
            "run": run,
            "msid": msid,
        }
        for strategy in _strategies()
        for run in _runs()
        for msid in _manuscripts()
    ]


def figure_fixtures():
    return [
        {
            "strategy": strategy,
            "run": run,
            "msid": msid,
            "figure_label": figure_label,
        }
        for strategy in _strategies()
        for run in _runs()
        for msid in _manuscripts()
        for figure_label in [
            f["figure_label"] for f in _get_ground_truth(msid)["figures"]
        ]
    ]


########################################################################################
# Tests
########################################################################################


def _get_extractor(strategy):
    """Get OpenAI extractor with validated configuration."""
    model, config_type, config_value = _validate_strategy(strategy)
    config = _get_base_config()
    config = _configure_openai_settings(config, model, config_type, config_value)
    return FigureCaptionExtractorGpt(config)


@lru_cache
def _extract_manuscript_content(msid, strategy):
    docx_path = _get_manuscript_path(msid)
    extractor = _get_extractor(strategy)
    return extractor._extract_docx_content(docx_path)


eval_base_dir = Path("data/eval")
cache_dir = eval_base_dir / "cache"


def _cache_file(task, strategy, msid, run):
    return cache_dir / f"{msid}_{strategy}_{run}_{task}.json"


def _get_cached_output(task, strategy, msid, run):
    cache_file = _cache_file(task, strategy, msid, run)
    try:
        with open(cache_file, "r") as f:
            return load(f)
    except FileNotFoundError:
        raise ValueError(f"Cache file not found: {cache_file}")


def _cache_output(task, strategy, msid, run, output):
    cache_file = _cache_file(task, strategy, msid, run)
    cache_file.parent.mkdir(exist_ok=True, parents=True)
    with open(cache_file, "w") as f:
        dump(output, f, indent=2)


def file_cache(task):
    def decorator(func):
        def wrapper(strategy, msid, run):
            try:
                cached_data = _get_cached_output(task, strategy, msid, run)
                print(f"Using cached {task} data for {strategy} {msid} {run}")
                return cached_data["input"], cached_data["output"]
            except ValueError:
                print(f"No cached {task} data found for {strategy} {msid} {run}")
                pass
            input_data, output_data = func(strategy, msid, run)
            _cache_output(
                task, strategy, msid, run, {"input": input_data, "output": output_data}
            )
            return input_data, output_data

        return wrapper

    return decorator


@file_cache("figure_legends")
def _extract_figure_legends_from_manuscript(strategy, msid, run):
    """Extract the figure legends section from a manuscript."""
    # Get manuscript content from docx
    manuscript_content = _extract_manuscript_content(msid, strategy)
    expected_figure_labels = _expected_figure_labels(msid)

    # Use raw manuscript content as input
    extractor = _get_extractor(strategy)
    expected_figure_count = len(expected_figure_labels)
    extracted_figures = extractor._locate_figure_captions(
        manuscript_content, expected_figure_count, expected_figure_labels
    )
    return manuscript_content, extracted_figures


@file_cache("figures")
def _extract_figures_from_figure_legends(strategy, msid, run):
    """
    Extract the figure captions from the figure legends section of a manuscript.

    Results are cached per strategy & manuscript as well as run to avoid re-extraction as much as possible, yet allow different runs.

    Returns a tuple of the input figure legends section and the figures extracted from it.
    The figure captions are returned as a dictionary like this:
    {
        "${figure_label}": {
            "figure_caption": "...",
            "caption_title": "...",
        },
        ...
    }
    """
    # Use ground truth figure legends as input instead of extracted ones
    figure_legends = _get_ground_truth(msid)["all_captions"]
    expected_figure_labels = _expected_figure_labels(msid)
    expected_figure_count = len(expected_figure_labels)

    extractor = _get_extractor(strategy)
    extracted_captions = extractor._extract_individual_captions(
        figure_legends,
        expected_figure_count,
        expected_figure_labels,
    )
    captions = extractor._parse_response(extracted_captions)

    return figure_legends, captions


def _fill_results_bag(
    results_bag,
    task: str,
    strategy: str = None,
    msid: str = None,
    run: int = None,
    figure_label: str = None,
    metrics: list = [],
    test_case: LLMTestCase = None,
    ai_response: str = None,
):
    """
    Fill the results bag with test metrics and metadata.

    Now includes additional scores for enhanced metrics (exact match and Jaccard similarity).
    """
    # Fill basic information
    results_bag.task = task
    results_bag.input = test_case.input
    results_bag.actual = test_case.actual_output
    results_bag.expected = test_case.expected_output
    results_bag.strategy = strategy
    results_bag.msid = msid
    results_bag.run = run
    results_bag.figure_label = figure_label
    results_bag.ai_response = ai_response

    # Process each metric
    for metric in metrics:
        # Calculate the main score
        score = metric.measure(test_case)

        # Store the main score and success status
        setattr(results_bag, metric.name, score)
        setattr(results_bag, f"{metric.name}_success", metric.is_successful())
        setattr(results_bag, f"{metric.name}_threshold", metric.threshold)

        # Store additional scores if available (for enhanced metrics)
        if hasattr(metric, "exact_match_score"):
            setattr(results_bag, f"{metric.name}_exact", metric.exact_match_score)

        if hasattr(metric, "jaccard_score"):
            setattr(results_bag, f"{metric.name}_jaccard", metric.jaccard_score)


@pytest.mark.parametrize(
    "strategy, msid, run",
    [(f["strategy"], f["msid"], f["run"]) for f in manuscript_fixtures()],
)
def test_extract_figure_legends_from_manuscript(strategy, msid, run, results_bag):
    """
    Test the extraction of the figure legends section from a manuscript.

    The extracted figure legends section is scored against the reference figure legends section.

    The test results are added to the results_bag for further processing in the synthesis step.
    """
    (
        manuscript_content,
        extracted_figure_legends,
    ) = _extract_figure_legends_from_manuscript(strategy, msid, run)
    expected_figure_legends = _get_ground_truth(msid)["all_captions"]

    test_case = LLMTestCase(
        input=manuscript_content,
        actual_output=extracted_figure_legends,
        expected_output=expected_figure_legends,
    )

    _fill_results_bag(
        results_bag,
        task="extract_figure_legends",
        strategy=strategy,
        msid=msid,
        run=run,
        metrics=_get_metrics(),
        test_case=test_case,
        ai_response=extracted_figure_legends,
    )
    assert_test(test_case, metrics=_get_metrics())


@pytest.mark.parametrize(
    "strategy, msid, run, figure_label",
    [
        (f["strategy"], f["msid"], f["run"], f["figure_label"])
        for f in figure_fixtures()
    ],
)
def test_extract_figures_from_figure_legends(
    strategy, msid, run, figure_label, results_bag
):
    """
    Test the extraction of individual figures from the figure legends section of a manuscript.

    The extracted figure labels must match the reference figure labels.
    The test results are added to the results_bag for further processing in the synthesis step.
    """
    # Use ground truth figure legends as input
    figure_legends = _get_ground_truth(msid)["all_captions"]
    _, extracted_figures = _extract_figures_from_figure_legends(strategy, msid, run)

    actual_figure_title = (
        extracted_figures[figure_label]["title"]
        if figure_label in extracted_figures
        else ""
    )

    # Compare against ground truth title
    ground_truth = _get_ground_truth(msid)
    expected_figure_title = next(
        f["caption_title"]
        for f in ground_truth["figures"]
        if f["figure_label"] == figure_label
    )

    test_case = LLMTestCase(
        input=figure_legends,
        actual_output=actual_figure_title,
        expected_output=expected_figure_title,
    )

    _fill_results_bag(
        results_bag,
        task="extract_figure_title",
        strategy=strategy,
        msid=msid,
        run=run,
        figure_label=figure_label,
        metrics=_get_metrics(),
        test_case=test_case,
        ai_response=actual_figure_title,
    )

    assert figure_label in extracted_figures
    assert_test(test_case, metrics=_get_metrics())


@pytest.mark.parametrize(
    "strategy, msid, run, figure_label",
    [
        (f["strategy"], f["msid"], f["run"], f["figure_label"])
        for f in figure_fixtures()
    ],
)
def test_extract_figure_titles_from_figure_legends(
    strategy, msid, run, figure_label, results_bag
):
    """
    Test the extraction of a single figure title from the figure legends section of a manuscript.

    The extracted figure title is scored against the reference figure title.
    The test passes if the score is above a threshold.

    The test results are added to the results_bag for further processing in the synthesis step.
    """
    """Test extraction of figure titles using ground truth legends."""
    # Use ground truth figure legends as input
    figure_legends = _get_ground_truth(msid)["all_captions"]
    _, extracted_figures = _extract_figures_from_figure_legends(strategy, msid, run)

    actual_figure_caption = (
        extracted_figures[figure_label]["caption"]
        if figure_label in extracted_figures
        else ""
    )

    # Compare against ground truth caption
    ground_truth = _get_ground_truth(msid)
    expected_figure_caption = next(
        f["figure_caption"]
        for f in ground_truth["figures"]
        if f["figure_label"] == figure_label
    )

    test_case = LLMTestCase(
        input=figure_legends,
        actual_output=actual_figure_caption,
        expected_output=expected_figure_caption,
    )

    _fill_results_bag(
        results_bag,
        task="extract_figure_caption",
        strategy=strategy,
        msid=msid,
        run=run,
        figure_label=figure_label,
        metrics=_get_metrics(),
        test_case=test_case,
        ai_response=actual_figure_caption,
    )

    assert figure_label in extracted_figures
    assert_test(test_case, metrics=_get_metrics())


@pytest.mark.parametrize(
    "strategy, msid, run, figure_label",
    [
        (f["strategy"], f["msid"], f["run"], f["figure_label"])
        for f in figure_fixtures()
    ],
)
def test_extract_figure_captions_from_figure_legends(
    strategy, msid, run, figure_label, results_bag
):
    """
    Test the extraction of a single figure caption from the figure legends section of a manuscript.

    The extracted figure caption is scored against the reference figure caption.
    The test passes if the score is above a threshold.

    The test results are added to the results_bag for further processing in the synthesis step.
    """
    # Use ground truth figure legends as input
    figure_legends = _get_ground_truth(msid)["all_captions"]
    _, extracted_figures = _extract_figures_from_figure_legends(strategy, msid, run)

    actual_figure_caption = (
        extracted_figures[figure_label]["caption"]
        if figure_label in extracted_figures
        else ""
    )

    # Compare against ground truth caption
    ground_truth = _get_ground_truth(msid)
    expected_figure_caption = next(
        f["figure_caption"]
        for f in ground_truth["figures"]
        if f["figure_label"] == figure_label
    )

    test_case = LLMTestCase(
        input=figure_legends,
        actual_output=actual_figure_caption,
        expected_output=expected_figure_caption,
    )

    _fill_results_bag(
        results_bag,
        task="extract_figure_caption",
        strategy=strategy,
        msid=msid,
        run=run,
        figure_label=figure_label,
        metrics=_get_metrics(),
        test_case=test_case,
        ai_response=actual_figure_caption,
    )

    assert figure_label in extracted_figures
    assert_test(test_case, metrics=_get_metrics())


########################################################################################
# Panel SourceData Assignement
########################################################################################


class PanelSourceMatchMetric(BaseMetric):
    """Metric to evaluate panel source data assignment accuracy at figure level."""

    def __init__(self, score_type: str = "panel_accuracy", threshold: float = 0.8):
        self.score_type = score_type
        self.threshold = threshold
        self.success = False
        self.score = 0.0
        self.exact_match_score = 0.0
        self.jaccard_score = 0.0

    def _calculate_panel_scores(
        self, expected_panel: Dict, actual_panel: Dict
    ) -> Tuple[float, float]:
        """Calculate both exact match and Jaccard similarity for a panel."""
        expected_files = set(expected_panel.get("sd_files", []))
        actual_files = set(actual_panel.get("sd_files", []))

        # Exact match score (1 if sets are identical, 0 otherwise)
        exact_match = 1.0 if expected_files == actual_files else 0.0

        # Jaccard similarity score
        jaccard = calculate_jaccard_similarity(expected_files, actual_files)

        return exact_match, jaccard

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure accuracy of panel source assignments using both exact matches and Jaccard similarity.
        Returns a combined score between 0-1.
        """
        try:
            expected_data = loads(test_case.expected_output)
            actual_data = loads(test_case.actual_output)

            total_figures = 0
            exact_match_sum = 0
            jaccard_sum = 0

            # Track detailed metrics
            figure_metrics = {}

            for expected_figure in expected_data:
                figure_label = expected_figure["figure_label"]
                total_figures += 1

                actual_figure = next(
                    (f for f in actual_data if f["figure_label"] == figure_label), None
                )

                if not actual_figure:
                    logger.warning(
                        f"No matching actual figure found for {figure_label}"
                    )
                    figure_metrics[figure_label] = {"exact_match": 0.0, "jaccard": 0.0}
                    continue

                # Process panels
                figure_exact_matches = []
                figure_jaccard_scores = []

                # Match panels
                expected_panels = expected_figure.get("panels", [])
                actual_panels = actual_figure.get("panels", [])

                for expected_panel in expected_panels:
                    expected_label = expected_panel["panel_label"]
                    actual_panel = next(
                        (
                            p
                            for p in actual_panels
                            if p["panel_label"] == expected_label
                        ),
                        None,
                    )

                    if actual_panel:
                        exact_match, jaccard = self._calculate_panel_scores(
                            expected_panel, actual_panel
                        )
                        figure_exact_matches.append(exact_match)
                        figure_jaccard_scores.append(jaccard)
                    else:
                        figure_exact_matches.append(0.0)
                        figure_jaccard_scores.append(0.0)

                # Handle unassigned files
                expected_unassigned = set(
                    expected_figure.get("unassigned_sd_files", [])
                )
                actual_unassigned = set(actual_figure.get("unassigned_sd_files", []))

                unassigned_exact = (
                    1.0 if expected_unassigned == actual_unassigned else 0.0
                )
                unassigned_jaccard = calculate_jaccard_similarity(
                    expected_unassigned, actual_unassigned
                )

                if expected_unassigned or actual_unassigned:
                    figure_exact_matches.append(unassigned_exact)
                    figure_jaccard_scores.append(unassigned_jaccard)

                # Calculate figure-level scores
                figure_exact_match = (
                    np.mean(figure_exact_matches) if figure_exact_matches else 0.0
                )
                figure_jaccard = (
                    np.mean(figure_jaccard_scores) if figure_jaccard_scores else 0.0
                )

                exact_match_sum += figure_exact_match
                jaccard_sum += figure_jaccard

                figure_metrics[figure_label] = {
                    "exact_match": figure_exact_match,
                    "jaccard": figure_jaccard,
                }

                logger.info(
                    f"Figure {figure_label}: Exact={figure_exact_match:.2f}, "
                    f"Jaccard={figure_jaccard:.2f}"
                )

            # Calculate overall scores
            self.exact_match_score = (
                exact_match_sum / total_figures if total_figures > 0 else 0.0
            )
            self.jaccard_score = (
                jaccard_sum / total_figures if total_figures > 0 else 0.0
            )

            # Combined score (equal weighting)
            self.score = (self.exact_match_score + self.jaccard_score) / 2
            self.success = self.score >= self.threshold

            # Log detailed metrics
            logger.info("\nDetailed Metrics:")
            for fig_label, metrics in figure_metrics.items():
                logger.info(
                    f"{fig_label}: Exact={metrics['exact_match']:.2f}, "
                    f"Jaccard={metrics['jaccard']:.2f}"
                )
            logger.info(
                f"\nOverall scores - Exact: {self.exact_match_score:.2f}, "
                f"Jaccard: {self.jaccard_score:.2f}, "
                f"Combined: {self.score:.2f}"
            )

            return self.score

        except Exception as e:
            logger.error(
                f"Error measuring panel source assignments: {str(e)}", exc_info=True
            )
            self.score = 0.0
            self.exact_match_score = 0.0
            self.jaccard_score = 0.0
            self.success = False
            return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def name(self):
        return f"panel_source_{self.score_type}"


def _get_panel_source_metrics():
    """Get metrics for evaluating panel source assignments."""
    return [
        PanelSourceMatchMetric(score_type="manuscript_accuracy", threshold=0.8),
        # You might want to add RougeMetric and BleuMetric here if you want those scores too
        RougeMetric(score_type="rougeL"),
        BleuMetric(bleu_type="bleu1"),
    ]


def _get_panel_source_assigner(strategy):
    """Get panel source assigner with validated configuration."""
    model, config_type, config_value = _validate_strategy(strategy)
    config = _get_base_config()
    config = _configure_openai_settings(config, model, config_type, config_value)
    return PanelSourceAssigner(config)


@file_cache("panel_source_assignment")
def _assign_panel_sources(strategy, msid, run):
    """Process panel source assignment with caching, returning only essential data."""
    # Get ground truth figures and setup
    ground_truth = _get_ground_truth(msid)
    extract_dir = manuscript_dir / msid

    # Create test figures with ground truth panels
    test_figures = []
    for expected_figure in ground_truth["figures"]:
        # Create panels with necessary information
        panels = []
        for p in expected_figure.get("panels", []):
            panel = Panel(
                panel_label=p["panel_label"],
                panel_caption=p["panel_caption"],
                panel_bbox=[0, 0, 1, 1],  # Dummy bbox
                confidence=1.0,
                sd_files=[],
                ai_response="",
            )
            panels.append(panel)

        # Get source data paths
        sd_files = expected_figure.get("sd_files", [])
        full_sd_files = [full_path(str(extract_dir), f) for f in sd_files]

        test_figure = Figure(
            figure_label=expected_figure["figure_label"],
            img_files=expected_figure.get("img_files", []),
            sd_files=sd_files,
            _full_sd_files=full_sd_files,
            panels=panels,
            unassigned_sd_files=[],
            figure_caption=expected_figure["figure_caption"],
            caption_title=expected_figure.get("caption_title", ""),
        )
        test_figures.append(test_figure)

    # Configure and run source assigner
    config = _get_base_config()
    config["extract_dir"] = str(extract_dir)
    assigner = PanelSourceAssigner(config)

    # Process each figure
    processed_figures = []
    for figure in test_figures:
        if (
            figure.figure_caption
            and figure.figure_caption != "Figure caption not found."
        ):
            try:
                processed_figure = assigner.assign_panel_source(figure)
                processed_figures.append(processed_figure)
            except Exception as e:
                logger.error(
                    f"Error assigning sources for figure {figure.figure_label}: {str(e)}"
                )
                processed_figures.append(figure)
        else:
            processed_figures.append(figure)

    # Format only the essential data we need for scoring
    actual_output = [
        {
            "figure_label": fig.figure_label,
            "panels": [
                {"panel_label": p.panel_label, "sd_files": p.sd_files}
                for p in fig.panels
            ],
            "unassigned_sd_files": fig.unassigned_sd_files,
        }
        for fig in processed_figures
    ]

    expected_output = [
        {
            "figure_label": fig["figure_label"],
            "panels": [
                {"panel_label": p["panel_label"], "sd_files": p.get("sd_files", [])}
                for p in fig.get("panels", [])
            ],
            "unassigned_sd_files": fig.get("unassigned_sd_files", []),
        }
        for fig in ground_truth["figures"]
    ]

    # Return only the minimal input context and the formatted outputs
    # We don't need to cache the entire ZipStructure
    minimal_input = {"manuscript_id": msid, "processed_figures": len(processed_figures)}

    return minimal_input, (actual_output, expected_output)


@pytest.mark.parametrize(
    "strategy, msid, run",
    [(f["strategy"], f["msid"], f["run"]) for f in manuscript_fixtures()],
)
def test_panel_source_assignment(strategy, msid, run, results_bag):
    """Test panel source data assignment accuracy using ground truth data."""
    try:
        minimal_input, (actual_output, expected_output) = _assign_panel_sources(
            strategy, msid, run
        )

        actual_json = dumps(actual_output)
        expected_json = dumps(expected_output)

        test_case = LLMTestCase(
            input=dumps(minimal_input),
            actual_output=actual_json,
            expected_output=expected_json,
        )

        _fill_results_bag(
            results_bag,
            task="panel_source_assignment",
            strategy=strategy,
            msid=msid,
            run=run,
            metrics=_get_panel_source_metrics(),
            test_case=test_case,
            ai_response=actual_json,
        )

        assert_test(test_case, metrics=_get_panel_source_metrics())

    except Exception as e:
        logger.error(f"Error in panel source assignment test: {str(e)}", exc_info=True)
        raise


########################################################################################
# Data availability test
########################################################################################
class DataSourceAccuracyMetric(BaseMetric):
    """Metric to evaluate accuracy of data source extraction."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.success = False
        self.score = 0.0
        self.exact_match_score = 0.0
        self.jaccard_score = 0.0

    def _source_to_tuple(self, source: Dict) -> Tuple[str, str]:
        """Convert source dict to tuple for comparison."""
        return (
            source.get("database", "").lower().strip(),
            source.get("accession_number", "").lower().strip(),
        )

    def _calculate_source_similarity(self, source1: Dict, source2: Dict) -> float:
        """Calculate similarity between two sources using combined field matching."""
        # Get normalized values
        db1 = source1.get("database", "").lower().strip()
        db2 = source2.get("database", "").lower().strip()
        acc1 = source1.get("accession_number", "").lower().strip()
        acc2 = source2.get("accession_number", "").lower().strip()

        # Calculate Jaccard similarity for each field
        db_jaccard = calculate_jaccard_similarity(set(db1.split()), set(db2.split()))
        acc_jaccard = calculate_jaccard_similarity(set(acc1.split()), set(acc2.split()))

        # Return average similarity
        return (db_jaccard + acc_jaccard) / 2

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure accuracy of data source extraction using both exact matches and Jaccard similarity.
        """
        try:
            expected_sources = loads(test_case.expected_output)
            actual_sources = loads(test_case.actual_output)

            if not expected_sources:
                self.score = 1.0 if not actual_sources else 0.0
                self.exact_match_score = self.score
                self.jaccard_score = self.score
                self.success = self.score >= self.threshold
                return self.score

            # Exact matching
            exact_matches = 0
            for expected in expected_sources:
                expected_tuple = self._source_to_tuple(expected)
                for actual in actual_sources:
                    if self._source_to_tuple(actual) == expected_tuple:
                        exact_matches += 1
                        break

            # Jaccard similarity matching
            source_similarities = []
            for expected in expected_sources:
                # Find best matching source
                max_similarity = 0.0
                for actual in actual_sources:
                    similarity = self._calculate_source_similarity(expected, actual)
                    max_similarity = max(max_similarity, similarity)
                source_similarities.append(max_similarity)

            # Calculate scores
            self.exact_match_score = exact_matches / len(expected_sources)
            self.jaccard_score = sum(source_similarities) / len(expected_sources)

            # Combined score (equal weighting)
            self.score = (self.exact_match_score + self.jaccard_score) / 2
            self.success = self.score >= self.threshold

            # Log detailed metrics
            logger.info("\nData Source Metrics:")
            logger.info(f"Exact Match Score: {self.exact_match_score:.2f}")
            logger.info(f"Jaccard Score: {self.jaccard_score:.2f}")
            logger.info(f"Combined Score: {self.score:.2f}")

            return self.score

        except Exception as e:
            logger.error(f"Error measuring data source accuracy: {str(e)}")
            self.score = 0.0
            self.exact_match_score = 0.0
            self.jaccard_score = 0.0
            self.success = False
            return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def name(self):
        return "data_source_accuracy"


def _get_data_availability_metrics():
    """Get metrics for evaluating data availability text extraction."""
    return [
        RougeMetric(score_type=rouge_type)
        for rouge_type in ["rouge1", "rouge2", "rougeL"]
    ] + [
        BleuMetric(bleu_type=bleu_type)
        for bleu_type in ["bleu1", "bleu2", "bleu3", "bleu4"]
    ]


def _get_data_sources_metrics():
    """Get metrics for evaluating data source extraction accuracy."""
    return [
        DataSourceAccuracyMetric(threshold=0.8),
        # You might want to add RougeMetric and BleuMetric here if you want those scores too
        RougeMetric(score_type="rougeL"),
        BleuMetric(bleu_type="bleu1"),
    ]


def _get_data_availability_extractor(strategy):
    """Get data availability extractor with validated configuration."""
    model, config_type, config_value = _validate_strategy(strategy)
    config = _get_base_config()
    config = _configure_openai_settings(config, model, config_type, config_value)

    return DataAvailabilityExtractorGPT(config)


@file_cache("data_availability")
def _extract_data_availability(strategy, msid, run):
    """Extract full data availability with caching."""
    manuscript_content = _extract_manuscript_content(msid, strategy)

    extractor = _get_data_availability_extractor(strategy)
    result = extractor.extract_data_availability(manuscript_content)
    section_text = result.get("section_text", "")
    data_sources = result.get("data_sources", [])

    return manuscript_content, (section_text, data_sources)


@file_cache("data_availability_section")
def _extract_data_availability_section(strategy, msid, run):
    """Extract just the data availability section from manuscript with caching."""
    manuscript_content = _extract_manuscript_content(msid, strategy)

    extractor = _get_data_availability_extractor(strategy)
    section_text = extractor._locate_data_availability_section(manuscript_content)

    return manuscript_content, section_text


@file_cache("data_sources")
def _extract_data_sources(strategy, msid, run):
    """Extract data sources from the data availability section with caching."""
    # Use ground truth section text as input to isolate source extraction capability
    ground_truth = _get_ground_truth(msid)
    section_text = ground_truth.get("data_availability", {}).get("section_text", "")

    extractor = _get_data_availability_extractor(strategy)
    extracted_sources = extractor._extract_data_records(section_text)

    return section_text, extracted_sources


@pytest.mark.parametrize(
    "strategy, msid, run",
    [
        (f["strategy"], f["msid"], f["run"]) for f in manuscript_fixtures()
    ],  # Remove the if "gpt" filter
)
def test_extract_data_availability_section(strategy, msid, run, results_bag):
    """Test extraction of the data availability section from manuscripts."""
    try:
        manuscript_content, extracted_section = _extract_data_availability_section(
            strategy, msid, run
        )

        # Get expected section from ground truth
        ground_truth = _get_ground_truth(msid)
        expected_section = ground_truth.get("data_availability", {}).get(
            "section_text", ""
        )

        test_case = LLMTestCase(
            input=manuscript_content,
            actual_output=extracted_section,
            expected_output=expected_section,
        )

        _fill_results_bag(
            results_bag,
            task="extract_data_availability_section",
            strategy=strategy,
            msid=msid,
            run=run,
            metrics=_get_data_availability_metrics(),
            test_case=test_case,
            ai_response=extracted_section,
        )

        assert_test(test_case, metrics=_get_data_availability_metrics())

    except Exception as e:
        logger.error(
            f"Error testing data availability section extraction: {str(e)}",
            exc_info=True,
        )
        raise


@pytest.mark.parametrize(
    "strategy, msid, run",
    [
        (f["strategy"], f["msid"], f["run"]) for f in manuscript_fixtures()
    ],  # Remove the if "gpt" filter
)
def test_extract_data_sources(strategy, msid, run, results_bag):
    """Test extraction of data sources from data availability section."""
    try:
        section_text, extracted_sources = _extract_data_sources(strategy, msid, run)

        # Get expected sources from ground truth
        ground_truth = _get_ground_truth(msid)
        expected_sources = ground_truth.get("data_availability", {}).get(
            "data_sources", []
        )

        # Convert to JSON strings for comparison
        extracted_json = dumps(extracted_sources)
        expected_json = dumps(expected_sources)

        test_case = LLMTestCase(
            input=section_text,
            actual_output=extracted_json,
            expected_output=expected_json,
        )

        _fill_results_bag(
            results_bag,
            task="extract_data_sources",
            strategy=strategy,
            msid=msid,
            run=run,
            metrics=_get_data_sources_metrics(),
            test_case=test_case,
            ai_response=extracted_json,
        )

        assert_test(test_case, metrics=_get_data_sources_metrics())

    except Exception as e:
        logger.error(f"Error testing data source extraction: {str(e)}", exc_info=True)
        raise


########################################################################################
# Report generation
########################################################################################


def test_synthesis(module_results_df):
    """
    This test mostly dumps the individual test results to a JSON file and prints a summary.

    Needs to be last in the test file to be executed last.
    """
    if module_results_df.empty:
        return

    # add a timestamp to the results
    ts = datetime.now()
    module_results_df = module_results_df.assign(timestamp=ts)

    # dump everything related to the test run to its own directory
    results_dir = eval_base_dir / ts.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir.mkdir(exist_ok=True, parents=True)

    # results to JSON
    results_file = results_dir / "results.json"
    module_results_df.to_json(results_file, orient="records")

    # move test run cache to directory
    cache_dir.rename(results_dir / "cache")

    # run eval notebook on the results
    eval_notebook = "results.ipynb"
    output_notebook = results_dir / eval_notebook
    pm.execute_notebook(
        eval_notebook,
        output_notebook,
        parameters=dict(results_file=str(results_file.resolve())),
    )

    print("\n   results written to:", results_file)
    print("\n   summary:\n")
    summary_df = (
        module_results_df[["task", "strategy", "bleu1"]]
        .groupby(["task", "strategy"])
        .describe()
    )
    print(tabulate(summary_df, headers="keys", tablefmt="pretty"))
