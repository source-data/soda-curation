from datetime import datetime
from functools import lru_cache
from json import dump, load, loads, dumps
from os import getenv
from pathlib import Path
from zipfile import ZipFile
import os 
import papermill as pm
import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase
from tabulate import tabulate
from soda_curation.data_availability.data_availability_openai import DataAvailabilityExtractorGPT
from soda_curation.config import load_config
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure, Panel, ZipStructure, full_path 
)
from soda_curation.pipeline.extract_captions.extract_captions_claude import (
    FigureCaptionExtractorClaude,
)
from soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorGpt,
)
from soda_curation.pipeline.extract_captions.extract_captions_regex import (
    FigureCaptionExtractorRegex,
)
from src.soda_curation.pipeline.object_detection.object_detection import (
    convert_to_pil_image,
    create_object_detection,
)
from src.soda_curation.pipeline.match_caption_panel.match_caption_panel_openai import (
    MatchPanelCaptionOpenAI,
)
from src.soda_curation.pipeline.assign_panel_source.assign_panel_source import (
    PanelSourceAssigner,
)
from src.soda_curation.pipeline.assign_panel_source.assign_panel_source_prompts import (
    get_assign_panel_source_prompt
)

import logging
from typing import Dict, List, Optional, Tuple


########################################################################################
# Scoring task results
########################################################################################

logger = logging.getLogger(__name__)

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


import nltk

nltk.download("punkt_tab")


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


def _strategies():
    def model_strats(model, temps=["0", "0.1", "0.5"], top_ps=["0", "0.1", "0.5"]):
        return [
            f"{model}_temp={temp}" for temp in temps
        ] + [
            f"{model}_top_p={top_p}" for top_p in top_ps
        ]
    all_strategies = (
        ["regex"]
        + model_strats("gpt-4o")
        + model_strats("claude-3-5-sonnet")
    )
    return _parse_env_list("STRATEGIES", all_strategies)


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


@lru_cache(maxsize=1)
def _get_base_config():
    config_path = "config.yaml"
    return load_config(config_path)


def _get_extractor(strategy):
    is_openai = "gpt" in strategy
    is_anthropic = "claude" in strategy

    config_id = (
        "openai" if is_openai
        else "anthropic" if is_anthropic
        else strategy
    )
    config = _get_base_config().get(config_id, {})

    if is_openai or is_anthropic:
        config_id = strategy.split("_")[1]
        if config_id.startswith("temp="):
            config["temperature"] = float(config_id.split("=")[1])
        elif config_id.startswith("top_p="):
            config["top_p"] = float(config_id.split("=")[1])

    if is_openai:
        extractor_cls = FigureCaptionExtractorGpt
    elif is_anthropic:
        extractor_cls = FigureCaptionExtractorClaude
    else:
        extractor_cls = {
            "regex": FigureCaptionExtractorRegex,
        }.get(strategy)
    return extractor_cls(config)


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
            _cache_output(task, strategy, msid, run, {"input": input_data, "output": output_data})
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
    results_bag.task = task
    results_bag.input = test_case.input
    results_bag.actual = test_case.actual_output
    results_bag.expected = test_case.expected_output
    results_bag.strategy = strategy
    results_bag.msid = msid
    results_bag.run = run
    results_bag.figure_label = figure_label
    results_bag.ai_response = ai_response 
    
    for metric in metrics:
        score = metric.measure(test_case)
        setattr(results_bag, metric.name, score)
        setattr(results_bag, f"{metric.name}_success", metric.is_successful())
        setattr(results_bag, f"{metric.name}_threshold", metric.threshold)

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
    manuscript_content, extracted_figure_legends = _extract_figure_legends_from_manuscript(strategy, msid, run)
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
        ai_response=extracted_figure_legends
        
    )
    assert_test(test_case, metrics=_get_metrics())


@pytest.mark.parametrize(
    "strategy, msid, run, figure_label",
    [
        (f["strategy"], f["msid"], f["run"], f["figure_label"])
        for f in figure_fixtures()
    ],
)
def test_extract_figures_from_figure_legends(strategy, msid, run, figure_label, results_bag):
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
        ai_response=actual_figure_title
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
        ai_response=actual_figure_caption
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
        ai_response=actual_figure_caption
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

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure accuracy of panel source assignments.
        Returns score between 0-1 representing proportion of correct assignments.
        """
        try:
            # Load JSON strings into Python objects
            expected_data = loads(test_case.expected_output)  # List of figures
            actual_data = loads(test_case.actual_output)      # List of figures
            
            total_figures = 0
            total_figures_correct = 0
            
            # Track per-figure metrics
            figure_metrics = {}
            
            # Process each expected figure
            for expected_figure in expected_data:
                figure_label = expected_figure['figure_label']
                total_figures += 1
                
                # Find matching actual figure
                actual_figure = next(
                    (f for f in actual_data if f['figure_label'] == figure_label),
                    None
                )
                
                if not actual_figure:
                    logger.warning(f"No matching actual figure found for {figure_label}")
                    figure_metrics[figure_label] = {
                        'total_files': 0,
                        'correct_files': 0,
                        'score': 0.0
                    }
                    continue

                # Track files for this figure
                total_files = 0
                correct_files = 0

                # Check each panel's assignments
                expected_panels = expected_figure.get('panels', [])
                actual_panels = actual_figure.get('panels', [])

                for expected_panel in expected_panels:
                    expected_label = expected_panel['panel_label']
                    expected_files = set(expected_panel.get('sd_files', []))
                    total_files += len(expected_files)
                    
                    # Find matching actual panel
                    actual_panel = next(
                        (p for p in actual_panels if p['panel_label'] == expected_label),
                        None
                    )
                    
                    if actual_panel:
                        actual_files = set(actual_panel.get('sd_files', []))
                        # Only count exact matches
                        if expected_files == actual_files:
                            correct_files += len(expected_files)

                # Handle unassigned files
                expected_unassigned = set(expected_figure.get('unassigned_sd_files', []))
                actual_unassigned = set(actual_figure.get('unassigned_sd_files', []))
                total_files += len(expected_unassigned)
                
                # Only count if exact match for unassigned
                if expected_unassigned == actual_unassigned:
                    correct_files += len(expected_unassigned)

                # Calculate figure-level score
                figure_score = 1.0 if total_files == 0 else correct_files / total_files
                figure_metrics[figure_label] = {
                    'total_files': total_files,
                    'correct_files': correct_files,
                    'score': figure_score
                }
                
                # Track perfectly matched figures
                if figure_score == 1.0:
                    total_figures_correct += 1

                logger.info(f"Figure {figure_label}: {correct_files}/{total_files} " 
                          f"files correct (score: {figure_score:.2f})")

            # Calculate overall manuscript score based on perfectly matched figures
            self.score = total_figures_correct / total_figures if total_figures > 0 else 0.0
            self.success = self.score >= self.threshold
            
            # Log detailed metrics
            logger.info("\nDetailed Figure Metrics:")
            for fig_label, metrics in figure_metrics.items():
                logger.info(f"{fig_label}: {metrics['correct_files']}/{metrics['total_files']} "
                          f"files correct (score: {metrics['score']:.2f})")
            logger.info(f"\nTotal figures perfectly matched: {total_figures_correct}/{total_figures} "
                       f"(score: {self.score:.2f})")
            
            return self.score

        except Exception as e:
            logger.error(f"Error measuring panel source assignments: {str(e)}", exc_info=True)
            self.score = 0.0
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
    return [PanelSourceMatchMetric(score_type="manuscript_accuracy", threshold=0.8)]

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
                ai_response=""
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
            caption_title=expected_figure.get("caption_title", "")
        )
        test_figures.append(test_figure)

    # Configure and run source assigner
    config = _get_base_config()
    config['extract_dir'] = str(extract_dir)
    assigner = PanelSourceAssigner(config)

    # Process each figure
    processed_figures = []
    for figure in test_figures:
        if figure.figure_caption and figure.figure_caption != "Figure caption not found.":
            try:
                processed_figure = assigner.assign_panel_source(figure)
                processed_figures.append(processed_figure)
            except Exception as e:
                logger.error(f"Error assigning sources for figure {figure.figure_label}: {str(e)}")
                processed_figures.append(figure)
        else:
            processed_figures.append(figure)

    # Format only the essential data we need for scoring
    actual_output = [{
        'figure_label': fig.figure_label,
        'panels': [{
            'panel_label': p.panel_label,
            'sd_files': p.sd_files
        } for p in fig.panels],
        'unassigned_sd_files': fig.unassigned_sd_files
    } for fig in processed_figures]

    expected_output = [{
        'figure_label': fig['figure_label'],
        'panels': [{
            'panel_label': p['panel_label'],
            'sd_files': p.get('sd_files', [])
        } for p in fig.get('panels', [])],
        'unassigned_sd_files': fig.get('unassigned_sd_files', [])
    } for fig in ground_truth['figures']]

    # Return only the minimal input context and the formatted outputs
    # We don't need to cache the entire ZipStructure
    minimal_input = {
        'manuscript_id': msid,
        'processed_figures': len(processed_figures)
    }
    
    return minimal_input, (actual_output, expected_output)

@pytest.mark.parametrize(
    "strategy, msid, run",
    [(f["strategy"], f["msid"], f["run"]) 
     for f in manuscript_fixtures()
     if "gpt" in f["strategy"]]
)
def test_panel_source_assignment(strategy, msid, run, results_bag):
    """Test panel source data assignment accuracy using ground truth data."""
    try:
        minimal_input, (actual_output, expected_output) = _assign_panel_sources(
            strategy, msid, run
        )
        
        # Convert outputs to JSON strings for test case
        actual_json = dumps(actual_output)
        expected_json = dumps(expected_output)

        # Create test case with minimal input and JSON outputs
        test_case = LLMTestCase(
            input=dumps(minimal_input),  # Convert minimal input to JSON string
            actual_output=actual_json,
            expected_output=expected_json
        )

        # Record results
        _fill_results_bag(
            results_bag,
            task="panel_source_assignment",
            strategy=strategy,
            msid=msid,
            run=run,
            metrics=_get_panel_source_metrics(),
            test_case=test_case,
            ai_response=actual_json
        )

        # Run test assertions
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
        
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure accuracy of data source extraction.
        Score is number of correctly extracted sources divided by total expected sources.
        """
        try:
            expected_sources = loads(test_case.expected_output)
            actual_sources = loads(test_case.actual_output)
            
            if not expected_sources:
                self.score = 1.0 if not actual_sources else 0.0
                self.success = self.score >= self.threshold
                return self.score
            
            correct_sources = 0
            for expected in expected_sources:
                # Look for matching source in actual results
                for actual in actual_sources:
                    if (expected.get("database") == actual.get("database") and
                        expected.get("accession_number") == actual.get("accession_number")):
                        correct_sources += 1
                        break
            
            self.score = correct_sources / len(expected_sources)
            self.success = self.score >= self.threshold
            return self.score
            
        except Exception as e:
            logger.error(f"Error measuring data source accuracy: {str(e)}")
            self.score = 0.0
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
    return [DataSourceAccuracyMetric(threshold=0.8)]

def _get_data_availability_extractor(strategy):
    """
    Get GPT-based data availability extractor with appropriate configuration.
    
    Args:
        strategy (str): The strategy name (e.g., 'gpt-4o_temp=0')
        
    Returns:
        DataAvailabilityExtractorGPT: Configured GPT extractor instance
        
    Raises:
        ValueError: If strategy is not a GPT-based strategy
    """
    if "gpt" not in strategy:
        raise ValueError(f"Only GPT strategies supported for data availability extraction. Got: {strategy}")

    config = _get_base_config()
    
    # Configure temperature or top_p if specified in strategy
    config_id = strategy.split("_")[1]
    if config_id.startswith("temp="):
        config["openai"]["temperature"] = float(config_id.split("=")[1])
    elif config_id.startswith("top_p="):
        config["openai"]["top_p"] = float(config_id.split("=")[1])

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
    [(f["strategy"], f["msid"], f["run"]) 
     for f in manuscript_fixtures()
     if "gpt" in f["strategy"]]
)
def test_extract_data_availability_section(strategy, msid, run, results_bag):
    """Test extraction of the data availability section from manuscripts."""
    try:
        manuscript_content, extracted_section = _extract_data_availability_section(
            strategy, msid, run
        )

        # Get expected section from ground truth
        ground_truth = _get_ground_truth(msid)
        expected_section = ground_truth.get("data_availability", {}).get("section_text", "")

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
            ai_response=extracted_section
        )

        assert_test(test_case, metrics=_get_data_availability_metrics())
        
    except Exception as e:
        logger.error(f"Error testing data availability section extraction: {str(e)}", exc_info=True)
        raise

@pytest.mark.parametrize(
    "strategy, msid, run",
    [(f["strategy"], f["msid"], f["run"]) 
     for f in manuscript_fixtures()
     if "gpt" in f["strategy"]]
)
def test_extract_data_sources(strategy, msid, run, results_bag):
    """Test extraction of data sources from data availability section."""
    try:
        section_text, extracted_sources = _extract_data_sources(strategy, msid, run)

        # Get expected sources from ground truth
        ground_truth = _get_ground_truth(msid)
        expected_sources = ground_truth.get("data_availability", {}).get("data_sources", [])

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
            ai_response=extracted_json
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
        parameters=dict(
            results_file=str(results_file.resolve())
        ),
    )

    print("\n   results written to:", results_file)
    print("\n   summary:\n")
    summary_df = (
        module_results_df[["task", "strategy", "bleu1"]]
        .groupby(["task", "strategy"])
        .describe()
    )
    print(tabulate(summary_df, headers="keys", tablefmt="pretty"))
