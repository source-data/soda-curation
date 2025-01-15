from datetime import datetime
from functools import lru_cache
from json import dump, load
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

MatchPanelCaptionOpenAI
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
    "strategy, msid, run",
    [(f["strategy"], f["msid"], f["run"]) for f in manuscript_fixtures()],
)
def test_extract_figures_from_figure_legends(strategy, msid, run, results_bag):
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
# Source Data assignation test
########################################################################################
class PanelSourceMatchMetric(BaseMetric):
    """Metric to evaluate panel source data assignment accuracy for entire manuscripts."""
    
    def __init__(self, score_type: str = "panel_accuracy", threshold: float = 0.8):
        self.score_type = score_type
        self.threshold = threshold
        self.success = False
        self.score = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure accuracy of panel source assignments across all figures in manuscript.
        Returns a score between 0 and 1 representing the proportion of correct assignments.
        """
        try:
            expected_data = eval(test_case.expected_output)
            actual_data = eval(test_case.actual_output)
            
            # Track totals across all figures
            total_expected_files = 0
            total_correct_assignments = 0
            
            # Process each figure
            for figure_label, figure_data in expected_data.items():
                # Get actual figure data
                actual_figure = actual_data.get(figure_label, {})
                
                # Process each panel in the figure
                expected_panel_files = {}
                for panel in figure_data.get("panels", []):
                    label = panel.get("panel_label", "")
                    files = set(panel.get("sd_files", []))
                    expected_panel_files[label] = files
                    total_expected_files += len(files)
                    
                    # Find matching actual panel
                    actual_panel = next(
                        (p for p in actual_figure.get("panels", [])
                         if p.get("panel_label") == label),
                        None
                    )
                    
                    if actual_panel:
                        actual_files = set(actual_panel.get("sd_files", []))
                        correct_for_panel = len(actual_files.intersection(files))
                        total_correct_assignments += correct_for_panel
                        logger.info(f"Figure {figure_label} Panel {label}: {correct_for_panel} correct of {len(files)}")

                # Handle unassigned files for this figure
                expected_unassigned = set(figure_data.get("unassigned_sd_files", []))
                actual_unassigned = set(actual_figure.get("unassigned_sd_files", []))
                total_expected_files += len(expected_unassigned)
                correct_unassigned = len(actual_unassigned.intersection(expected_unassigned))
                total_correct_assignments += correct_unassigned
                
                logger.info(f"Figure {figure_label} unassigned: {correct_unassigned} correct of {len(expected_unassigned)}")

            # Calculate final manuscript score
            if total_expected_files == 0:
                self.score = 1.0 if total_correct_assignments == 0 else 0.0
            else:
                self.score = total_correct_assignments / total_expected_files
                
            self.success = self.score >= self.threshold
            logger.info(f"Manuscript total: {total_correct_assignments} correct of {total_expected_files} ({self.score:.2f})")
            
            return self.score

        except Exception as e:
            logger.error(f"Error measuring manuscript panel source assignments: {str(e)}")
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

@pytest.mark.parametrize(
    "strategy, msid, run",
    [(f["strategy"], f["msid"], f["run"]) 
     for f in manuscript_fixtures()
     if "gpt" in f["strategy"]]  # Only test OpenAI strategies
)
def test_panel_source_assignment(strategy, msid, run, results_bag):
    """Test panel source data assignment for entire manuscript."""
    try:
        # Get ground truth figures
        ground_truth = _get_ground_truth(msid)
        extract_dir = manuscript_dir / msid
        
        # Process all figures in the manuscript
        test_figures = []
        for expected_figure in ground_truth["figures"]:
            # Create Panel objects for this figure
            input_panels = [
                Panel(
                    panel_label=p.get("panel_label", ""),
                    panel_caption=p.get("panel_caption", ""),
                    panel_bbox=p.get("panel_bbox", []),
                    confidence=p.get("confidence", 0.0),
                    sd_files=[],
                    ai_response=""
                ) for p in expected_figure.get("panels", [])
            ]

            # Get paths for this figure
            sd_files = expected_figure.get("sd_files", [])
            full_sd_files = [full_path(str(extract_dir), f) for f in sd_files]

            # Create Figure object
            test_figure = Figure(
                figure_label=expected_figure["figure_label"],
                img_files=expected_figure.get("img_files", []),
                sd_files=sd_files,
                _full_sd_files=full_sd_files,
                panels=input_panels,
                figure_caption=expected_figure.get("figure_caption", ""),
                caption_title=expected_figure.get("caption_title", "")
            )
            test_figures.append(test_figure)

        # Create ZipStructure with all figures
        test_zip_structure = ZipStructure(
            manuscript_id=msid,
            xml=ground_truth.get("xml", ""),
            docx=ground_truth.get("docx", ""),
            pdf=ground_truth.get("pdf", ""),
            _full_docx=full_path(str(extract_dir), ground_truth["docx"]) if ground_truth.get("docx") else None,
            _full_pdf=full_path(str(extract_dir), ground_truth["pdf"]) if ground_truth.get("pdf") else None,
            appendix=ground_truth.get("appendix", []),
            _full_appendix=[full_path(str(extract_dir), f) for f in ground_truth.get("appendix", [])],
            figures=test_figures,
            errors=[],
            non_associated_sd_files=[]
        )

        # Process entire manuscript
        config = _get_base_config()
        assigner = PanelSourceAssigner(config)
        actual_zip_structure = assigner.assign_panel_source(test_zip_structure)

        # Convert to comparable dictionaries
        actual_data = {}
        expected_data = {}
        
        # Build actual data
        for figure in actual_zip_structure.figures:
            actual_data[figure.figure_label] = {
                "panels": [{
                    "panel_label": p.panel_label,
                    "sd_files": p.sd_files
                } for p in figure.panels],
                "unassigned_sd_files": getattr(figure, "unassigned_sd_files", [])
            }
        
        # Build expected data
        for figure in ground_truth["figures"]:
            expected_data[figure["figure_label"]] = {
                "panels": [{
                    "panel_label": p["panel_label"],
                    "sd_files": p.get("sd_files", [])
                } for p in figure["panels"]],
                "unassigned_sd_files": figure.get("unassigned_sd_files", [])
            }

        test_case = LLMTestCase(
            input=str(test_zip_structure),
            actual_output=str(actual_data),
            expected_output=str(expected_data)
        )

        _fill_results_bag(
            results_bag,
            task="panel_source_assignment",
            strategy=strategy,
            msid=msid,
            run=run,
            metrics=_get_panel_source_metrics(),
            test_case=test_case,
            ai_response=str(actual_data)
        )

        assert_test(test_case, metrics=_get_panel_source_metrics())
        
    except Exception as e:
        logger.error(f"Error in panel source assignment test: {str(e)}")
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
