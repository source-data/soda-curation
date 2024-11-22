from datetime import datetime
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase
from functools import lru_cache
from json import load
from os import getenv
import pandas as pd
from pathlib import Path
import pytest
from tabulate import tabulate
from zipfile import ZipFile

from soda_curation.config import load_config
from soda_curation.pipeline.extract_captions.extract_captions_claude import (
    FigureCaptionExtractorClaude,
)
from soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorGpt,
)
from soda_curation.pipeline.extract_captions.extract_captions_regex import (
    FigureCaptionExtractorRegex,
)


########################################################################################
# Scoring task results
########################################################################################


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


def available_manuscripts():
    msids_with_ground_truth = set([f.stem for f in ground_truth_dir.glob("*.json")])
    msids_with_manuscript_archive = set([f.stem for f in manuscript_dir.glob("*.zip")])
    print("msids_with_ground_truth", msids_with_ground_truth)
    print("msids_with_manuscript_archive", msids_with_manuscript_archive)
    msids_with_both = msids_with_ground_truth.intersection(
        msids_with_manuscript_archive
    )
    return list(sorted(msids_with_both))


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


def strategies():
    all_strategies = (
        ["regex"]
        + [f"gpt-4o_temp={temp}" for temp in [0.0, 0.1, 0.5]]
        + [f"gpt-4o_top_p={top_p}" for top_p in [0.0, 0.1, 0.5]]
    )
    return _parse_env_list("STRATEGIES", all_strategies)


def manuscripts():
    all_manuscripts = available_manuscripts()
    return _parse_env_list("MANUSCRIPTS", all_manuscripts)


def runs():
    runs = getenv("RUNS", "")
    try:
        n_runs = int(runs)
    except ValueError:
        print("Invalid RUNS value, using default of 2")
        n_runs = 2
    return range(n_runs)


def figure_legends():
    return [
        {
            "strategy": strategy,
            "msid": msid,
            "run": run,
        }
        for strategy in strategies()
        for msid in manuscripts()
        for run in runs()
    ]


def figure_captions():
    return [
        {
            "strategy": strategy,
            "msid": msid,
            "run": run,
            "figure_label": figure_label,
        }
        for strategy in strategies()
        for msid in manuscripts()
        for run in runs()
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

    config_id = "openai" if is_openai else strategy
    config = _get_base_config().get(config_id, {})

    if is_openai:
        config_id = strategy.replace("gpt-4o_", "")
        if config_id.startswith("temp="):
            config["temperature"] = float(config_id.split("=")[1])
        elif config_id.startswith("top_p="):
            config["top_p"] = float(config_id.split("=")[1])

        extractor_cls = FigureCaptionExtractorGpt
    else:
        extractor_cls = {
            "anthropic": FigureCaptionExtractorClaude,
            "regex": FigureCaptionExtractorRegex,
        }.get(strategy)
    return extractor_cls(config)


@lru_cache
def _extract_manuscript_content(msid, strategy):
    docx_path = _get_manuscript_path(msid)
    extractor = _get_extractor(strategy)
    return extractor._extract_docx_content(docx_path)


def _extract_figure_legends_from_manuscript(
    strategy, manuscript_content, expected_figure_labels
):
    """
    Extract the figure legends section from a manuscript.

    """
    extractor = _get_extractor(strategy)
    expected_figure_count = len(expected_figure_labels)
    return extractor._locate_figure_captions(
        manuscript_content, expected_figure_count, expected_figure_labels
    )


@lru_cache
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
    figure_legends = _expected_figure_legends(msid)
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
):
    results_bag.task = task
    results_bag.input = test_case.input
    results_bag.actual = test_case.actual_output
    results_bag.expected = test_case.expected_output
    results_bag.strategy = strategy
    results_bag.msid = msid
    results_bag.run = run
    results_bag.figure_label = figure_label
    for metric in metrics:
        score = metric.measure(test_case)
        setattr(results_bag, metric.name, score)
        setattr(results_bag, f"{metric.name}_success", metric.is_successful())
        setattr(results_bag, f"{metric.name}_threshold", metric.threshold)


@pytest.mark.parametrize(
    "strategy, msid, run",
    [(f["strategy"], f["msid"], f["run"]) for f in figure_legends()],
)
def test_extract_figure_legends_from_manuscript(strategy, msid, run, results_bag):
    """
    Test the extraction of the figure legends section from a manuscript.

    The extracted figure legends section is scored against the reference figure legends section.

    The test results are added to the results_bag for further processing in the synthesis step.
    """
    ground_truth = _get_ground_truth(msid)
    expected_figure_labels = _expected_figure_labels(msid)
    manuscript_content = _extract_manuscript_content(msid, strategy)
    extracted_figure_legends = _extract_figure_legends_from_manuscript(
        strategy, manuscript_content, expected_figure_labels
    )
    expected_figure_legends = ground_truth["all_captions"]

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
    )
    assert_test(test_case, metrics=_get_metrics())


@pytest.mark.parametrize(
    "strategy, msid, run",
    [
        (f["strategy"], f["msid"], f["run"])
        for f in figure_legends()
    ],
)
def test_extract_figures_from_figure_legends(
    strategy, msid, run, results_bag
):
    """
    Test the extraction of individual figures from the figure legends section of a manuscript.

    The extracted figure labels must match the reference figure labels.

    The test results are added to the results_bag for further processing in the synthesis step.
    """
    figure_legends, extracted_figures = _extract_figures_from_figure_legends(
        strategy, msid, run
    )

    def _stringify_figure_labels(labels):
        return "\n".join(sorted(labels))

    actual_figure_labels = extracted_figures.keys()
    expected_figure_labels = _expected_figure_labels(msid)

    test_case = LLMTestCase(
        input=figure_legends,
        actual_output=_stringify_figure_labels(actual_figure_labels),
        expected_output=_stringify_figure_labels(expected_figure_labels),
    )

    _fill_results_bag(
        results_bag,
        task="extract_figures",
        strategy=strategy,
        msid=msid,
        run=run,
        metrics=_get_metrics(),
        test_case=test_case,
    )

    assert_test(test_case, metrics=_get_metrics())


@pytest.mark.parametrize(
    "strategy, msid, run, figure_label",
    [
        (f["strategy"], f["msid"], f["run"], f["figure_label"])
        for f in figure_captions()
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
    figure_legends, extracted_figures = _extract_figures_from_figure_legends(
        strategy, msid, run
    )

    assert figure_label in extracted_figures, f"Figure {figure_label} not found in extracted figures: {extracted_figures.keys()}"
    actual_figure_caption = extracted_figures[figure_label]["caption"]

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
    )

    assert_test(test_case, metrics=_get_metrics())


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
    results_dir = Path("data/eval")
    results_dir.mkdir(exist_ok=True, parents=True)

    # dump results to its own file
    ts = datetime.now()
    module_results_df = module_results_df.assign(timestamp=ts)
    module_results_df.to_json(
        results_dir / f"{ts.strftime('%Y-%m-%d_%H:%M:%S')}_results.json",
        orient="records",
    )

    is_prod_run = (
        module_results_df["msid"].nunique() > 2
        and module_results_df["run"].nunique() > 2
    )
    if is_prod_run:
        # dump results to global file
        global_results_file = results_dir / "results.json"
        if global_results_file.exists():
            global_results_df = pd.concat(
                [
                    pd.read_json(global_results_file),
                    module_results_df,
                ]
            )
        else:
            global_results_df = module_results_df
        global_results_df.to_json(global_results_file, orient="records")

    print("\n   summary:\n")
    summary_df = (
        module_results_df[["task", "strategy", "bleu1"]]
        .groupby(["task", "strategy"])
        .describe()
    )
    print(tabulate(summary_df, headers="keys", tablefmt="pretty"))
