import difflib
import enum
import json
import pathlib
import re
import tqdm
import typing
import zipfile
from .config import load_config
from .pipeline.extract_captions.extract_captions_claude import (
    FigureCaptionExtractorClaude,
)
from .pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from .pipeline.extract_captions.extract_captions_regex import (
    FigureCaptionExtractorRegex,
)
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor


class Strategy(enum.Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    REGEX = "regex"


class Task(enum.Enum):
    FIGURE_LEGEND = "figure_legend"
    FIGURE_CAPTIONS = "figure_captions"


########################################################################################
# Running the individual tasks
########################################################################################


def _extract_figure_legend(context, strategy_id):
    zip_structure = context["zip_structure"]
    extracted_archive_path = context["extracted_archive_path"]

    docx_path = extracted_archive_path / zip_structure.docx
    expected_figure_labels = context["expected_figure_labels"]
    expected_figure_count = len(expected_figure_labels)

    config = context["config"].get(strategy_id, {})
    if strategy_id == "openai":
        extractor = FigureCaptionExtractorGpt(config)
    elif strategy_id == "claude":
        extractor = FigureCaptionExtractorClaude(config)
    elif strategy_id == "regex":
        extractor = FigureCaptionExtractorRegex(config)
    else:
        raise ValueError(f"Invalid strategy_id: {strategy_id}")
    file_content = extractor._extract_docx_content(docx_path)

    return extractor._locate_figure_captions(
        file_content, expected_figure_count, expected_figure_labels
    )


def _extract_figure_captions(context, strategy_id):
    zip_structure = context["zip_structure"]

    expected_figure_labels = context["expected_figure_labels"]
    expected_figure_count = len(expected_figure_labels)

    figure_legend = context["figure_legend"]
    config = context["config"].get(strategy_id, {})
    if strategy_id == "openai":
        extractor = FigureCaptionExtractorGpt(config)
    elif strategy_id == "claude":
        extractor = FigureCaptionExtractorClaude(config)
    elif strategy_id == "regex":
        extractor = FigureCaptionExtractorRegex(config)

    extracted_captions_response = extractor._extract_individual_captions(
        figure_legend, expected_figure_count, expected_figure_labels
    )

    # Parse the response into caption dictionary
    captions = extractor._parse_response(extracted_captions_response)
    # Process each figure
    figures = []
    for figure in zip_structure.figures:
        if figure.figure_label in captions:
            caption_info = captions[figure.figure_label]
            caption_text = caption_info["caption"]
            caption_title = caption_info["title"]
            figures.append(
                {
                    "label": figure.figure_label,
                    "title": caption_title,
                    "caption": caption_text,
                }
            )
        else:
            figures.append(
                {
                    "label": figure.figure_label,
                    "caption": "",
                    "title": "",
                }
            )

    return {
        f["label"]: {
            "caption": f["caption"],
            "title": f["title"],
        }
        for f in figures
    }


def run_task(task, strategy, context):
    task_fns = {
        Task.FIGURE_LEGEND: _extract_figure_legend,
        Task.FIGURE_CAPTIONS: _extract_figure_captions,
    }
    task_fn = task_fns.get(task, None)
    if task_fn is None:
        raise ValueError(
            f"Invalid task_id: {task.value}. Valid tasks are: {task_fns.keys()}"
        )

    expected_config_keys = {
        Task.FIGURE_LEGEND: [
            "config",
            "expected_figure_labels",
            "zip_structure",
            "extracted_archive_path",
        ],
        Task.FIGURE_CAPTIONS: [
            "config",
            "expected_figure_labels",
            "zip_structure",
            "figure_legend",
        ],
    }
    missing_keys = list(
        filter(lambda key: key not in context, expected_config_keys[task])
    )
    if missing_keys:
        raise ValueError(f"Missing required keys from context: {missing_keys}")

    return task_fn(context, strategy.value)


########################################################################################
# Evaluating the task results against the ground truth
########################################################################################


def _score_text_extraction(ground_truth, extracted_text):
    sm = difflib.SequenceMatcher(a=ground_truth, b=extracted_text)
    mbs = sm.get_matching_blocks()
    true_positives = sum([mb.size for mb in mbs])
    false_positives = len(extracted_text) - true_positives
    false_negatives = len(ground_truth) - true_positives

    accuracy = true_positives / len(ground_truth)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "diff": "\n".join(
            difflib.ndiff(ground_truth.splitlines(), extracted_text.splitlines()),
        ),
    }


def _get_captions(figures):
    return {
        f["figure_label"]: {
            "caption": f["figure_caption"],
            "title": f["caption_title"],
        }
        for f in figures
    }


def _sum_scores(scores, attr):
    return sum([score[attr] for score in scores.values()])


def _avg_scores(scores, attr):
    return _sum_scores(scores, attr) / len(scores)


def _score_figure_captions(ground_truth_figures, extracted_figures):
    if set(ground_truth_figures.keys()) != set(extracted_figures.keys()):
        raise ValueError(
            "Ground truth and extracted figures must have the same figure labels."
            f" Ground truth: {ground_truth_figures.keys()}, Extracted: {extracted_figures.keys()}"
        )
    scores = {}
    for label, ground_truth in ground_truth_figures.items():
        extracted = extracted_figures[label]
        scores[label] = _score_text_extraction(
            ground_truth["caption"], extracted["caption"]
        )
    return {
        "tp": _sum_scores(scores, "tp"),
        "fp": _sum_scores(scores, "fp"),
        "fn": _sum_scores(scores, "fn"),
        "accuracy": _avg_scores(scores, "accuracy"),
        "precision": _avg_scores(scores, "precision"),
        "recall": _avg_scores(scores, "recall"),
        "f1": _avg_scores(scores, "f1"),
        "diff": "\n\n".join([score["diff"] for score in scores.values()]),
        "scores": scores,
    }


def score_task_run(full_ground_truth, task, output):
    scorings = {
        Task.FIGURE_LEGEND: {
            "ground_truth_fn": lambda ground_truth: ground_truth["all_captions"],
            "score_fn": _score_text_extraction,
        },
        Task.FIGURE_CAPTIONS: {
            "ground_truth_fn": lambda ground_truth: _get_captions(
                ground_truth["figures"]
            ),
            "score_fn": _score_figure_captions,
        },
    }
    scoring = scorings.get(task, None)
    if scoring is None:
        raise ValueError(
            f"Invalid task: {task.value}. Valid tasks are: {scorings.keys()}"
        )

    ground_truth = scoring["ground_truth_fn"](full_ground_truth)
    return {
        "ground_truth": ground_truth,
        "score": scoring["score_fn"](ground_truth, output),
    }


########################################################################################
# Running and scoring the tasks
########################################################################################


def _extract_archive(archive_path, extracted_archive_path):
    extracted_archive_path.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extracted_archive_path)


def _get_zip_structure(archive_path, extracted_archive_path):
    xml_extractor = XMLStructureExtractor(archive_path, extracted_archive_path)
    return xml_extractor.extract_structure()


def _expected_figure_labels(zip_structure):
    return [
        fig.figure_label
        for fig in zip_structure.figures
        if not re.search(r"EV", fig.figure_label, re.IGNORECASE)
    ]


def _prepare_manuscripts(archives_dir, ground_truth_dir, working_dir):
    manuscripts = []
    for ground_truth_file in ground_truth_dir.glob("*.json"):
        msid = ground_truth_file.stem
        archive_file = archives_dir / f"{msid}.zip"
        extracted_archive_dir = working_dir / msid
        _extract_archive(archive_file, extracted_archive_dir)
        manuscripts.append(
            {
                "msid": msid,
                "archive_path": archive_file,
                "extracted_archive_path": extracted_archive_dir,
                "zip_structure": _get_zip_structure(
                    archive_file, extracted_archive_dir
                ),
                "ground_truth_path": ground_truth_file.resolve(),
                "ground_truth_data": json.loads(ground_truth_file.read_text()),
            }
        )
    return manuscripts


def _get_config():
    config_path = "config.yaml"
    return load_config(config_path)


def _run_and_score_task(task, strategy, config, manuscript):
    zip_structure = manuscript["zip_structure"]
    context = {
        "config": config,
        "expected_figure_labels": _expected_figure_labels(zip_structure),
        "zip_structure": zip_structure,
        "extracted_archive_path": manuscript["extracted_archive_path"],
        "figure_legend": manuscript["ground_truth_data"]["all_captions"],
    }
    output = run_task(task, strategy, context)
    ground_truth = manuscript["ground_truth_data"]
    score_result = score_task_run(ground_truth, task, output)
    return {
        "output": output,
        "ground_truth": score_result["ground_truth"],
        "score": score_result["score"],
    }


def run_and_score_tasks(
    archives_dir: pathlib.Path,
    ground_truth_dir: pathlib.Path,
    working_dir: pathlib.Path,
    n_repeats: int = 1,
    tasks: typing.List = [Task.FIGURE_LEGEND, Task.FIGURE_CAPTIONS],
    strategies: typing.List = [Strategy.CLAUDE, Strategy.OPENAI, Strategy.REGEX],
):
    manuscripts = _prepare_manuscripts(archives_dir, ground_truth_dir, working_dir)
    if not manuscripts:
        raise ValueError("No manuscripts with ground truth found.")

    tasks = [
        (task, strategy)
        for task in tasks
        for strategy in strategies
        if task in tasks and strategy in strategies
    ]
    runs = [
        (task, strategy, manuscript, run_id)
        for task, strategy in tasks
        for manuscript in manuscripts
        for run_id in range(n_repeats)
    ]
    config = _get_config()
    results = []
    for task, strategy, manuscript, run_id in tqdm.tqdm(runs, desc="Running tasks"):
        result = _run_and_score_task(task, strategy, config, manuscript)
        results.append(
            {
                "task": task.value,
                "strategy": strategy.value,
                "msid": manuscript["msid"],
                "run_id": run_id,
                "output": result["output"],
                "ground_truth": result["ground_truth"],
                "score": result["score"],
            }
        )
    return results
