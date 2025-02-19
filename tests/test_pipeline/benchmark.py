"""
Benchmark testing framework for SODA curation pipeline.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import nltk
import pandas as pd
import pytest
import yaml
from deepeval.test_case import LLMTestCase

from src.soda_curation.pipeline.assign_panel_source.assign_panel_source_openai import (
    PanelSourceAssignerOpenAI,
)
from src.soda_curation.pipeline.data_availability.data_availability_openai import (
    DataAvailabilityExtractorOpenAI,
)
from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
)
from src.soda_curation.pipeline.extract_sections.extract_sections_openai import (
    SectionExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    ZipStructure,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

from .metrics import get_metrics_for_task


# At the very top of the file, after imports
def setup_logging():
    """Configure detailed logging."""
    # Clear any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Set logging level for specific loggers
    logging.getLogger("tests.test_pipeline").setLevel(logging.DEBUG)
    logging.getLogger("src.soda_curation").setLevel(logging.DEBUG)


setup_logging()

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Define a results bag to store test results
results_bag = []


def preprocess_text(text: str) -> str:
    """
    Preprocess text for comparison by:
    1. Removing escape characters
    2. Normalizing whitespace
    3. Stripping extra spaces

    Args:
        text (str): Input text to process

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return text

    # Replace common escape sequences with their actual characters
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    text = text.replace("\\r", " ")

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


class BenchmarkCache:
    """Handle caching of benchmark results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(
        self,
        test_name: str,
        msid: str,
        provider: str,
        model: str,
        temperature: float,
        top_p: float,
        run: int,
    ) -> Path:
        """Get path for cached result."""
        return (
            self.cache_dir
            / f"{test_name}_{msid}_{provider}_{model}_{temperature}_{top_p}_{run}.json"
        )

    def get_cached_result(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached result if it exists."""
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def cache_result(self, cache_path: Path, result: Dict[str, Any]) -> None:
        """Cache test result."""
        with open(cache_path, "w") as f:
            json.dump(result, f)


class BenchmarkRunner:
    """Runner for benchmark tests."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize benchmark runner with configuration."""
        self.config = config

        # Setup base directories
        self.base_output_dir = Path(
            self.config.get("output_dir", "/app/data/benchmark")
        )

        # Setup cache directory
        self.cache_dir = self.base_output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup timestamp-based results directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_dir = self.base_output_dir / timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache
        self.cache = BenchmarkCache(self.cache_dir)

        # Initialize results DataFrame
        self.results_df = pd.DataFrame(
            columns=[
                "pytest_obj",
                "status",
                "duration_ms",
                "strategy",
                "msid",
                "run",
                "task",
                "input",
                "actual",
                "expected",
                "figure_label",
                "ai_response",
                "rouge1",
                "rouge1_success",
                "rouge1_threshold",
                "rouge2",
                "rouge2_success",
                "rouge2_threshold",
                "rougeL",
                "rougeL_success",
                "rougeL_threshold",
                "bleu1",
                "bleu1_success",
                "bleu1_threshold",
                "bleu2",
                "bleu2_success",
                "bleu2_threshold",
                "bleu3",
                "bleu3_success",
                "bleu3_threshold",
                "bleu4",
                "bleu4_success",
                "bleu4_threshold",
                "data_source_accuracy",
                "data_source_accuracy_success",
                "data_source_accuracy_threshold",
                "data_source_accuracy_exact",
                "data_source_accuracy_jaccard",
                "timestamp",
            ]
        )

    def get_test_cases(self) -> Generator[Dict[str, Any], None, None]:
        """Generate test cases from configuration."""
        for provider, prov_config in self.config["providers"].items():
            for model in prov_config["models"]:
                for temp in model["temperatures"]:
                    for top_p in model["top_p"]:
                        for test_name in self.config[
                            "enabled_tests"
                        ]:  # Changed from "tests" to "enabled_tests"
                            # Handle different manuscript configurations
                            manuscripts = self.config["test_runs"]["manuscripts"]
                            if (
                                isinstance(manuscripts, str)
                                and manuscripts.lower() == "all"
                            ):
                                manuscripts = self._get_all_manuscripts()
                            elif isinstance(manuscripts, int):
                                manuscripts = self._get_n_manuscripts(manuscripts)
                            elif isinstance(manuscripts, list):
                                manuscripts = manuscripts
                            else:
                                raise ValueError(
                                    f"Invalid manuscripts configuration: {manuscripts}"
                                )

                            for msid in manuscripts:
                                for run in range(
                                    self.config["test_runs"]["n_runs"]
                                ):  # Changed from "runs" to "test_runs.n_runs"
                                    yield {
                                        "test_name": test_name,
                                        "msid": msid,
                                        "provider": provider,
                                        "model": model["name"],
                                        "temperature": temp,
                                        "top_p": top_p,
                                        "run": run,
                                    }

    def _get_all_manuscripts(self) -> List[str]:
        """Get all available manuscript IDs from ground truth directory."""
        ground_truth_dir = Path(self.config["ground_truth_dir"])
        return [f.stem for f in ground_truth_dir.glob("*.json")]

    def _get_n_manuscripts(self, n: int) -> List[str]:
        """Get first n manuscript IDs from ground truth directory."""
        all_manuscripts = self._get_all_manuscripts()
        return all_manuscripts[:n]

    def _load_ground_truth(self, msid: str) -> Dict[str, Any]:
        """Load ground truth data for manuscript."""
        ground_truth_path = Path(self.config["ground_truth_dir"]) / f"{msid}.json"
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")
        with open(ground_truth_path) as f:
            return json.load(f)

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run single test case."""
        # Get cache path
        cache_path = self.cache.get_cache_path(
            test_case["test_name"],
            test_case["msid"],
            test_case["provider"],
            test_case["model"],
            test_case["temperature"],
            test_case["top_p"],
            test_case["run"],
        )

        # Check cache
        cached_result = self.cache.get_cached_result(cache_path)
        if cached_result is not None:
            # Even with cached results, we need to calculate metrics
            if test_case["test_name"] == "extract_sections":
                self._fill_results_bag(
                    test_case=test_case,
                    result=cached_result,
                    actual_output=cached_result["output"],
                    expected_output=cached_result["expected"],
                )
            elif test_case["test_name"] == "extract_individual_captions":
                # For each figure in the cached results
                for result in cached_result.get("results", []):
                    self._fill_results_bag(
                        test_case=test_case,
                        result=cached_result,
                        actual_output=result["actual_caption"],
                        expected_output=result["expected_caption"],
                        actual_title=result["actual_title"],
                        expected_title=result["expected_title"],
                        figure_label=result["figure_label"],
                    )
            return cached_result

        # Load ground truth
        ground_truth = self._load_ground_truth(test_case["msid"])

        # Run appropriate test
        result = self._run_specific_test(test_case, ground_truth)

        # Cache result
        self.cache.cache_result(cache_path, result)

        return result

    def _run_specific_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run specific test based on test name."""
        test_name = test_case["test_name"]

        if test_name == "extract_sections":
            return self._run_extract_sections(test_case, ground_truth)
        elif test_name == "extract_individual_captions":
            return self._run_extract_captions(test_case, ground_truth)
        elif test_name == "assign_panel_source":
            return self._run_assign_panel_source(test_case, ground_truth)
        elif test_name == "extract_data_availability":
            return self._run_extract_data_availability(test_case, ground_truth)
        elif test_name == "assign_panel_source":
            return self._run_assign_panel_source(test_case, ground_truth)
        else:
            raise ValueError(f"Unknown test: {test_name}")

    def _run_extract_sections(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run extract sections test."""
        try:
            start_time = time.time()
            # Get manuscript path
            msid = test_case["msid"]

            # Create temporary extraction directory
            temp_extract_dir = self.base_output_dir / "temp_extracts" / msid
            temp_extract_dir.mkdir(parents=True, exist_ok=True)

            # Get ZIP path
            zip_path = Path(self.config["manuscript_dir"]) / f"{msid}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")

            # Use XML extractor to get structure and extract files
            extractor = XMLStructureExtractor(str(zip_path), str(temp_extract_dir))
            zip_structure = extractor.extract_structure()

            # Get the DOCX content
            docx_content = extractor.extract_docx_content(zip_structure.docx)

            # Load prompts from development configuration
            with open(self.config["prompts_source"]) as f:
                dev_config = yaml.safe_load(f)

            # Get the default configuration which contains our pipeline setup
            pipeline_config = dev_config.get("default", {}).get("pipeline", {})
            if not pipeline_config:
                raise ValueError("No pipeline configuration found in default profile")

            # Provider-specific configuration
            provider = test_case["provider"]

            # Create extractor configuration
            extractor_config = {
                "api_key": os.environ.get(f"{provider.upper()}_API_KEY"),
                "pipeline": {
                    "extract_sections": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config["extract_sections"][provider][
                                    "prompts"
                                ]["system"],
                                "user": pipeline_config["extract_sections"][provider][
                                    "prompts"
                                ]["user"],
                            },
                        }
                    }
                },
            }

            # Create prompt handler with the pipeline configuration
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create sections extractor instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                sections_extractor = SectionExtractorOpenAI(
                    extractor_config, prompt_handler
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Extract sections
            (
                figure_legends,
                data_availability,
                updated_structure,
            ) = sections_extractor.extract_sections(
                doc_content=docx_content, zip_structure=zip_structure
            )
            duration_ms = (time.time() - start_time) * 1000

            # Get ground truth sections
            ground_truth_figures = "\n".join(
                figure["figure_caption"] for figure in ground_truth["figures"]
            )
            ground_truth_data_availability = ground_truth.get(
                "data_availability", {}
            ).get("section_text", "")

            # Create result dictionary
            result = {
                "input": str(zip_path),
                "output": figure_legends,
                "expected": ground_truth_figures,
                "data_availability_output": data_availability,
                "data_availability_expected": ground_truth_data_availability,
                "cost": updated_structure.cost.extract_sections.model_dump()
                if hasattr(updated_structure.cost.extract_sections, "model_dump")
                else {},
            }

            # Fill results bag
            self._fill_results_bag(
                test_case={**test_case, "duration_ms": duration_ms},
                result=result,
                actual_output=figure_legends,
                expected_output=ground_truth_figures,
            )
            return result

        except Exception as e:
            logger.error(
                f"Error running extract sections test: {str(e)}", exc_info=True
            )
            raise

        finally:
            # Cleanup temporary directory after processing
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

    def _run_extract_captions(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run extract captions test."""
        try:
            start_time = time.time()
            # Load prompts from development configuration
            with open(self.config["prompts_source"]) as f:
                dev_config = yaml.safe_load(f)

            # Get the default configuration which contains our pipeline setup
            pipeline_config = dev_config.get("default", {}).get("pipeline", {})
            if not pipeline_config:
                raise ValueError("No pipeline configuration found in default profile")

            # Provider-specific configuration
            provider = test_case["provider"]

            # Create extractor configuration
            extractor_config = {
                "api_key": os.environ.get(f"{provider.upper()}_API_KEY"),
                "pipeline": {
                    "extract_individual_captions": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config[
                                    "extract_individual_captions"
                                ][provider]["prompts"]["system"],
                                "user": pipeline_config["extract_individual_captions"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    }
                },
            }

            # Create prompt handler
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create captions extractor instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                captions_extractor = FigureCaptionExtractorOpenAI(
                    extractor_config, prompt_handler
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Create a ZipStructure object with just the figures
            zip_structure = ZipStructure(figures=ground_truth["figures"])

            # Extract captions using all_captions from ground truth
            extracted_zip_structure = captions_extractor.extract_individual_captions(
                doc_content=ground_truth["all_captions"], zip_structure=zip_structure
            )

            duration_ms = (time.time() - start_time) * 1000

            # Compare each caption and title to its ground truth
            results = []
            for figure in ground_truth["figures"]:
                figure_label = figure["figure_label"]
                expected_caption = figure["figure_caption"]
                expected_title = figure["caption_title"]

                # Get the actual caption and title from extracted results
                extracted_figure = next(
                    (
                        f
                        for f in extracted_zip_structure.figures
                        if f["figure_label"] == figure_label
                    ),
                    None,
                )

                if extracted_figure:
                    actual_caption = extracted_figure["figure_caption"]
                    actual_title = extracted_figure["caption_title"]

                    # Calculate panel sequence score
                    (
                        panel_sequence_score,
                        extracted_labels,
                        ground_truth_labels,
                    ) = self._get_panel_sequence_score(
                        extracted_figure.get("panels", []), figure.get("panels", [])
                    )
                    # Fill results bag with all metrics including panel sequence
                    self._fill_results_bag(
                        test_case={**test_case, "duration_ms": duration_ms},
                        result={
                            "input": ground_truth["all_captions"],
                            "ai_response": str(extracted_zip_structure),
                        },
                        actual_output=actual_caption,
                        expected_output=expected_caption,
                        actual_title=actual_title,
                        expected_title=expected_title,
                        figure_label=figure_label,
                        panel_sequence_score=panel_sequence_score,
                        actual_panels=extracted_labels,
                        expected_panels=ground_truth_labels,
                    )

                    results.append(
                        {
                            "figure_label": figure_label,
                            "actual_caption": actual_caption,
                            "expected_caption": expected_caption,
                            "actual_title": actual_title,
                            "expected_title": expected_title,
                            "panel_sequence_score": panel_sequence_score,
                            "actual_panels": extracted_labels,
                            "expected_panels": ground_truth_labels,
                        }
                    )
            return {
                "input": ground_truth["all_captions"],
                "results": results,
                "cost": extracted_zip_structure.cost.extract_individual_captions.model_dump()
                if hasattr(
                    extracted_zip_structure.cost.extract_individual_captions,
                    "model_dump",
                )
                else {},
            }

        except Exception as e:
            logger.error(
                f"Error running extract captions test: {str(e)}", exc_info=True
            )
            raise

    def _run_extract_data_availability(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run data availability extraction test."""
        try:
            start_time = time.time()

            # Get manuscript path
            msid = test_case["msid"]

            # Create temporary extraction directory
            temp_extract_dir = self.base_output_dir / "temp_extracts" / msid
            temp_extract_dir.mkdir(parents=True, exist_ok=True)

            # Get ZIP path
            zip_path = Path(self.config["manuscript_dir"]) / f"{msid}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")

            # Use XML extractor to get structure and extract files
            extractor = XMLStructureExtractor(str(zip_path), str(temp_extract_dir))
            zip_structure = extractor.extract_structure()

            # Get the DOCX content
            docx_content = extractor.extract_docx_content(zip_structure.docx)

            # Load prompts from development configuration
            with open(self.config["prompts_source"]) as f:
                dev_config = yaml.safe_load(f)

            # Get the default configuration which contains our pipeline setup
            pipeline_config = dev_config.get("default", {}).get("pipeline", {})
            if not pipeline_config:
                raise ValueError("No pipeline configuration found in default profile")

            # Provider-specific configuration
            provider = test_case["provider"]

            # Create extractor configuration
            extractor_config = {
                "api_key": os.environ.get(f"{provider.upper()}_API_KEY"),
                "pipeline": {
                    "extract_data_sources": {  # Note: changed from extract_sections
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config["extract_data_sources"][
                                    provider
                                ]["prompts"]["system"],
                                "user": pipeline_config["extract_data_sources"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    }
                },
            }

            # Create prompt handler with the pipeline configuration
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create data availability extractor instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                data_extractor = DataAvailabilityExtractorOpenAI(
                    extractor_config, prompt_handler
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Get the ground truth text and sources
            ground_truth_text = ground_truth["data_availability"]["section_text"]
            ground_truth_sources = ground_truth["data_availability"]["data_sources"]

            # Extract data availability using the correct method
            updated_structure = data_extractor.extract_data_sources(
                section_text=ground_truth_text, zip_structure=zip_structure
            )

            # Get the extracted data
            data_availability_text = updated_structure.data_availability["section_text"]
            data_sources = updated_structure.data_availability["data_sources"]

            duration_ms = (time.time() - start_time) * 1000

            # Calculate scores
            scores = self._calculate_data_availability_scores(
                data_sources, ground_truth_sources
            )

            # Prepare specific outputs for each score type
            specific_outputs = {
                "database": {
                    "actual": str([source["database"] for source in data_sources]),
                    "expected": str(
                        [source["database"] for source in ground_truth_sources]
                    ),
                },
                "accession": {
                    "actual": str(
                        [source["accession_number"] for source in data_sources]
                    ),
                    "expected": str(
                        [source["accession_number"] for source in ground_truth_sources]
                    ),
                },
                "url": {
                    "actual": str([source["url"] for source in data_sources]),
                    "expected": str([source["url"] for source in ground_truth_sources]),
                },
                "combined": {
                    "actual": str(data_sources),
                    "expected": str(ground_truth_sources),
                },
            }

            # Fill results bag with specific outputs for each score type
            self._fill_results_bag(
                test_case={**test_case, "duration_ms": duration_ms},
                result={"input": docx_content, "ai_response": str(updated_structure)},
                actual_output=specific_outputs,  # Now passing dictionary of specific outputs
                expected_output=specific_outputs,  # Now passing dictionary of specific outputs
                data_availability_scores=scores,
            )

            return {
                "input": docx_content,
                "output": data_availability_text,
                "expected": ground_truth_text,
                "extracted_sources": data_sources,
                "ground_truth_sources": ground_truth_sources,
                "scores": scores,
                "cost": updated_structure.cost.extract_data_sources.model_dump()  # Changed from extract_data_availability
                if hasattr(updated_structure.cost.extract_data_sources, "model_dump")
                else {},
            }

        except Exception as e:
            logger.error(
                f"Error running data availability test: {str(e)}", exc_info=True
            )
            raise

        finally:
            # Cleanup temporary directory after processing
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

    def _get_panel_sequence_score(
        self, extracted_panels: List[Dict], ground_truth_panels: List[Dict]
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate score for panel sequence matching and return the sequences.

        Args:
            extracted_panels: List of panel dictionaries from extraction
            ground_truth_panels: List of panel dictionaries from ground truth

        Returns:
            tuple: (score, extracted_labels, ground_truth_labels)
                - score: 1.0 if sequences match exactly, 0.0 otherwise
                - extracted_labels: List of extracted panel labels
                - ground_truth_labels: List of ground truth panel labels
        """
        # Extract just the panel labels in order
        extracted_labels = [p["panel_label"] for p in extracted_panels]
        ground_truth_labels = [p["panel_label"] for p in ground_truth_panels]

        # If either list is empty, return 0 and empty lists
        if not extracted_labels or not ground_truth_labels:
            return 0.0, [], []

        # Return score and both sequences
        return (
            1.0 if extracted_labels == ground_truth_labels else 0.0,
            extracted_labels,
            ground_truth_labels,
        )

    def _calculate_data_availability_scores(
        self,
        extracted_sources: List[Dict[str, str]],
        ground_truth_sources: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """Calculate scores for data availability matching.

        Args:
            extracted_sources: List of extracted data sources
            ground_truth_sources: List of ground truth data sources

        Returns:
            Dict with scores for database, accession, url and combined
        """
        if not extracted_sources or not ground_truth_sources:
            return {
                "database_score": 0.0,
                "accession_score": 0.0,
                "url_score": 0.0,
                "combined_score": 0.0,
            }

        # Initialize counters for each metric
        database_matches = 0
        accession_matches = 0
        url_matches = 0

        # For each ground truth source, find best match in extracted sources
        for gt_source in ground_truth_sources:
            for ext_source in extracted_sources:
                # Database matching (partial)
                if (
                    gt_source["database"].lower() in ext_source["database"].lower()
                    or ext_source["database"].lower() in gt_source["database"].lower()
                ):
                    database_matches += 1

                # Exact matching for accession and URL
                if gt_source["accession_number"] == ext_source["accession_number"]:
                    accession_matches += 1
                if gt_source["url"] == ext_source["url"]:
                    url_matches += 1

        # Calculate scores
        total_sources = len(ground_truth_sources)
        database_score = database_matches / total_sources
        accession_score = accession_matches / total_sources
        url_score = url_matches / total_sources
        combined_score = (database_score + accession_score + url_score) / 3

        return {
            "database_score": database_score,
            "accession_score": accession_score,
            "url_score": url_score,
            "combined_score": combined_score,
        }

    def _calculate_panel_source_scores(
        self,
        extracted_figures: List[Figure],
        ground_truth_figures: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate scores for panel source assignments."""
        scores = {}
        total_score = 0.0
        panel_count = 0

        for gt_figure in ground_truth_figures:
            ext_figure = next(
                (
                    f
                    for f in extracted_figures
                    if f.figure_label == gt_figure["figure_label"]
                ),
                None,
            )

            for gt_panel in gt_figure["panels"]:
                panel_key = f"{gt_figure['figure_label']}_{gt_panel['panel_label']}"
                expected_files = gt_panel["sd_files"]

                if ext_figure:
                    ext_panel = next(
                        (
                            p
                            for p in ext_figure.panels
                            if p.panel_label == gt_panel["panel_label"]
                        ),
                        None,
                    )
                    actual_files = ext_panel.sd_files if ext_panel else []
                else:
                    actual_files = []

                # Calculate score for this panel
                if not expected_files:
                    panel_score = 1.0 if not actual_files else 0.0
                else:
                    correct_matches = len(
                        set(expected_files).intersection(set(actual_files))
                    )
                    panel_score = correct_matches / len(expected_files)

                scores[panel_key] = panel_score
                total_score += panel_score
                panel_count += 1

        # Add average score
        scores["average_score"] = total_score / panel_count if panel_count > 0 else 0.0

        return scores

    def _run_assign_panel_source(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run panel source assignment test."""
        try:
            start_time = time.time()

            # Get manuscript path
            msid = test_case["msid"]

            # Create temporary extraction directory
            temp_extract_dir = self.base_output_dir / "temp_extracts" / msid
            temp_extract_dir.mkdir(parents=True, exist_ok=True)

            # Get ZIP path
            zip_path = Path(self.config["manuscript_dir"]) / f"{msid}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")

            # Use XML extractor to get structure and extract files
            extractor = XMLStructureExtractor(str(zip_path), str(temp_extract_dir))
            zip_structure = extractor.extract_structure()

            # Load prompts from development configuration
            with open(self.config["prompts_source"]) as f:
                dev_config = yaml.safe_load(f)

            # Get the default configuration which contains our pipeline setup
            pipeline_config = dev_config.get("default", {}).get("pipeline", {})
            if not pipeline_config:
                raise ValueError("No pipeline configuration found in default profile")

            # Provider-specific configuration
            provider = test_case["provider"]

            # Create extractor configuration
            extractor_config = {
                "api_key": os.environ.get(f"{provider.upper()}_API_KEY"),
                "pipeline": {
                    "assign_panel_source": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config["assign_panel_source"][
                                    provider
                                ]["prompts"]["system"],
                                "user": pipeline_config["assign_panel_source"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    }
                },
                "extraction_dir": str(temp_extract_dir),
            }

            # Create prompt handler with the pipeline configuration
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create assigner instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                panel_assigner = PanelSourceAssignerOpenAI(
                    extractor_config, prompt_handler
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Assign panel sources
            logger.info("Assigning panel sources")
            extracted_figures = panel_assigner.assign_panel_source(zip_structure)

            duration_ms = (time.time() - start_time) * 1000

            # Debug logging
            logger.info(
                f"Number of ground truth figures: {len(ground_truth['figures'])}"
            )
            logger.info(f"Number of extracted figures: {len(extracted_figures)}")

            # Calculate scores
            scores = self._calculate_panel_source_scores(
                extracted_figures, ground_truth["figures"]
            )

            for fig in ground_truth["figures"]:
                for panel in fig["panels"]:
                    print(f"  Panel {panel['panel_label']}: {panel['sd_files']}")

            for fig in extracted_figures:
                print(f"EX Figure {fig.figure_label}: {len(fig.panels)} panels")
                for panel in fig.panels:
                    print(f"  Panel {panel.panel_label}: {panel.sd_files}")

            # Calculate scores and prepare outputs
            specific_outputs = {}
            scores = {}

            for gt_figure, ext_figure in zip(
                ground_truth["figures"], extracted_figures
            ):
                figure_label = gt_figure["figure_label"]

                for gt_panel in gt_figure["panels"]:
                    panel_label = gt_panel["panel_label"]

                    # get ext_panel from ext_figure if ext_panel.panel_label == panel_label
                    ext_panel = next(
                        (p for p in ext_figure.panels if panel_label in p.panel_label),
                        None,
                    )

                    panel_key = f"{figure_label}_{panel_label}"
                    expected_files = gt_panel["sd_files"]
                    actual_files = ext_panel.sd_files if ext_panel else []

                    # Calculate score for this panel
                    if not expected_files:
                        panel_score = 1.0 if not actual_files else 0.0
                    else:
                        correct_matches = len(
                            set(expected_files).intersection(set(actual_files))
                        )
                        panel_score = correct_matches / len(expected_files)

                    print(f"Panel {panel_label}:")
                    print(f"Expected: {expected_files}")
                    print(f"Actual: {actual_files}")
                    print(f"Score: {panel_score}")

                    scores[panel_key] = panel_score
                    specific_outputs[panel_key] = {
                        "actual": str(actual_files),
                        "expected": str(expected_files),
                        "figure_label": figure_label,
                        "panel_label": panel_label,
                        "score": panel_score,
                    }
            else:
                print(f"No matching extracted figure found for {figure_label}")

            for panel_id, data in specific_outputs.items():
                print(f"Adding result for panel: {panel_id}")
                panel_row = {
                    **test_case,
                    "duration_ms": duration_ms,
                    "figure_label": data["figure_label"],
                    "panel_label": data["panel_label"],
                    "task": "panel_source_assignment",
                    "score": data["score"],
                }

                self._fill_results_bag(
                    test_case=panel_row,
                    result={
                        "input": str(zip_structure),
                        "ai_response": str(extracted_figures),
                    },
                    actual_output=data["actual"],
                    expected_output=data["expected"],
                )

            return {
                "input": str(zip_structure),
                "extracted_figures": [
                    {
                        "figure_label": fig.figure_label,
                        "panels": [
                            {"panel_label": p.panel_label, "sd_files": p.sd_files}
                            for p in fig.panels
                        ],
                    }
                    for fig in extracted_figures
                ],
                "ground_truth_figures": ground_truth["figures"],
                "scores": scores,
            }

        except Exception as e:
            logger.error(f"Error running panel source assignment test: {str(e)}")
            raise

        finally:
            # Cleanup temporary directory after processing
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

    def _fill_results_bag(
        self,
        test_case: Dict[str, Any],
        result: Dict[str, Any],
        actual_output: str = "",
        expected_output: str = "",
        actual_title: str = "",
        expected_title: str = "",
        figure_label: str = "",
        panel_sequence_score: float = 0.0,
        actual_panels: List[str] = [],
        expected_panels: List[str] = [],
        data_availability_scores: Dict[str, float] = {},
        panel_source_scores: Dict[str, float] = {},  # Added new parameter
    ) -> None:
        """Fill the results DataFrame with test metrics and metadata."""
        try:
            # Create base row for DataFrame
            base_row = {
                "pytest_obj": None,
                "status": "completed",
                "duration_ms": test_case.get("duration_ms", 0),
                "strategy": f"{test_case['provider']}_{test_case['model']}",
                "msid": test_case["msid"],
                "run": test_case["run"],
                "input": result.get("input", ""),
                "figure_label": figure_label,
                "ai_response": result.get("ai_response", ""),
                "timestamp": pd.Timestamp.now(),
                # Add AI model parameters
                "model": test_case["model"],
                "provider": test_case["provider"],
                "temperature": test_case["temperature"],
                "top_p": test_case["top_p"],
                # Add any other parameters that might be in test_case
                "frequency_penalty": test_case.get("frequency_penalty", None),
                "presence_penalty": test_case.get("presence_penalty", None),
            }

            rows_to_add = []

            # Add main caption row
            caption_row = base_row.copy()
            caption_row.update(
                {
                    "task": test_case.get("test_name", ""),
                    "actual": actual_output,
                    "expected": expected_output,
                }
            )

            # Calculate metrics for caption
            test_case_obj = LLMTestCase(
                input=result.get("input", ""),
                actual_output=actual_output,
                expected_output=expected_output,
            )

            metrics = get_metrics_for_task(test_case.get("test_name", ""))
            for metric in metrics:
                score = metric.measure(test_case_obj)
                caption_row[metric.name] = score
                caption_row[f"{metric.name}_success"] = metric.is_successful()
                caption_row[f"{metric.name}_threshold"] = metric.threshold

            rows_to_add.append(caption_row)

            # Only add title and panel sequence rows for extract_individual_captions task
            if test_case.get("test_name") == "extract_individual_captions":
                # Add title row if we have a title
                if actual_title:
                    title_row = base_row.copy()
                    title_row.update(
                        {
                            "task": f"{test_case.get('test_name')}_title",
                            "actual": actual_title,
                            "expected": expected_title,
                        }
                    )

                    # Calculate metrics for title
                    test_case_obj = LLMTestCase(
                        input=result.get("input", ""),
                        actual_output=actual_title,
                        expected_output=expected_title,
                    )

                    for metric in metrics:
                        score = metric.measure(test_case_obj)
                        title_row[metric.name] = score
                        title_row[f"{metric.name}_success"] = metric.is_successful()
                        title_row[f"{metric.name}_threshold"] = metric.threshold

                    rows_to_add.append(title_row)

                # Add panel sequence row
                panel_row = base_row.copy()
                panel_row.update(
                    {
                        "task": "panel_sequence",
                        "actual": str(actual_panels),  # Store the actual panel sequence
                        "expected": str(
                            expected_panels
                        ),  # Store the expected panel sequence
                    }
                )

                # Set all other metrics to NaN except for a custom accuracy
                for metric in metrics:
                    panel_row[metric.name] = float("nan")
                    panel_row[f"{metric.name}_success"] = float("nan")
                    panel_row[f"{metric.name}_threshold"] = float("nan")

                # Add the panel sequence score as its own accuracy metric
                panel_row["panel_sequence_accuracy"] = panel_sequence_score
                panel_row["panel_sequence_accuracy_success"] = (
                    panel_sequence_score == 1.0
                )
                panel_row["panel_sequence_accuracy_threshold"] = 1.0

                rows_to_add.append(panel_row)

            # For data availability task
            if (
                test_case.get("test_name") == "extract_data_availability"
                and data_availability_scores
            ):
                score_types = {
                    "database": "Database Name Matching",
                    "accession": "Accession Number Matching",
                    "url": "URL Matching",
                    "combined": "Combined Score",
                }

                for score_type, description in score_types.items():
                    score_row = base_row.copy()
                    score_row.update(
                        {
                            "task": f"data_availability_{score_type}",
                            "actual": actual_output[score_type][
                                "actual"
                            ],  # Use specific output
                            "expected": expected_output[score_type][
                                "expected"
                            ],  # Use specific output
                            "accuracy": data_availability_scores[f"{score_type}_score"],
                            "accuracy_success": data_availability_scores[
                                f"{score_type}_score"
                            ]
                            == 1.0,
                            "accuracy_threshold": 1.0,
                            "description": description,
                        }
                    )

                    # Set other metrics to NaN
                    for metric in metrics:
                        score_row[metric.name] = float("nan")
                        score_row[f"{metric.name}_success"] = float("nan")
                        score_row[f"{metric.name}_threshold"] = float("nan")

                    rows_to_add.append(score_row)

            # For panel source assignment task
            # For panel source assignment task
            if (
                test_case.get("test_name") == "panel_source_assignment"
                and panel_source_scores
            ):
                for panel_id, score in panel_source_scores.items():
                    if panel_id == "average_score":  # Skip average for now
                        continue

                    score_row = base_row.copy()
                    score_row.update(
                        {
                            "task": "panel_source_assignment",
                            "panel_id": panel_id,
                            "actual": actual_output,  # Direct string of files
                            "expected": expected_output,  # Direct string of files
                            "accuracy": score,
                            "accuracy_success": score == 1.0,
                            "accuracy_threshold": 1.0,
                            "description": "Panel Source File Assignment",
                            "figure_label": test_case.get("figure_label", ""),
                            "panel_label": test_case.get("panel_label", ""),
                        }
                    )

                    # Set other metrics to NaN
                    for metric in metrics:
                        score_row[metric.name] = float("nan")
                        score_row[f"{metric.name}_success"] = float("nan")
                        score_row[f"{metric.name}_threshold"] = float("nan")

                    rows_to_add.append(score_row)
                # Add average score row
                if "average_score" in panel_source_scores:
                    avg_row = base_row.copy()
                    avg_row.update(
                        {
                            "task": "panel_source_assignment_average",
                            "panel_id": "average",
                            "actual": "N/A",
                            "expected": "N/A",
                            "accuracy": panel_source_scores["average_score"],
                            "accuracy_success": panel_source_scores["average_score"]
                            == 1.0,
                            "accuracy_threshold": 1.0,
                            "description": "Panel Source Assignment Average Score",
                        }
                    )
                    rows_to_add.append(avg_row)

            # Append all rows to DataFrame
            self.results_df = pd.concat(
                [self.results_df, pd.DataFrame(rows_to_add)], ignore_index=True
            )

        except Exception as e:
            logger.error(f"Error filling results: {str(e)}", exc_info=True)
            raise

    def cleanup(self):
        """Clean up extraction directories."""
        try:
            if self.extracts_dir.exists():
                shutil.rmtree(self.extracts_dir)
        except Exception as e:
            logger.error(f"Error cleaning up extraction directories: {str(e)}")

    def save_results(self):
        """Save results DataFrame to CSV file."""
        try:
            # Save DataFrame to CSV
            results_csv_path = self.results_dir / "metrics.csv"
            self.results_df.to_csv(results_csv_path, index=False)

            # Also save as Excel for easier viewing (optional)
            results_excel_path = self.results_dir / "metrics.xlsx"
            self.results_df.to_excel(results_excel_path, index=False)

            logger.info(f"Saved metrics to {results_csv_path} and {results_excel_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise


# Load configuration once at module level
def load_config() -> Dict[str, Any]:
    """Load benchmark configuration."""
    config_path = os.environ.get("BENCHMARK_CONFIG", "config.benchmark.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


# Initialize runner with configuration
runner = BenchmarkRunner(load_config())

logger.info("Starting test generation")


def pytest_generate_tests(metafunc):
    """Generate test cases."""
    if "test_case" in metafunc.fixturenames:
        logger.info("Generating test cases")
        metafunc.parametrize("test_case", runner.get_test_cases())
        logger.info("Test cases generated")


logger.info("Starting test execution")


def test_pipeline(test_case: Dict[str, Any]) -> None:
    """Run pipeline test with generated test case."""
    try:
        logger.info("Starting test case")
        logger.info(f"Test case parameters: {test_case}")

        # Run the test and collect results
        result = runner.run_test(test_case)
        logger.info("Test case completed")

        # Save results after each test
        results_path = runner.results_dir / "results.json"

        # Update results file
        if results_path.exists():
            # Load existing results
            with open(results_path) as f:
                existing_results = json.load(f)
        else:
            existing_results = []

        # Add new result
        existing_results.append(result)

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2)

        # Save metrics DataFrame
        runner.save_results()

        logger.info(f"Updated results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up any temporary files
        temp_extracts_dir = runner.base_output_dir / "temp_extracts"
        if temp_extracts_dir.exists():
            shutil.rmtree(temp_extracts_dir)


# Also add this to make sure we see all logs
@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # This ensures we override any existing configuration
    )

    # Test that logging is working
    logger.info("Logging configured for tests")
    return None
