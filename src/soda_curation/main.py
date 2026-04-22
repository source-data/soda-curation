"""Main entry point for SODA curation pipeline."""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

from ._main_utils import (
    calculate_hallucination_score,
    cleanup_extract_dir,
    setup_extract_dir,
    validate_paths,
)
from .config import ConfigurationLoader
from .data_storage import save_figure_data, save_zip_structure
from .logging_config import setup_logging
from .pipeline.assign_panel_source.assign_panel_source_anthropic import (
    PanelSourceAssignerAnthropic,
)
from .pipeline.assign_panel_source.assign_panel_source_openai import (
    PanelSourceAssignerOpenAI,
)
from .pipeline.data_availability.data_availability_anthropic import (
    DataAvailabilityExtractorAnthropic,
)
from .pipeline.data_availability.data_availability_openai import (
    DataAvailabilityExtractorOpenAI,
)
from .pipeline.extract_captions.extract_captions_anthropic import (
    FigureCaptionExtractorAnthropic,
)
from .pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
)
from .pipeline.extract_sections.extract_sections_anthropic import (
    SectionExtractorAnthropic,
)
from .pipeline.extract_sections.extract_sections_openai import SectionExtractorOpenAI
from .pipeline.manuscript_structure.manuscript_structure import (
    CustomJSONEncoder,
    ZipStructure,
)
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
from .pipeline.match_caption_panel.match_caption_panel_anthropic import (
    MatchPanelCaptionAnthropic,
)
from .pipeline.match_caption_panel.match_caption_panel_openai import (
    MatchPanelCaptionOpenAI,
)
from .pipeline.prompt_handler import PromptHandler

# Import QC module (to be implemented)
from .qc.qc_pipeline import QCPipeline

# from src.soda_curation.pipeline.extract_sections.extract_sections_openai import (
#     SectionExtractorOpenAI,
# )


logger = logging.getLogger(__name__)
SUPPORTED_AI_PROVIDERS = {"openai", "anthropic"}
AI_PROVIDER_STEPS = (
    "extract_sections",
    "extract_caption_title",
    "extract_panel_sequence",
    "extract_data_sources",
    "match_caption_panel",
    "assign_panel_source",
)


@dataclass
class StepFailure:
    """Structured record of recoverable step failures."""

    step: str
    reason: str


def _execute_pipeline_step(
    step_name: str,
    runner,
    run_id: str,
    critical: bool,
    recoverable_failures: list[StepFailure],
):
    """Execute a pipeline step with structured logging and severity handling."""
    started = time.perf_counter()
    logger.info(
        "Pipeline step started",
        extra={"run_id": run_id, "step": step_name, "critical": critical},
    )
    try:
        result = runner()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Pipeline step completed",
            extra={
                "run_id": run_id,
                "step": step_name,
                "critical": critical,
                "elapsed_ms": elapsed_ms,
            },
        )
        return result
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if critical:
            logger.error(
                "Critical pipeline step failed",
                extra={
                    "run_id": run_id,
                    "step": step_name,
                    "critical": True,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                },
            )
            raise

        logger.warning(
            "Recoverable pipeline step failed; continuing with fallback behavior",
            extra={
                "run_id": run_id,
                "step": step_name,
                "critical": False,
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
            },
        )
        recoverable_failures.append(StepFailure(step=step_name, reason=str(exc)))
        return None


def _validate_ai_provider_config(config: dict, ai_provider: str, run_id: str) -> None:
    """Ensure runtime provider selection is explicit and configuration is consistent."""
    if ai_provider not in SUPPORTED_AI_PROVIDERS:
        raise ValueError(
            f"Unsupported ai_provider '{ai_provider}'. Expected one of {sorted(SUPPORTED_AI_PROVIDERS)}."
        )

    pipeline_config = config.get("pipeline", {})
    if not isinstance(pipeline_config, dict):
        raise ValueError("Invalid configuration: missing `pipeline` section.")

    ignored_provider = "anthropic" if ai_provider == "openai" else "openai"
    for step_name in AI_PROVIDER_STEPS:
        step_config = pipeline_config.get(step_name, {})
        if not isinstance(step_config, dict):
            raise ValueError(
                f"Invalid configuration for step '{step_name}': expected mapping."
            )

        has_openai = "openai" in step_config
        has_anthropic = "anthropic" in step_config
        assert has_openai or has_anthropic, (
            f"Configuration error in step '{step_name}': no provider block found. "
            "Define exactly one of 'openai' or 'anthropic'."
        )
        assert not (has_openai and has_anthropic), (
            f"Configuration error in step '{step_name}': both 'openai' and 'anthropic' "
            "are defined. Define only one provider per step."
        )

        configured_provider = "openai" if has_openai else "anthropic"
        if configured_provider != ai_provider:
            raise ValueError(
                "Selected provider does not match step configuration. "
                f"step={step_name}, configured_provider={configured_provider}, "
                f"ai_provider={ai_provider}, ignored_provider={ignored_provider}"
            )

    logger.info(
        "AI provider configuration validated",
        extra={
            "run_id": run_id,
            "ai_provider": ai_provider,
        },
    )


def run_qc_pipeline_async(
    config, zip_structure: ZipStructure, extract_dir: Path, figure_data=None
) -> dict:
    """Run QC pipeline in a separate thread."""
    try:
        logger.info("*** QC PIPELINE TRIGGERED - include_qc=true IS WORKING! ***")
        logger.info(
            f"Received {len(figure_data) if figure_data else 0} figures for QC processing"
        )

        qc_pipeline = QCPipeline(config, extract_dir)
        qc_results = qc_pipeline.run(zip_structure, figure_data)

        # Store QC results in zip_structure if possible
        if hasattr(zip_structure, "qc_results"):
            zip_structure.qc_results = qc_results

        logger.info("QC pipeline completed successfully")
        return qc_results
    except ImportError as e:
        logger.error(f"QC pipeline module not found: {str(e)}")
        return {"error": f"QC module not available: {str(e)}"}
    except Exception as e:
        logger.error(f"QC pipeline failed: {str(e)}")
        return {"error": str(e)}


def main(zip_path: str, config_path: str, output_path: Optional[str] = None) -> str:
    """
    Main entry point for SODA curation pipeline.

    Args:
        zip_path: Path to input ZIP file
        config_path: Path to configuration file
        output_path: Optional path to output JSON file

    Returns:
        JSON string containing processing results

    Raises:
        Various exceptions for validation and processing errors
    """
    # Validate inputs
    validate_paths(zip_path, config_path, output_path)

    # Load configuration
    config_loader = ConfigurationLoader(config_path)

    # Setup logging based on environment
    setup_logging(config_loader.config)
    run_id = uuid4().hex[:10]
    logger.info("Starting SODA curation pipeline", extra={"run_id": run_id})

    # Setup extraction directory
    extract_dir = setup_extract_dir()
    config_loader.config["extraction_dir"] = str(extract_dir)
    recoverable_failures: list[StepFailure] = []

    try:
        # Extract manuscript structure (first pipeline step)
        extractor = XMLStructureExtractor(zip_path, str(extract_dir))
        zip_structure = _execute_pipeline_step(
            step_name="extract_structure",
            runner=extractor.extract_structure,
            run_id=run_id,
            critical=True,
            recoverable_failures=recoverable_failures,
        )

        # Extract manuscript content for downstream AI steps
        manuscript_content = _execute_pipeline_step(
            step_name="extract_docx_content",
            runner=lambda: extractor.extract_docx_content(zip_structure.docx),
            run_id=run_id,
            critical=True,
            recoverable_failures=recoverable_failures,
        )
        zip_structure.manuscript_text = manuscript_content
        prompt_handler = PromptHandler(config_loader.config["pipeline"])

        # Select AI provider (default: openai)
        ai_provider = config_loader.config.get("ai_provider", "openai").lower()
        logger.info(
            f"Using AI provider: {ai_provider}",
            extra={"run_id": run_id, "ai_provider": ai_provider},
        )
        _validate_ai_provider_config(config_loader.config, ai_provider, run_id)

        # Extract relevant sections for the pipeline
        if ai_provider == "anthropic":
            section_extractor = SectionExtractorAnthropic(
                config_loader.config, prompt_handler
            )
        else:
            section_extractor = SectionExtractorOpenAI(
                config_loader.config, prompt_handler
            )
        (
            figure_legends,
            data_availability_text,
            zip_structure,
        ) = _execute_pipeline_step(
            step_name="extract_sections",
            runner=lambda: section_extractor.extract_sections(
                doc_content=manuscript_content, zip_structure=zip_structure
            ),
            run_id=run_id,
            critical=True,
            recoverable_failures=recoverable_failures,
        )

        # Extract individual captions from figure legends
        if ai_provider == "anthropic":
            caption_extractor = FigureCaptionExtractorAnthropic(
                config_loader.config, prompt_handler
            )
        else:
            caption_extractor = FigureCaptionExtractorOpenAI(
                config_loader.config, prompt_handler
            )
        zip_structure = _execute_pipeline_step(
            step_name="extract_individual_captions",
            runner=lambda: caption_extractor.extract_individual_captions(
                doc_content=figure_legends, zip_structure=zip_structure
            ),
            run_id=run_id,
            critical=True,
            recoverable_failures=recoverable_failures,
        )

        # Extract data sources from data availability section
        if ai_provider == "anthropic":
            data_availability_extractor = DataAvailabilityExtractorAnthropic(
                config_loader.config, prompt_handler
            )
        else:
            data_availability_extractor = DataAvailabilityExtractorOpenAI(
                config_loader.config, prompt_handler
            )
        zip_structure = _execute_pipeline_step(
            step_name="extract_data_sources",
            runner=lambda: data_availability_extractor.extract_data_sources(
                section_text=data_availability_text, zip_structure=zip_structure
            ),
            run_id=run_id,
            critical=True,
            recoverable_failures=recoverable_failures,
        )

        # Match panels with captions using object detection
        if ai_provider == "anthropic":
            panel_matcher = MatchPanelCaptionAnthropic(
                config=config_loader.config,
                prompt_handler=prompt_handler,
                extract_dir=extractor.manuscript_extract_dir,
            )
        else:
            panel_matcher = MatchPanelCaptionOpenAI(
                config=config_loader.config,
                prompt_handler=prompt_handler,
                extract_dir=extractor.manuscript_extract_dir,
            )
        panel_processing_result = _execute_pipeline_step(
            step_name="match_caption_panel",
            runner=lambda: panel_matcher.process_figures(zip_structure),
            run_id=run_id,
            critical=False,
            recoverable_failures=recoverable_failures,
        )
        if panel_processing_result is not None:
            zip_structure = panel_processing_result

        # Get figure images and captions for QC pipeline
        figure_data = _execute_pipeline_step(
            step_name="collect_qc_figure_payloads",
            runner=panel_matcher.get_figure_images_and_captions,
            run_id=run_id,
            critical=False,
            recoverable_failures=recoverable_failures,
        )
        if figure_data is None:
            figure_data = []

        # Assign panel source
        if ai_provider == "anthropic":
            panel_source_assigner = PanelSourceAssignerAnthropic(
                config_loader.config, prompt_handler, extractor.manuscript_extract_dir
            )
        else:
            panel_source_assigner = PanelSourceAssignerOpenAI(
                config_loader.config, prompt_handler, extractor.manuscript_extract_dir
            )
        processed_figures = _execute_pipeline_step(
            step_name="assign_panel_source",
            runner=lambda: panel_source_assigner.assign_panel_source(
                zip_structure,
            ),
            run_id=run_id,
            critical=False,
            recoverable_failures=recoverable_failures,
        )
        # Preserve all ZipStructure data while updating figures
        if processed_figures is not None:
            zip_structure.figures = processed_figures
        else:
            logger.warning(
                "Proceeding without panel source assignments due to recoverable failure",
                extra={"run_id": run_id, "step": "assign_panel_source"},
            )

        # Update total costs before returning results
        zip_structure.update_total_cost()

        # Check for possible hallucinations
        zip_structure.locate_captions_hallucination_score = (
            calculate_hallucination_score(
                zip_structure.ai_response_locate_captions, manuscript_content
            )
        )
        zip_structure.locate_data_section_hallucination_score = (
            calculate_hallucination_score(
                zip_structure.data_availability["section_text"], manuscript_content
            )
        )
        for fig in zip_structure.figures:
            if fig.figure_caption:
                if fig.hallucination_score == 1:
                    fig.hallucination_score = calculate_hallucination_score(
                        fig.figure_caption, manuscript_content
                    )

        # Save data for QC pipeline
        if output_path:
            # Extract directory and base filename without extension
            output_path_obj = Path(output_path)
            output_dir = output_path_obj.parent
            base_filename = output_path_obj.stem

            # Create filenames for QC data
            figure_data_path = str(output_dir / f"{base_filename}_figure_data.json")
            zip_structure_path = str(
                output_dir / f"{base_filename}_zip_structure.pickle"
            )

            # Save the data files for QC pipeline
            save_figure_data(figure_data, figure_data_path)
            save_zip_structure(zip_structure, zip_structure_path)
            logger.info(
                f"Saved QC pipeline data: {figure_data_path} and {zip_structure_path}"
            )
        else:
            # Default location if no output path specified
            data_dir = Path("data/qc_data")
            data_dir.mkdir(parents=True, exist_ok=True)

            # Save with default filenames
            figure_data_path = str(data_dir / "figure_data.json")
            zip_structure_path = str(data_dir / "zip_structure.pickle")
            save_figure_data(figure_data, figure_data_path)
            save_zip_structure(zip_structure, zip_structure_path)
            logger.info(
                f"Saved QC pipeline data: {figure_data_path} and {zip_structure_path}"
            )

        # Convert to JSON using CustomJSONEncoder
        output_json = json.dumps(
            zip_structure, cls=CustomJSONEncoder, ensure_ascii=False, indent=2
        )

        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_json)

        if recoverable_failures:
            logger.warning(
                "Pipeline completed with recoverable failures",
                extra={
                    "run_id": run_id,
                    "recoverable_failure_count": len(recoverable_failures),
                    "recoverable_failures": [
                        {"step": item.step, "reason": item.reason}
                        for item in recoverable_failures
                    ],
                },
            )
        else:
            logger.info(
                "Pipeline completed without recoverable failures",
                extra={"run_id": run_id},
            )

        return output_json

    finally:
        cleanup_extract_dir(extract_dir)


def cli() -> int:
    """Command-line entry point for the ``soda-curation`` script."""
    parser = argparse.ArgumentParser(
        description="Process a ZIP file using SODA curation"
    )
    parser.add_argument("--zip", required=True, help="Path to the input ZIP file")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )
    parser.add_argument("--output", help="Path to the output JSON file")

    args = parser.parse_args()

    output_json = main(args.zip, args.config, args.output)
    if not args.output:
        print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
