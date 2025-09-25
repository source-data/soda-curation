"""Main entry point for SODA curation pipeline."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from ._main_utils import (
    calculate_hallucination_score,
    cleanup_extract_dir,
    setup_extract_dir,
    validate_paths,
)
from .config import ConfigurationLoader
from .data_storage import save_figure_data, save_zip_structure
from .logging_config import setup_logging
from .pipeline.assign_panel_source.assign_panel_source_openai import (
    PanelSourceAssignerOpenAI,
)
from .pipeline.data_availability.data_availability_openai import (
    DataAvailabilityExtractorOpenAI,
)
from .pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
)
from .pipeline.extract_sections.extract_sections_openai import SectionExtractorOpenAI
from .pipeline.manuscript_structure.manuscript_structure import (
    CustomJSONEncoder,
    ZipStructure,
)
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
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
    logger.info("Starting SODA curation pipeline")

    try:
        # Setup extraction directory
        extract_dir = setup_extract_dir()
        config_loader.config["extraction_dir"] = str(extract_dir)

        # Extract manuscript structure (first pipeline step)
        extractor = XMLStructureExtractor(zip_path, str(extract_dir))
        zip_structure = extractor.extract_structure()

        # Extract captions from figures (second pipeline step)
        manuscript_content = extractor.extract_docx_content(zip_structure.docx)
        prompt_handler = PromptHandler(config_loader.config["pipeline"])

        # Extract relevant sections for the pipeline
        section_extractor = SectionExtractorOpenAI(
            config_loader.config, prompt_handler
        )
        (
            figure_legends,
            data_availability_text,
            zip_structure,
        ) = section_extractor.extract_sections(
            doc_content=manuscript_content, zip_structure=zip_structure
        )
        # Extract individual captions from figure legends
        caption_extractor = FigureCaptionExtractorOpenAI(
            config_loader.config, prompt_handler
        )
        zip_structure = caption_extractor.extract_individual_captions(
            doc_content=figure_legends, zip_structure=zip_structure
        )

        # Extract data sources from data availability section
        data_availability_extractor = DataAvailabilityExtractorOpenAI(
            config_loader.config, prompt_handler
        )
        zip_structure = data_availability_extractor.extract_data_sources(
            section_text=data_availability_text, zip_structure=zip_structure
        )

        # Match panels with captions using object detection
        panel_matcher = MatchPanelCaptionOpenAI(
            config=config_loader.config,
            prompt_handler=prompt_handler,
            extract_dir=extractor.manuscript_extract_dir,  # Pass the manuscript-specific directory
        )
        _ = panel_matcher.process_figures(zip_structure)

        # Get figure images and captions for QC pipeline
        figure_data = panel_matcher.get_figure_images_and_captions()

        # Assign panel source
        panel_source_assigner = PanelSourceAssignerOpenAI(
            config_loader.config, prompt_handler, extractor.manuscript_extract_dir
        )
        processed_figures = panel_source_assigner.assign_panel_source(
            zip_structure,  # Pass figures list instead of whole structure
        )
        # Preserve all ZipStructure data while updating figures
        zip_structure.figures = processed_figures

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

        return output_json

    finally:
        cleanup_extract_dir(extract_dir)


if __name__ == "__main__":
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
