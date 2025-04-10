"""Main entry point for SODA curation pipeline."""

import json
import logging
from pathlib import Path
from typing import Optional

from src.soda_curation.pipeline.extract_sections.extract_sections_smolagents import (
    SectionExtractorSmolagents,
)

from ._main_utils import (
    calculate_hallucination_score,
    clean_original_source_data_files,
    cleanup_extract_dir,
    setup_extract_dir,
    validate_paths,
)
from .config import ConfigurationLoader
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
from .pipeline.manuscript_structure.manuscript_structure import CustomJSONEncoder
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
from .pipeline.match_caption_panel.match_caption_panel_openai import (
    MatchPanelCaptionOpenAI,
)
from .pipeline.prompt_handler import PromptHandler

# from src.soda_curation.pipeline.extract_sections.extract_sections_openai import (
#     SectionExtractorOpenAI,
# )


logger = logging.getLogger(__name__)


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

        try:
            # Extract manuscript structure (first pipeline step)
            extractor = XMLStructureExtractor(zip_path, str(extract_dir))
            zip_structure = extractor.extract_structure()
            # Store the original source data files for each figure
            original_source_data_files = {
                fig.figure_label: list(fig.sd_files) for fig in zip_structure.figures
            }

            # Extract captions from figures (second pipeline step)
            manuscript_content = extractor.extract_docx_content(zip_structure.docx)
            prompt_handler = PromptHandler(config_loader.config["pipeline"])

            ##################################################
            ##################################################
            ##################################################
            ##################################################
            ##################################################
            ##################################################
            # Extract relevant sections for the pipeline
            # section_extractor = SectionExtractorOpenAI(
            #     config_loader.config, prompt_handler
            # )
            # (
            #     figure_legends,
            #     data_availability_text,
            #     zip_structure,
            # ) = section_extractor.extract_sections(
            #     doc_content=manuscript_content, zip_structure=zip_structure
            # )
            # Extract relevant sections for the pipeline
            section_extractor = SectionExtractorSmolagents(
                config_loader.config, prompt_handler
            )
            (
                figure_legends,
                data_availability_text,
                zip_structure,
            ) = section_extractor.extract_sections(
                doc_content=manuscript_content, zip_structure=zip_structure
            )
            exit()
            ##################################################
            ##################################################
            ##################################################
            ##################################################
            ##################################################
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

            # Assign panel source
            panel_source_assigner = PanelSourceAssignerOpenAI(
                config_loader.config, prompt_handler, extractor.manuscript_extract_dir
            )
            # Pass only
            processed_figures = panel_source_assigner.assign_panel_source(
                zip_structure,  # Pass figures list instead of whole structure
            )
            # Preserve all ZipStructure data while updating figures
            zip_structure.figures = processed_figures

            # Match panels with captions using object detection
            panel_matcher = MatchPanelCaptionOpenAI(
                config=config_loader.config,
                prompt_handler=prompt_handler,
                extract_dir=extractor.manuscript_extract_dir,  # Pass the manuscript-specific directory
            )
            _ = panel_matcher.process_figures(zip_structure)

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
                    fig.hallucination_score = calculate_hallucination_score(
                        fig.figure_caption, manuscript_content
                    )

                # Check for hallucinations in panel captions
                for panel in fig.panels:
                    if panel.panel_caption:
                        panel.hallucination_score = calculate_hallucination_score(
                            panel.panel_caption, manuscript_content
                        )

            zip_structure = clean_original_source_data_files(
                zip_structure, original_source_data_files
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

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            error_json = json.dumps({"error": str(e)})
            if output_path:
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(error_json)
            return error_json
    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        return json.dumps({"error": str(e)})

    finally:
        cleanup_extract_dir(extract_dir)


if __name__ == "__main__":
    import argparse

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
