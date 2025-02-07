"""OpenAI implementation of caption extraction."""

import json
import logging
import os
from typing import Dict, List

import openai
from pydantic import BaseModel

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure

# from ..prompt_handler import PromptHandler
from .extract_captions_base import FigureCaptionExtractor

logger = logging.getLogger(__name__)


class LocateFigureCaptions(BaseModel):
    figure_legends: str


class IndividualCaption(BaseModel):
    figure_label: str
    caption_title: str
    figure_caption: str


class ExtractIndividualCaptions(BaseModel):
    figures: List[IndividualCaption]


class FigureCaptionExtractorOpenAI(FigureCaptionExtractor):
    """Implementation of caption extraction using OpenAI's GPT models."""

    def __init__(self, config: Dict, prompt_handler):
        """Initialize with OpenAI configuration."""
        super().__init__(config, prompt_handler)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = openai.OpenAI(api_key=api_key)

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        # Validate model
        valid_models = ["gpt-4o", "gpt-4o-mini"]
        for step_ in ["locate_captions", "extract_individual_captions"]:
            config_ = self.config["pipeline"][step_]["openai"]
            model_ = config_.get("model", "gpt-4o")
            if model_ not in valid_models:
                raise ValueError(
                    f"Invalid model: {model_}. Must be one of {valid_models}"
                )

            # Validate numerical parameters
            if not 0 <= config_.get("temperature", 0.1) <= 2:
                raise ValueError(
                    f"Temperature must be between 0 and 2 for step: `{step_}`, value: `{config_.get('temperature', 1.0)}`"
                )
            if not 0 <= config_.get("top_p", 1.0) <= 1:
                raise ValueError(
                    f"Top_p must be between 0 and 1 for step: `{step_}`, value: `{config_.get('top_p', 1.0)}`"
                )
            if (
                "frequency_penalty" in config_
                and not -2 <= config_["frequency_penalty"] <= 2
            ):
                raise ValueError(
                    f"Frequency penalty must be between -2 and 2 for step: `{step_}`, value: `{config_.get('frequency_penalty', 0.)}`"
                )
            if (
                "presence_penalty" in config_
                and not -2 <= config_["presence_penalty"] <= 2
            ):
                raise ValueError(
                    f"Presence penalty must be between -2 and 2 for step: `{step_}`, value: `{config_.get('presence_penalty', 0.)}`"
                )

    def locate_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> (str, ZipStructure):
        """Call OpenAI API to locate figure captions."""
        # Get prompts with variables substituted
        prompts = self.prompt_handler.get_prompt(
            step="locate_captions",
            variables={
                "expected_figure_count": len(zip_structure.figures),
                "expected_figure_labels": [
                    figure.figure_label for figure in zip_structure.figures
                ],
                "manuscript_text": doc_content,
            },
        )

        # Prepare messages
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]
        config_ = self.config["pipeline"]["locate_captions"]["openai"]
        model_ = config_.get("model", "gpt-4o")
        response = self.client.beta.chat.completions.parse(
            model=model_,
            messages=messages,
            response_format=LocateFigureCaptions,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
        )

        # Updating the token usage
        update_token_usage(zip_structure.cost.locate_captions, response, model_)

        # Update the `zip_structure` object wit the answer
        zip_structure.ai_response_locate_captions = response.choices[0].message.content

        return (
            json.loads(response.choices[0].message.content)["figure_legends"],
            zip_structure,
        )

    def extract_individual_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> dict:
        """Call OpenAI API to extract individual captions."""

        caption_section, zip_structure = self.locate_captions(
            doc_content, zip_structure
        )
        # Get prompts with variables substituted
        prompts = self.prompt_handler.get_prompt(
            step="extract_individual_captions",
            variables={
                "expected_figure_count": len(zip_structure.figures),
                "expected_figure_labels": [
                    figure.figure_label for figure in zip_structure.figures
                ],
                "figure_captions": caption_section,
            },
        )

        # Prepare messages
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]
        config_ = self.config["pipeline"]["extract_individual_captions"]["openai"]
        model_ = config_.get("model", "gpt-4o")

        response = self.client.beta.chat.completions.parse(
            model=model_,
            messages=messages,
            response_format=ExtractIndividualCaptions,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
        )

        # Updating the token usage
        update_token_usage(
            zip_structure.cost.extract_individual_captions, response, model_
        )

        # Updating the figure captions
        import pdb

        pdb.set_trace()
        self._update_figures_with_captions(
            zip_structure, json.loads(response.choices[0].message.content)["figures"]
        )
        zip_structure.ai_response_extract_individual_captions = response.choices[
            0
        ].message.content

        return (
            json.loads(response.choices[0].message.content)["figures"],
            zip_structure,
        )


# import logging
# import time
# from typing import Dict, Optional

# import openai

# from ..cost_tracking import update_token_usage
# from ..manuscript_structure.manuscript_structure import ZipStructure
# from .extract_captions_base import FigureCaptionExtractor
# from .extract_captions_prompts import (
#     EXTRACT_CAPTIONS_PROMPT,
#     LOCATE_CAPTIONS_PROMPT,
#     get_extract_captions_prompt,
#     get_locate_captions_prompt,
# )

# logger = logging.getLogger(__name__)


# class FigureCaptionExtractorGpt(FigureCaptionExtractor):
#     """Implementation of caption extraction using OpenAI's GPT models."""

#     def __init__(self, config: Dict):
#         """Initialize with OpenAI configuration."""
#         super().__init__(config)
#         self.client = openai.OpenAI(api_key=self.config["api_key"])
#         self.model = self.config.get("model", "gpt-4-1106-preview")
#         self.locate_assistant = self._setup_locate_assistant()
#         self.extract_assistant = self._setup_extract_assistant()
#         self.zip_structure = None  # Initialize to None
#         logger.info(f"Initialized with model: {self.model}")

#     def _setup_locate_assistant(self):
#         """Set up or retrieve the OpenAI assistant for locating captions."""
#         assistant_id = self.config.get("caption_location_assistant_id")
#         if not assistant_id:
#             raise ValueError(
#                 "caption_location_assistant_id is not set in configuration"
#             )

#         return self.client.beta.assistants.update(
#             assistant_id,
#             model=self.config["model"],
#             instructions=LOCATE_CAPTIONS_PROMPT,
#         )

#     def _setup_extract_assistant(self):
#         """Set up or retrieve the OpenAI assistant for extracting captions."""
#         assistant_id = self.config.get("caption_extraction_assistant_id")
#         if not assistant_id:
#             raise ValueError(
#                 "caption_extraction_assistant_id is not set in configuration"
#             )

#         return self.client.beta.assistants.update(
#             assistant_id, model=self.model, instructions=EXTRACT_CAPTIONS_PROMPT
#         )

#     def _cleanup_thread(self, thread_id: str):
#         """
#         Helper method to clean up a thread and its messages.

#         Args:
#             thread_id: The ID of the thread to clean up
#         """
#         try:
#             # Delete the thread which also deletes all associated messages
#             self.client.beta.threads.delete(thread_id=thread_id)
#             logger.info(f"Successfully cleaned up thread {thread_id}")
#         except Exception as e:
#             logger.error(f"Error cleaning up thread {thread_id}: {str(e)}")

#     def _locate_figure_captions(
#         self,
#         doc_string: str,
#         expected_figure_count: int = None,
#         expected_figure_labels: str = "",
#     ) -> Optional[str]:
#         """
#         Locate figure captions using AI, optionally using both DOCX and PDF files.

#         Args:
#             docx_path (str): Path to DOCX file
#             pdf_path (str): Optional path to PDF file
#             expected_figure_count (int): Expected number of figures (from Figure 1 to X)

#         Returns:
#             Optional[str]: Found captions text or None
#         """
#         try:
#             # Create message with enhanced prompt that mentions all available files
#             message_content = get_locate_captions_prompt(
#                 manuscript_text=doc_string,
#                 expected_figure_count=expected_figure_count,
#                 expected_figure_labels=expected_figure_labels,
#             )

#             thread = self.client.beta.threads.create(
#                 messages=[
#                     {"role": "user", "content": message_content},
#                 ]
#             )

#             # Rest of the assistant communication code...
#             run_ = self.client.beta.threads.runs.create(
#                 thread_id=thread.id, assistant_id=self.locate_assistant.id
#             )

#             # Wait for completion with timeout
#             start_time = time.time()
#             timeout = 300  # 5 minutes timeout
#             while run_.status not in ["completed", "failed"]:
#                 if time.time() - start_time > timeout:
#                     logger.error("Assistant run timed out")
#                     return None

#                 time.sleep(1)
#                 run_ = self.client.beta.threads.runs.retrieve(
#                     thread_id=thread.id, run_id=run_.id
#                 )

#                 if run_.status == "failed":
#                     logger.error(f"Assistant run failed: {run_.last_error}")
#                     return None

#             # Get response
#             messages = self.client.beta.threads.messages.list(thread_id=thread.id)
#             # Track token usage
#             update_token_usage(
#                 self.zip_structure.cost.extract_captions, run_, self.model
#             )

#             if messages.data:
#                 located_captions = messages.data[0].content[0].text.value
#                 self._cleanup_thread(thread.id)
#                 return located_captions
#             return None
#         except Exception as e:
#             logger.error(f"Error locating figure captions: {str(e)}")
#             return None

#     def _extract_individual_captions(
#         self, all_captions: str, expected_figure_count: int, expected_figure_labels: str
#     ) -> Dict[str, str]:
#         """Extract individual figure captions using AI."""
#         try:
#             # Create thread for extraction
#             message_content = get_extract_captions_prompt(
#                 figure_captions=all_captions,
#                 expected_figure_count=expected_figure_count,
#                 expected_figure_labels=expected_figure_labels,
#             )

#             thread = self.client.beta.threads.create(
#                 messages=[
#                     # {"role": "system", "content": EXTRACT_CAPTIONS_PROMPT},
#                     {"role": "user", "content": message_content},
#                 ]
#             )

#             # Run the assistant
#             run_ = self.client.beta.threads.runs.create(
#                 thread_id=thread.id, assistant_id=self.extract_assistant.id
#             )

#             # Wait for completion with timeout
#             start_time = time.time()
#             timeout = 300  # 5 minutes timeout
#             while run_.status not in ["completed", "failed"]:
#                 if time.time() - start_time > timeout:
#                     logger.error("Assistant run timed out")
#                     return {}

#                 time.sleep(1)
#                 run_ = self.client.beta.threads.runs.retrieve(
#                     thread_id=thread.id, run_id=run_.id
#                 )

#                 if run_.status == "failed":
#                     logger.error(f"Assistant run failed: {run_.last_error}")
#                     return {}

#             # Get response
#             messages = self.client.beta.threads.messages.list(thread_id=thread.id)
#             # Track token usage
#             update_token_usage(
#                 self.zip_structure.cost.extract_individual_captions, run_, self.model
#             )

#             if messages.data:
#                 self._cleanup_thread(thread.id)
#                 return messages.data[0].content[0].text.value

#         except Exception as e:
#             logger.error(f"Error extracting individual captions: {str(e)}")

#             return {}

#     def extract_captions(
#         self,
#         doc_content: str,
#         zip_structure,
#         expected_figure_count: int,
#         expected_figure_labels: str,
#     ) -> ZipStructure:
#         """
#         Extract figure captions using pre-extracted document content.

#         Args:
#             doc_content (str): The pre-extracted document content in HTML format
#             zip_structure: Current ZIP structure
#             expected_figure_count (int): Expected number of figures
#             expected_figure_labels (str): Expected figure labels

#         Returns:
#             ZipStructure: Updated structure with extracted captions
#         """
#         self.zip_structure = zip_structure
#         try:
#             logger.info("Processing document content")

#             # First locate the figure captions section
#             located_captions = self._locate_figure_captions(
#                 doc_content, expected_figure_count, expected_figure_labels
#             )

#             if not located_captions:
#                 logger.error("Failed to locate figure captions section")
#                 return zip_structure

#             # Store the located captions section
#             zip_structure.ai_response_locate_captions = located_captions

#             # Now extract individual captions from the located section
#             extracted_captions_response = self._extract_individual_captions(
#                 located_captions, expected_figure_count, expected_figure_labels
#             )
#             # Store the raw response from caption extraction
#             zip_structure.ai_response_extract_captions = extracted_captions_response

#             # Parse the response into caption dictionary
#             captions = self._parse_response(extracted_captions_response)
#             logger.info(f"Extracted {len(captions)} individual captions")

#             # Process each figure
#             for figure in zip_structure.figures:
#                 normalized_label = self.normalize_figure_label(figure.figure_label)
#                 logger.info(f"Processing {normalized_label}")

#                 if normalized_label in captions:
#                     caption_info = captions[normalized_label]
#                     caption_text = caption_info["caption"]
#                     caption_title = caption_info["title"]

#                     # Validate caption
#                     is_valid, rouge_score, diff_text = self._validate_caption(
#                         zip_structure._full_docx, caption_text
#                     )

#                     figure.figure_caption = caption_text
#                     figure.caption_title = caption_title
#                     figure.rouge_l_score = rouge_score
#                     figure.possible_hallucination = not is_valid
#                     figure.diff = diff_text
#                 else:
#                     logger.warning(f"No caption found for {figure.figure_label}")
#                     figure.figure_caption = "Figure caption not found."
#                     figure.caption_title = ""
#                     figure.rouge_l_score = 0.0
#                     figure.possible_hallucination = True
#                     figure.diff = ""

#             return zip_structure

#         except Exception as e:
#             logger.error(f"Error in caption extraction: {str(e)}", exc_info=True)
#             return zip_structure
