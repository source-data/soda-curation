"""OpenAI model API with retry logic for QC pipeline."""

import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .openai_utils import call_openai_with_fallback, validate_model_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelAPI:
    """OpenAI model API with retry logic for QC pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelAPI.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.client = openai.OpenAI()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.OpenAIError, json.JSONDecodeError)),
        reraise=True,
    )
    def generate_response(
        self,
        prompt_config: Dict[str, Any],
        response_type: Optional[Type[T]] = None,
        encoded_image: Optional[str] = None,
        caption: Optional[str] = None,
        manuscript_text: Optional[str] = None,
        word_file_content: Optional[str] = None,
    ) -> Union[Dict[str, Any], T]:
        """
        Generate response from OpenAI using beta.chat.completions.parse.

        Args:
            prompt_config: Prompt configuration with system and user prompts
            response_type: Optional Pydantic model type for parsing response
            encoded_image: Base64 encoded image (for figure analysis)
            caption: Figure caption (for figure analysis)
            manuscript_text: Manuscript text (for document analysis)
            word_file_content: Word file content (for document analysis)

        Returns:
            Response from OpenAI, either as parsed Pydantic model or raw dict
        """
        # Get prompts
        system_prompt = prompt_config.get("prompts", {}).get("system", "")
        user_prompt = prompt_config.get("prompts", {}).get("user", "")

        # Determine the type of analysis and create appropriate messages
        if encoded_image is not None and caption is not None:
            # Figure analysis with image and caption
            user_prompt = user_prompt.replace("$figure_caption", caption)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ]
        elif manuscript_text is not None or word_file_content is not None:
            # Document/manuscript analysis with text
            text_content = word_file_content or manuscript_text or ""
            user_prompt = user_prompt.replace("$manuscript_text", text_content)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            raise ValueError(
                "Must provide either (encoded_image + caption) for figure analysis or "
                "(manuscript_text or word_file_content) for document analysis"
            )

        # Get model configuration
        model = prompt_config.get("model", "gpt-4o")

        # Validate model configuration
        validate_model_config(model, prompt_config)

        # Make API call with fallback support
        response = call_openai_with_fallback(
            client=self.client,
            model=model,
            messages=messages,
            response_format=response_type,
            temperature=prompt_config.get("temperature", 0.1),
            top_p=prompt_config.get("top_p", 1.0),
            frequency_penalty=prompt_config.get("frequency_penalty", 0.0),
            presence_penalty=prompt_config.get("presence_penalty", 0.0),
            max_tokens=prompt_config.get("max_tokens", 2048),
            json_mode=prompt_config.get("json_mode", True),
        )

        # Return appropriate type
        if response_type:
            return response.choices[0].message.content
        else:
            # Parse JSON content
            content = response.choices[0].message.content
            if isinstance(content, str):
                return json.loads(content)
            return content
