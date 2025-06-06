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
        encoded_image: str,
        caption: str,
        prompt_config: Dict[str, Any],
        response_type: Optional[Type[T]] = None,
    ) -> Union[Dict[str, Any], T]:
        """
        Generate response from OpenAI using beta.chat.completions.parse.

        Args:
            encoded_image: Base64 encoded image
            caption: Figure caption
            prompt_config: Prompt configuration with system and user prompts
            response_type: Optional Pydantic model type for parsing response

        Returns:
            Response from OpenAI, either as parsed Pydantic model or raw dict
        """
        try:
            # Get prompts
            system_prompt = prompt_config.get("prompts", {}).get("system", "")
            user_prompt = (
                prompt_config.get("prompts", {})
                .get("user", "")
                .replace("$figure_caption", caption)
            )

            # Create messages
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

            # Get model parameters
            model_params = {
                "model": prompt_config.get("model", "gpt-4o"),
                "messages": messages,
                "temperature": prompt_config.get("temperature", 0.1),
                "top_p": prompt_config.get("top_p", 1.0),
                "frequency_penalty": prompt_config.get("frequency_penalty", 0.0),
                "presence_penalty": prompt_config.get("presence_penalty", 0.0),
                "max_tokens": prompt_config.get("max_tokens", 2048),
            }

            # Add response format
            if response_type:
                model_params["response_format"] = response_type
            elif prompt_config.get("json_mode", True):
                model_params["response_format"] = {"type": "json_object"}

            # Make API call
            response = self.client.beta.chat.completions.parse(**model_params)

            # Return appropriate type
            if response_type:
                return response.choices[0].message.content
            else:
                # Parse JSON content
                content = response.choices[0].message.content
                if isinstance(content, str):
                    return json.loads(content)
                return content

        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            raise
