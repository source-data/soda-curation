"""Model API orchestrator with provider abstraction for the QC pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

import openai
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..pipeline.ai_observability import summarize_messages, summarize_text
from ..pipeline.anthropic_utils import is_retryable_anthropic_error
from ..pipeline.cost_tracking import update_token_usage
from ..pipeline.manuscript_structure.manuscript_structure import TokenUsage
from ..pipeline.openai_utils import is_retryable_openai_error
from .providers import build_qc_provider
from .providers.base import BaseQCProvider, QCProviderRequest, QCProviderResponse

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_qc_retryable_exception(exc: Exception) -> bool:
    """Retry only known transient provider errors and JSON parse hiccups."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, openai.OpenAIError):
        return is_retryable_openai_error(exc)
    if is_retryable_anthropic_error(exc):
        return True

    class_name = exc.__class__.__name__
    retryable_names = {
        "APIError",
        "ServiceUnavailable",
        "TooManyRequests",
        "ServerError",
    }
    if class_name in retryable_names:
        return True

    message = str(exc).lower()
    transient_markers = [
        "rate limit",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "503",
        "504",
    ]
    return any(marker in message for marker in transient_markers)


class ModelAPI:
    """Provider-agnostic QC model API with retry and normalized outputs."""

    def __init__(
        self,
        config: Dict[str, Any],
        provider: Optional[BaseQCProvider] = None,
        provider_clients: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.ai_provider = config.get("ai_provider", "openai").lower()
        self.provider = provider or build_qc_provider(
            provider_name=self.ai_provider,
            clients=provider_clients,
        )
        self.token_usage = TokenUsage()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(_is_qc_retryable_exception),
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
        expected_panels: Optional[list] = None,
        operation: str = "qc.generate_response",
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], T]:
        """Generate a response for QC figure/manuscript prompts."""
        system_prompt = prompt_config.get("prompts", {}).get("system", "")
        user_prompt = prompt_config.get("prompts", {}).get("user", "")

        if expected_panels:
            panels_constraint = (
                f"**CRITICAL PANEL LABEL CONSTRAINT**:\n"
                f"The output panel labels for this task MUST be exactly equal to "
                f"those defined for this figure. The allowed panel labels are: {expected_panels}\n"
                f"Rules:\n"
                f"- You MUST ONLY use panel labels from the list above.\n"
                f"- Do NOT invent new panel labels or subdivisions.\n"
                f"- Do NOT use sub-panel labels (e.g., 'A-a', 'A-b', 'A-l').\n"
                f"- Do NOT use descriptive labels (e.g., 'Rice cell', 'Figure 8').\n"
                f"- Do NOT add modifiers to labels (e.g., 'C (plot)', 'C (right)').\n"
                f"- Each `panel_label` in your response MUST be exactly one of: {expected_panels}\n"
            )
            system_prompt = panels_constraint + "\n" + system_prompt
            user_prompt += "\n\n" + panels_constraint

        request_context = context or {}
        if encoded_image is not None and caption is not None:
            if not encoded_image.strip():
                logger.warning(
                    "QC image payload is empty",
                    extra={
                        "operation": operation,
                        "severity": "recoverable",
                        "reason": "empty_image_payload",
                        "context": request_context,
                    },
                )
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
            text_content = word_file_content or manuscript_text or ""
            if text_content.lower().startswith(
                "no word file"
            ) or text_content.lower().startswith("error"):
                logger.warning(
                    "QC manuscript payload appears to be placeholder/error text",
                    extra={
                        "operation": operation,
                        "severity": "recoverable",
                        "reason": "placeholder_manuscript_text",
                        "manuscript_summary": summarize_text(text_content),
                        "context": request_context,
                    },
                )
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

        model = prompt_config.get(
            "model", _default_model_for_provider(self.ai_provider)
        )
        model_config = prompt_config.get("model_config", {}) or {}
        agentic_enabled = bool(prompt_config.get("agentic", False))

        logger.info(
            "QC model request prepared",
            extra={
                "operation": operation,
                "provider": self.ai_provider,
                "model": model,
                "agentic_enabled": agentic_enabled,
                "message_summary": summarize_messages(messages),
                "context": request_context,
            },
        )

        provider_request = QCProviderRequest(
            model=model,
            messages=messages,
            prompt_config=prompt_config,
            response_type=response_type,
            operation=operation,
            context=request_context,
            agentic_enabled=agentic_enabled,
            model_config=model_config,
        )
        provider_response = self.provider.generate(provider_request)
        self._update_usage(provider_response)
        return self._format_response(
            provider_response=provider_response,
            response_type=response_type,
            operation=operation,
            context=request_context,
        )

    def _update_usage(self, provider_response: QCProviderResponse) -> None:
        response_dict = {"usage": provider_response.usage}
        update_token_usage(self.token_usage, response_dict, provider_response.model)

    def _format_response(
        self,
        provider_response: QCProviderResponse,
        response_type: Optional[Type[T]],
        operation: str,
        context: Dict[str, Any],
    ) -> Union[Dict[str, Any], T]:
        if response_type:
            parsed = provider_response.parsed
            if parsed is not None:
                if hasattr(parsed, "model_dump"):
                    return json.dumps(parsed.model_dump(mode="json"))
                return json.dumps(parsed)

            if not provider_response.content:
                logger.warning(
                    "Structured QC response had empty content and no parsed object",
                    extra={
                        "operation": operation,
                        "provider": self.ai_provider,
                        "context": context,
                    },
                )
                return "{}"
            return provider_response.content

        content = provider_response.content
        if isinstance(content, str):
            return json.loads(content)
        return content


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
        "gemini": "gemini-2.5-flash",
    }
    return defaults.get(provider.lower(), "gpt-4o")
