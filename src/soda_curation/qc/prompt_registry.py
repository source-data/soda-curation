# src/soda_curation/qc/prompt_registry.py
"""Langfuse-backed prompt registry for the QC pipeline.

Prompts and JSON schemas are stored in Langfuse (v3).  The registry fetches
them by the test name (converted to kebab-case) and caches results in-process.
No local file access to soda-mmQC is required.

Langfuse prompt structure expected per test:
  - prompt.prompt  : system prompt text (text prompt) OR list of role-messages
                     (chat prompt, where the 'system' role is used as the
                     system prompt and the optional 'user' role as the user
                     prompt template).
  - prompt.config  : dict that MAY contain:
      "schema"         : JSON Schema dict used to generate the Pydantic model
      "name"           : human-readable test name (overrides local config)
      "description"    : description of the test
      "checklist_type" : "fig-checklist" | "doc-checklist"
"""

import importlib.util
import json
import logging
import os
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Type, Union, cast

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


class PromptMetadata(NamedTuple):
    """Metadata for a prompt."""

    name: str
    description: str
    permalink: str
    version: str
    prompt_file: str  # Always "" for Langfuse-backed prompts; kept for compatibility


class ChecklistType(str, Enum):
    """Type of checklist."""

    DOC = "doc-checklist"
    FIG = "fig-checklist"


class PromptRegistry:
    """Registry backed by Langfuse for prompts, schemas and metadata.

    Langfuse credentials are read from environment variables:
      LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
    These are typically provided via a .env file loaded by python-dotenv.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._prompt_cache: Dict[str, Any] = {}
        self._model_cache: Dict[str, Type[BaseModel]] = {}

        # Build test_name -> checklist_type mapping from local config overrides
        self._test_mapping: Dict[str, str] = {}
        if "qc_check_metadata" in self.config:
            for level, level_tests in self.config["qc_check_metadata"].items():
                if isinstance(level_tests, dict):
                    for test_name, metadata in level_tests.items():
                        if isinstance(metadata, dict) and "checklist_type" in metadata:
                            self._test_mapping[test_name] = metadata["checklist_type"]

        # Langfuse client is initialised lazily to avoid import-time side-effects.
        # _langfuse_failed is set True on first failed init so we don't retry.
        self._langfuse = None
        self._langfuse_failed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_langfuse(self):
        """Lazily create and return the Langfuse client.

        Returns None (and sets _langfuse_failed) if initialisation fails so that
        callers can degrade gracefully without retrying on every call.
        """
        if self._langfuse is None and not self._langfuse_failed:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass

            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse()
            except Exception as exc:
                logger.debug("Langfuse client init failed (check env vars): %s", exc)
                self._langfuse_failed = True

        return self._langfuse  # None when unavailable

    def _to_langfuse_name(self, test_name: str) -> str:
        """Return the full Langfuse prompt name for a test.

        If the test config has a ``langfuse_name`` override, that value is used
        verbatim.  Otherwise the name is constructed as:
            checklists/{checklist_type}/{snake_to_kebab(test_name)}

        This matches the namespace used in the soda-mmQC Langfuse project.
        """
        local_cfg = self.get_test_config(test_name)
        if local_cfg.get("langfuse_name"):
            return local_cfg["langfuse_name"]

        checklist_type = self.get_checklist_type(test_name)
        kebab = test_name.replace("_", "-")
        return f"checklists/{checklist_type}/{kebab}"

    def _get_langfuse_prompt(self, test_name: str):
        """Fetch a prompt from Langfuse, with in-process caching.

        Raises RuntimeError if Langfuse is unavailable or the prompt is not found.
        Callers are expected to catch this and return fallback values.
        """
        if test_name not in self._prompt_cache:
            client = self._get_langfuse()
            if client is None:
                raise RuntimeError(
                    f"Langfuse is not available "
                    f"(check LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY env vars). "
                    f"Cannot fetch prompt for '{test_name}'."
                )
            langfuse_name = self._to_langfuse_name(test_name)
            logger.debug("Fetching Langfuse prompt: %s", langfuse_name)
            try:
                # Use explicit label="production" so the SDK caches under the
                # correct key.  Without an explicit label the SDK silently falls
                # back to 'latest' but still caches under the 'production' key,
                # causing background-refresh warnings for every non-promoted prompt.
                self._prompt_cache[test_name] = client.get_prompt(
                    langfuse_name, label="production"
                )
            except Exception:
                # Fallback: fetch with label="latest" (for prompts not yet
                # promoted to production).  The SDK now caches under the
                # 'latest' key, so background refresh works without warnings.
                try:
                    self._prompt_cache[test_name] = client.get_prompt(
                        langfuse_name, label="latest"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to fetch Langfuse prompt '{langfuse_name}' "
                        f"(tried both 'production' and 'latest' labels): {exc}"
                    ) from exc
        return self._prompt_cache[test_name]

    # ------------------------------------------------------------------
    # Config helpers (local qc_check_metadata)
    # ------------------------------------------------------------------

    def get_test_config(self, test_name: str) -> Dict[str, Any]:
        """Return local qc_check_metadata entry for a test, or {}."""
        if "qc_check_metadata" in self.config:
            for level, level_tests in self.config["qc_check_metadata"].items():
                if isinstance(level_tests, dict) and test_name in level_tests:
                    return level_tests[test_name] or {}
        return {}

    # ------------------------------------------------------------------
    # Public interface (same as previous PromptRegistry)
    # ------------------------------------------------------------------

    def get_checklist_type(self, test_name: str) -> str:
        """Return checklist type for a test.

        Priority: local config > Langfuse prompt config > default "fig-checklist".
        """
        local_cfg = self.get_test_config(test_name)
        if "checklist_type" in local_cfg:
            return local_cfg["checklist_type"]

        try:
            lf_cfg = self._get_langfuse_prompt(test_name).config or {}
            return lf_cfg.get("checklist_type", "fig-checklist")
        except Exception:
            return "fig-checklist"

    def get_prompt(self, test_name: str) -> str:
        """Return the system prompt text for a test from Langfuse.

        Supports both Langfuse text prompts (prompt is a plain string) and
        chat prompts (prompt is a list of role-message dicts).
        """
        lf_prompt = self._get_langfuse_prompt(test_name)
        content = lf_prompt.prompt

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # Chat prompt – use the 'system' role message
            for msg in content:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    return msg.get("content", "")
            # Fallback: join all content
            return "\n".join(
                msg.get("content", "") if isinstance(msg, dict) else str(msg)
                for msg in content
            )

        return str(content)

    def get_user_prompt(self, test_name: str) -> Optional[str]:
        """Return the user prompt template from a Langfuse chat prompt, if present."""
        try:
            lf_prompt = self._get_langfuse_prompt(test_name)
            content = lf_prompt.prompt
            if isinstance(content, list):
                for msg in content:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content")
        except Exception:
            pass
        return None

    def get_schema(self, test_name: str) -> Dict[str, Any]:
        """Return the JSON schema for a test from Langfuse prompt config.

        Expects prompt.config["schema"] to be a JSON Schema dict.
        Raises FileNotFoundError (for compatibility with existing callers) when
        no schema is present or Langfuse is unavailable.
        """
        try:
            lf_prompt = self._get_langfuse_prompt(test_name)
        except RuntimeError as exc:
            raise FileNotFoundError(
                f"Could not retrieve schema for '{test_name}' from Langfuse: {exc}"
            ) from exc

        lf_cfg = lf_prompt.config or {}

        # Primary location: config["schema"]
        if lf_cfg.get("schema"):
            return lf_cfg["schema"]

        # Actual Langfuse structure: config["output_schema"]["format"]["schema"]
        output_schema = lf_cfg.get("output_schema")
        if output_schema:
            fmt = output_schema.get("format", {})
            if fmt.get("schema"):
                return fmt["schema"]
            # Return the whole output_schema; generate_pydantic_model_from_schema
            # can unwrap the "format.schema" nesting.
            return output_schema

        raise FileNotFoundError(
            f"No schema found in Langfuse prompt config for test "
            f"'{test_name}' (prompt name: '{self._to_langfuse_name(test_name)}'). "
            f"Expected config['schema'] or config['output_schema']."
        )

    def get_prompt_metadata(self, test_name: str) -> PromptMetadata:
        """Return PromptMetadata fetched from Langfuse, with local config fallback."""
        try:
            lf_prompt = self._get_langfuse_prompt(test_name)
            lf_cfg = lf_prompt.config or {}
            langfuse_name = self._to_langfuse_name(test_name)
            base_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
            permalink = f"{base_url}/prompts/{langfuse_name}"

            local_cfg = self.get_test_config(test_name)
            return PromptMetadata(
                name=(
                    local_cfg.get("name")
                    or lf_cfg.get("name")
                    or test_name.replace("_", " ").title()
                ),
                description=lf_cfg.get("description", local_cfg.get("description", "")),
                permalink=permalink,
                version=str(lf_prompt.version),
                prompt_file="",
            )
        except Exception as exc:
            logger.warning(
                "Could not fetch Langfuse metadata for '%s': %s. Using local config.",
                test_name,
                exc,
            )
            local_cfg = self.get_test_config(test_name)
            return PromptMetadata(
                name=local_cfg.get("name", test_name.replace("_", " ").title()),
                description=local_cfg.get("description", ""),
                permalink="",
                version="latest",
                prompt_file="",
            )

    def determine_test_type_from_model(self, model_class) -> Optional[str]:
        """Determine test type by inspecting the generated Pydantic model structure.

        Returns 'panel', 'figure', 'document', or None.
        """
        try:
            if hasattr(model_class, "__fields__"):
                fields = model_class.__fields__
            elif hasattr(model_class, "model_fields"):
                fields = model_class.model_fields
            else:
                return None

            if "outputs" in fields:
                outputs_field = fields["outputs"]

                if hasattr(outputs_field, "type_"):
                    field_type = outputs_field.type_
                elif hasattr(outputs_field, "annotation"):
                    field_type = outputs_field.annotation
                else:
                    return None

                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    if hasattr(field_type, "__args__") and field_type.__args__:
                        item_type = field_type.__args__[0]

                        if hasattr(item_type, "__fields__"):
                            item_fields = item_type.__fields__
                        elif hasattr(item_type, "model_fields"):
                            item_fields = item_type.model_fields
                        else:
                            item_fields = {}

                        if "panel_label" in item_fields:
                            return "panel"
                        elif any(
                            kw in str(item_fields).lower()
                            for kw in ["section", "manuscript", "document"]
                        ):
                            return "document"
                        else:
                            return None
                else:
                    return "figure"

            field_names = list(fields.keys())
            if any(
                kw in " ".join(field_names).lower()
                for kw in ["section", "manuscript", "document"]
            ):
                return "document"

            return None

        except Exception:
            return None

    def generate_pydantic_model_from_schema(
        self, schema: Dict[str, Any], model_name: str
    ) -> Type[BaseModel]:
        """Generate a Pydantic model from a JSON schema via datamodel-code-generator."""
        try:
            from datamodel_code_generator import InputFileType, generate
        except ImportError:
            raise ImportError(
                "datamodel-code-generator is required for schema→model conversion. "
                "Install it with: poetry add datamodel-code-generator"
            )

        # Unwrap nested format if present
        if "format" in schema and "schema" in schema.get("format", {}):
            schema = schema["format"]["schema"]

        # Build a valid Python class name
        clean_name = (
            model_name.replace(".", "_").replace("-", "_").title().replace("_", "")
        )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", delete=False
        ) as output_file:
            try:
                output_path = Path(output_file.name)
                generate(
                    json.dumps(schema),
                    input_file_type=InputFileType.JsonSchema,
                    output=output_path,
                    class_name=clean_name,
                    field_include_all_keys=False,
                    use_standard_collections=True,
                    use_field_description=True,
                )

                code = output_path.read_text()

                module_name = (
                    f"soda_curation_generated_"
                    f"{model_name.replace('-', '_').replace('.', '_')}"
                )
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                if spec is None:
                    raise ImportError(f"Failed to create module spec for {module_name}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                exec(code, module.__dict__)  # noqa: S102

                if hasattr(module, clean_name):
                    return cast(Type[BaseModel], getattr(module, clean_name))

                # Fallback: return the first BaseModel subclass found
                for obj in module.__dict__.values():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, BaseModel)
                        and obj is not BaseModel
                    ):
                        return cast(Type[BaseModel], obj)

                raise ValueError(
                    f"Could not find Pydantic model in generated code for {model_name}"
                )
            finally:
                try:
                    os.unlink(output_file.name)
                except Exception:
                    pass

    def get_pydantic_model(self, test_name: str) -> Type[BaseModel]:
        """Return (or generate) the Pydantic response model for a test."""
        if test_name not in self._model_cache:
            try:
                schema = self.get_schema(test_name)
                self._model_cache[test_name] = self.generate_pydantic_model_from_schema(
                    schema, test_name
                )
            except (FileNotFoundError, Exception) as exc:
                logger.warning(
                    "Could not build Pydantic model for '%s': %s. Using fallback model.",
                    test_name,
                    exc,
                )
                self._model_cache[test_name] = create_model(
                    f"{test_name.title().replace('_', '')}Model",
                    outputs=(List[Dict[str, Any]], ...),
                )
        return self._model_cache[test_name]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def create_registry(config_path: Optional[str] = None) -> PromptRegistry:
    """Create a PromptRegistry with config loaded from disk."""
    import yaml

    if config_path is None:
        candidates = [
            Path("config.qc.yaml"),
            Path(__file__).parents[3] / "config.qc.yaml",
            Path(__file__).parents[4] / "config.qc.yaml",
        ]
        for path in candidates:
            if path.exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)
    else:
        config = {}

    return PromptRegistry(config=config)


registry = create_registry()
