# src/soda_curation/qc/prompt_registry.py
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Type, Union, cast

import git
from pydantic import BaseModel, create_model


class PromptMetadata(NamedTuple):
    """Metadata for a prompt."""

    name: str
    description: str
    permalink: str
    version: str
    prompt_file: str  # Added to store the actual prompt filename


class BenchmarkMetadata(NamedTuple):
    """Metadata from benchmark.json file."""

    name: str
    description: str
    examples: List[str]


class ChecklistType(str, Enum):
    """Type of checklist."""

    DOC = "doc-checklist"
    FIG = "fig-checklist"


class PromptRegistry:
    """Registry for accessing prompts and schemas from soda-mmQC."""

    def __init__(
        self, mmqc_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        # Use provided path, env variable, or default to neighboring directory
        self.mmqc_path = Path(
            mmqc_path
            or os.environ.get("SODA_MMQC_PATH")
            or Path(__file__).parents[4] / "soda-mmQC"
        )
        self.base_data_path = self.mmqc_path / "soda_mmqc" / "data" / "checklist"
        self._model_cache: Dict[str, Type[BaseModel]] = {}
        self._benchmark_cache: Dict[
            str, BenchmarkMetadata
        ] = {}  # Cache for benchmark data
        self.config = config or {}
        self._module_cache = {}

        # Initialize git repo if it's a git repository
        try:
            self.repo = git.Repo(self.mmqc_path)
            self.has_git = True
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            self.repo = None
            self.has_git = False

        # GitHub repo info (can be configurable)
        self.github_owner = "source-data"
        self.github_repo = "soda-mmQC"

        # Build test_name to checklist_type mapping
        self._test_mapping = {}
        if "qc_check_metadata" in self.config:
            for level, level_tests in self.config["qc_check_metadata"].items():
                if isinstance(level_tests, dict):
                    for test_name, metadata in level_tests.items():
                        if isinstance(metadata, dict) and "checklist_type" in metadata:
                            self._test_mapping[test_name] = metadata["checklist_type"]

    def get_test_config(self, test_name: str) -> Dict[str, Any]:
        """Get configuration for a specific test."""
        if "qc_check_metadata" in self.config:
            # Search in each level (panel, figure, document)
            for level, level_tests in self.config["qc_check_metadata"].items():
                if isinstance(level_tests, dict) and test_name in level_tests:
                    return level_tests[test_name]
        return {}

    def get_prompt_version(self, test_name: str) -> int:
        """Get the prompt version for a test from config."""
        return self.get_test_config(test_name).get("prompt_version", 1)

    def get_prompt_file(self, test_name: str) -> str:
        """Get the prompt filename for a test from config. Falls back to prompt.{version}.txt if not specified."""
        test_config = self.get_test_config(test_name)

        # First check if a specific prompt_file is specified
        if "prompt_file" in test_config:
            return test_config["prompt_file"]

        # Fall back to the old version-based naming
        version = test_config.get("prompt_version", 1)
        return f"prompt.{version}.txt"

    def get_checklist_type(self, test_name: str) -> str:
        """Get the checklist type for a test."""
        return self.get_test_config(test_name).get("checklist_type", "fig-checklist")

    def get_checklist_path(self, checklist_type: Union[ChecklistType, str]) -> Path:
        """Get the path for a specific checklist type."""
        if isinstance(checklist_type, str):
            checklist_type = ChecklistType(checklist_type)
        return self.base_data_path / checklist_type.value

    def list_tests(self, checklist_type: Union[ChecklistType, str]) -> List[str]:
        """List all available QC tests for a checklist type."""
        checklist_path = self.get_checklist_path(checklist_type)
        return [d.name for d in checklist_path.iterdir() if d.is_dir()]

    def get_mmqc_test_name(self, test_name: str) -> str:
        """Convert snake_case to kebab-case for soda-mmQC directory names."""
        return test_name.replace("_", "-")

    def get_benchmark_metadata(self, test_name: str) -> Optional[BenchmarkMetadata]:
        """Get benchmark metadata for a test from benchmark.json file."""
        if test_name in self._benchmark_cache:
            return self._benchmark_cache[test_name]

        checklist_type = self.get_checklist_type(test_name)
        mmqc_test_name = self.get_mmqc_test_name(test_name)

        checklist_path = self.get_checklist_path(checklist_type)
        benchmark_path = checklist_path / mmqc_test_name / "benchmark.json"

        if not benchmark_path.exists():
            self._benchmark_cache[test_name] = None
            return None

        try:
            benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
            metadata = BenchmarkMetadata(
                name=benchmark_data.get("name", test_name),
                description=benchmark_data.get("description", ""),
                examples=benchmark_data.get("examples", []),
            )
            self._benchmark_cache[test_name] = metadata
            return metadata
        except (json.JSONDecodeError, KeyError):
            self._benchmark_cache[test_name] = None
            return None

    def get_prompt(self, test_name: str) -> str:
        """Get the prompt for a test based on config settings."""
        checklist_type = self.get_checklist_type(test_name)
        mmqc_test_name = self.get_mmqc_test_name(test_name)
        prompt_file = self.get_prompt_file(test_name)

        checklist_path = self.get_checklist_path(checklist_type)
        prompt_path = checklist_path / mmqc_test_name / "prompts" / prompt_file

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file '{prompt_file}' not found for test {test_name} at {prompt_path}"
            )

        return prompt_path.read_text(encoding="utf-8")

    def list_prompts(self, test_name: str) -> List[int]:
        """List available prompt numbers for a test."""
        checklist_type = self.get_checklist_type(test_name)
        mmqc_test_name = self.get_mmqc_test_name(test_name)

        checklist_path = self.get_checklist_path(checklist_type)
        test_dir = checklist_path / mmqc_test_name / "prompts"

        if not test_dir.exists():
            raise ValueError(f"Test {test_name} not found in {checklist_type}")

        prompt_files = list(test_dir.glob("prompt.*.txt"))
        return sorted([int(f.stem.split(".")[1]) for f in prompt_files])

    def get_schema(self, test_name: str) -> Dict[str, Any]:
        """Get the JSON schema for a test."""
        checklist_type = self.get_checklist_type(test_name)
        mmqc_test_name = self.get_mmqc_test_name(test_name)

        checklist_path = self.get_checklist_path(checklist_type)
        schema_path = checklist_path / mmqc_test_name / "schema.json"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found for test {test_name}")

        return json.loads(schema_path.read_text())

    def get_permalink(self, file_path: Path) -> str:
        """Generate a GitHub permalink for a file."""
        if not self.has_git:
            # Fallback if git not available
            relative_path = file_path.relative_to(self.mmqc_path)
            return f"https://github.com/{self.github_owner}/{self.github_repo}/blob/main/{relative_path}"

        # Get the latest commit hash for the file
        try:
            relative_path = file_path.relative_to(self.mmqc_path)
            commits = list(
                self.repo.iter_commits(paths=str(relative_path), max_count=1)
            )
            if commits:
                commit_hash = commits[0].hexsha
                return f"https://github.com/{self.github_owner}/{self.github_repo}/blob/{commit_hash}/{relative_path}"
        except Exception:
            pass

        # Fallback to branch name if commit lookup fails
        relative_path = file_path.relative_to(self.mmqc_path)
        return f"https://github.com/{self.github_owner}/{self.github_repo}/blob/main/{relative_path}"

    def get_prompt_metadata(self, test_name: str) -> PromptMetadata:
        """Get metadata for a test's prompt, including permalink."""
        # Check if test exists in config
        test_config = self.get_test_config(test_name)
        if not test_config:
            # If not found in config, create minimal metadata
            return PromptMetadata(
                name=test_name.replace("_", " ").title(),
                description="",
                permalink="",
                version="latest",
                prompt_file="prompt.1.txt",  # Default to version 1
            )

        checklist_type = self.get_checklist_type(test_name)
        mmqc_test_name = self.get_mmqc_test_name(test_name)
        prompt_file = self.get_prompt_file(test_name)

        checklist_path = self.get_checklist_path(checklist_type)
        prompt_path = checklist_path / mmqc_test_name / "prompts" / prompt_file

        # Get version from git if available
        version = "latest"
        if self.has_git and prompt_path.exists():
            try:
                relative_path = prompt_path.relative_to(self.mmqc_path)
                commits = list(
                    self.repo.iter_commits(paths=str(relative_path), max_count=1)
                )
                if commits:
                    version = commits[0].hexsha[:7]  # Short hash
            except Exception:
                pass

        # Try to get benchmark metadata for more complete information
        benchmark_metadata = self.get_benchmark_metadata(test_name)

        # Use benchmark description if available, otherwise fall back to config
        description = test_config.get("description", "")
        if benchmark_metadata and benchmark_metadata.description:
            description = benchmark_metadata.description

        # Note: example_class logic removed - now using schema-based type determination

        return PromptMetadata(
            name=test_config.get("name", test_name.replace("_", " ").title()),
            description=description,
            permalink=self.get_permalink(prompt_path) if prompt_path.exists() else "",
            version=version,
            prompt_file=prompt_file,
        )

    def determine_test_type_from_model(self, model_class) -> Optional[str]:
        """Determine test type by inspecting the Pydantic model structure.

        Args:
            model_class: The generated Pydantic model class

        Returns:
            'panel' for panel-level tests (list of panels)
            'figure' for figure-level tests (single result)
            'document' for document/manuscript tests
            None if type cannot be determined
        """
        try:
            # Get the model fields
            if hasattr(model_class, "__fields__"):
                fields = model_class.__fields__
            elif hasattr(model_class, "model_fields"):
                fields = model_class.model_fields
            else:
                return None

            # Check if there's an 'outputs' field
            if "outputs" in fields:
                outputs_field = fields["outputs"]

                # Get the type information
                if hasattr(outputs_field, "type_"):
                    field_type = outputs_field.type_
                elif hasattr(outputs_field, "annotation"):
                    field_type = outputs_field.annotation
                else:
                    return None

                # Check if it's a List type (indicates panel-level)
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    # Further inspect the list items to distinguish panel vs document
                    if hasattr(field_type, "__args__") and field_type.__args__:
                        item_type = field_type.__args__[0]

                        # Check if the list items have 'panel_label' field (panel-level)
                        if hasattr(item_type, "__fields__"):
                            item_fields = item_type.__fields__
                        elif hasattr(item_type, "model_fields"):
                            item_fields = item_type.model_fields
                        else:
                            item_fields = {}

                        if "panel_label" in item_fields:
                            return "panel"
                        elif any(
                            keyword in str(item_fields).lower()
                            for keyword in ["section", "manuscript", "document"]
                        ):
                            return "document"
                        else:
                            return "panel"  # Default for list types

                # Single object output (figure-level)
                else:
                    return "figure"

            # Check for document-specific indicators in field names
            field_names = list(fields.keys())
            if any(
                keyword in " ".join(field_names).lower()
                for keyword in ["section", "manuscript", "document"]
            ):
                return "document"

            return None

        except Exception:
            # Error in determining test type from model, falling back to other methods
            return None

    def generate_pydantic_model_from_schema(
        self, schema: Dict[str, Any], model_name: str
    ) -> Type[BaseModel]:
        """Generate a Pydantic model from a JSON schema using datamodel-code-generator."""
        try:
            from datamodel_code_generator import InputFileType, generate
            from datamodel_code_generator.model.pydantic import (
                BaseModel as CodegenBaseModel,
            )
        except ImportError:
            raise ImportError(
                "datamodel-code-generator is required for schema to model conversion. "
                "Install it with: poetry add datamodel-code-generator"
            )

        # Clean up schema format if needed
        if "format" in schema and "schema" in schema["format"]:
            schema = schema["format"]["schema"]

        # Convert test_name to a valid Python class name
        clean_name = model_name.replace("-", "_").title().replace("_", "")

        # Create temporary files for output
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", delete=False
        ) as output_file:
            try:
                # Generate Python code from schema string
                schema_str = json.dumps(schema)
                output_path = Path(output_file.name)

                # Generate Python code
                generate(
                    schema_str,
                    input_file_type=InputFileType.JsonSchema,
                    output=output_path,
                    class_name=clean_name,
                    field_include_all_keys=False,
                    use_standard_collections=True,
                    use_field_description=True,
                )

                # Read generated code
                with open(output_path, "r") as f:
                    code = f.read()

                # Create a module from the generated code
                module_name = f"soda_curation_generated_{model_name.replace('-', '_')}"
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                if spec is None:
                    raise ImportError(f"Failed to create module spec for {module_name}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                exec(code, module.__dict__)

                # Extract the main model class
                if hasattr(module, clean_name):
                    model_class = getattr(module, clean_name)
                    return cast(Type[BaseModel], model_class)
                else:
                    # If the main class wasn't found, look for other classes
                    for name, obj in module.__dict__.items():
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, BaseModel)
                            and obj != BaseModel
                        ):
                            return cast(Type[BaseModel], obj)

                raise ValueError(
                    f"Could not find Pydantic model in generated code for {model_name}"
                )

            finally:
                # Clean up temporary file
                try:
                    os.unlink(output_file.name)
                except Exception:
                    pass

    def get_pydantic_model(self, test_name: str) -> Type[BaseModel]:
        """Get or create a Pydantic model from the JSON schema."""
        if test_name not in self._model_cache:
            try:
                schema = self.get_schema(test_name)
                self._model_cache[test_name] = self.generate_pydantic_model_from_schema(
                    schema, test_name
                )
            except FileNotFoundError:
                # If schema not found, create a simple default model
                self._model_cache[test_name] = create_model(
                    f"{test_name.title().replace('_', '')}Model",
                    outputs=(List[Dict[str, Any]], ...),
                )

        return self._model_cache[test_name]


# Function to create the registry with the current config
def create_registry(config_path: Optional[str] = None) -> PromptRegistry:
    """Create a PromptRegistry with the current configuration."""
    import yaml

    if config_path is None:
        # Default config paths to try
        config_paths = [
            Path("config.qc.yaml"),
            Path(__file__).parents[3] / "config.qc.yaml",
            Path(__file__).parents[4] / "config.qc.yaml",
        ]

        for path in config_paths:
            if path.exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    return PromptRegistry(config=config)


# Create a singleton instance
registry = create_registry()
