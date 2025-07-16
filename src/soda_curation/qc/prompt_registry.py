# src/soda_curation/qc/prompt_registry.py
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Type, Union

import git
from pydantic import BaseModel, create_model


class PromptMetadata(NamedTuple):
    """Metadata for a prompt."""

    name: str
    description: str
    permalink: str
    version: str
    prompt_number: int


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
        self.config = config or {}

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
        for test_name, metadata in self.config.get("qc_test_metadata", {}).items():
            if "checklist_type" in metadata:
                self._test_mapping[test_name] = metadata["checklist_type"]

    def get_test_config(self, test_name: str) -> Dict[str, Any]:
        """Get configuration for a specific test."""
        return self.config.get("qc_test_metadata", {}).get(test_name, {})

    def get_prompt_version(self, test_name: str) -> int:
        """Get the prompt version for a test from config."""
        return self.get_test_config(test_name).get("prompt_version", 1)

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

    def get_prompt(self, test_name: str) -> str:
        """Get the prompt for a test based on config settings."""
        checklist_type = self.get_checklist_type(test_name)
        mmqc_test_name = self.get_mmqc_test_name(test_name)
        prompt_version = self.get_prompt_version(test_name)

        checklist_path = self.get_checklist_path(checklist_type)
        prompt_path = (
            checklist_path / mmqc_test_name / "prompts" / f"prompt.{prompt_version}.txt"
        )

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt {prompt_version} not found for test {test_name}"
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
        config = self.get_test_config(test_name)
        checklist_type = self.get_checklist_type(test_name)
        mmqc_test_name = self.get_mmqc_test_name(test_name)
        prompt_version = self.get_prompt_version(test_name)

        checklist_path = self.get_checklist_path(checklist_type)
        prompt_path = (
            checklist_path / mmqc_test_name / "prompts" / f"prompt.{prompt_version}.txt"
        )

        # Get version from git if available
        version = "latest"
        if self.has_git:
            try:
                relative_path = prompt_path.relative_to(self.mmqc_path)
                commits = list(
                    self.repo.iter_commits(paths=str(relative_path), max_count=1)
                )
                if commits:
                    version = commits[0].hexsha[:7]  # Short hash
            except Exception:
                pass

        return PromptMetadata(
            name=config.get("name", test_name.replace("_", " ").title()),
            description=config.get("description", ""),
            permalink=self.get_permalink(prompt_path),
            version=version,
            prompt_number=prompt_version,
        )

    def get_pydantic_model(self, test_name: str) -> Type[BaseModel]:
        """Get or create a Pydantic model from the JSON schema."""
        if test_name not in self._model_cache:
            schema = self.get_schema(test_name)
            self._model_cache[test_name] = self._create_model_from_schema(
                schema, test_name
            )

        return self._model_cache[test_name]

    def _create_model_from_schema(
        self, schema: Dict[str, Any], model_name: str
    ) -> Type[BaseModel]:
        """Create a Pydantic model from a JSON schema."""
        # Extract the schema from the format wrapper if needed
        if "format" in schema and "schema" in schema["format"]:
            schema = schema["format"]["schema"]

        # Create a model name in PascalCase
        clean_name = model_name.replace("-", "_").title().replace("_", "")

        # For now, we'll just create a simple wrapper model
        return create_model(
            clean_name,
            __base__=BaseModel,
            outputs=(List[Dict[str, Any]], ...),
        )


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
