import os
import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

from pydantic import BeforeValidator, ValidationError
from typing_extensions import Annotated

from src.soda_curation.pipeline.assign_panel_source.assign_panel_source_base import (
    AsignedFiles,
    AsignedFilesList,
    PanelSourceAssigner,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
)


# Add these helper functions at module level
def create_file_validator(valid_files: List[str]) -> Annotated[str, Any]:
    """Create a validator for file paths based on the list of valid files."""

    def validate_file_path(file_path: str) -> str:
        if file_path not in valid_files:
            raise ValueError(f"File path '{file_path}' not in valid files list")
        return file_path

    return Annotated[str, BeforeValidator(validate_file_path)]


class TestPanelSourceAssigner(unittest.TestCase):
    class ConcretePanelSourceAssigner(PanelSourceAssigner):
        def _validate_config(self):
            pass

        def call_ai_service(
            self, prompt: str, allowed_files: List[str]
        ) -> AsignedFilesList:
            # Mock response with validation context
            return AsignedFilesList(
                assigned_files=[
                    AsignedFiles.model_validate(
                        {
                            "panel_label": "A",
                            "panel_sd_files": [
                                "suppl_data/figure_1.zip:A/A_1.csv",
                                "suppl_data/figure_1.zip:A/A_2.csv",
                                "suppl_data/figure_1.zip:A/A_3.csv",
                            ],
                        },
                        context={"allowed_files": allowed_files},
                    ),
                    AsignedFiles.model_validate(
                        {
                            "panel_label": "B",
                            "panel_sd_files": [
                                "suppl_data/figure_1.zip:B/B_1.csv",
                                "suppl_data/figure_1.zip:B/B_2.csv",
                                "suppl_data/figure_1.zip:B/B_3.csv",
                            ],
                        },
                        context={"allowed_files": allowed_files},
                    ),
                ],
                not_assigned_files=["suppl_data/data_table.csv"],
            )

    def setUp(self):
        # Use the concrete subclass for testing
        self.config = {
            "assign_panel_source": {
                "prompts": {
                    "user": "Assign the following files to the panels: {panel_labels} with files: {file_list}"
                },
                "openai": {
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "top_p": 1.0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
                "cost": {},
            },
            "extraction_dir": "/mock/extraction/dir",  # Mock extraction directory
        }
        self.prompt_handler = MagicMock()  # Mock the PromptHandler
        self.assigner = self.ConcretePanelSourceAssigner(
            self.config, self.prompt_handler
        )

    @patch("zipfile.ZipFile")
    def test_get_zip_contents(self, mock_zipfile):
        # Mock the zipfile and its contents
        mock_zip = MagicMock()
        mock_zip.__enter__.return_value = mock_zip
        mock_zip.infolist.return_value = [
            MagicMock(filename="zip_file/A/file1.csv"),
            MagicMock(filename="zip_file/A/file2.xlsx"),
            MagicMock(filename="zip_file/A/file3.txt"),
            MagicMock(filename="zip_file/B/file4.dat"),
            MagicMock(filename="zip_file/B/file5.csv"),
            MagicMock(filename="zip_file/B/file6.xlsx"),
            MagicMock(filename="zip_file/B/file7.txt"),
        ]
        mock_zipfile.return_value = mock_zip

        # Define the input and expected output
        sd_files = [
            os.path.join(self.config["extraction_dir"], "path/to/zip_file.zip"),
            os.path.join(self.config["extraction_dir"], "path/to/data_file.dat"),
        ]
        expected_output = [
            "path/to/zip_file.zip:zip_file/A/file1.csv",
            "path/to/zip_file.zip:zip_file/A/file2.xlsx",
            "path/to/zip_file.zip:zip_file/A/file3.txt",
            "path/to/zip_file.zip:zip_file/B/file4.dat",
            "path/to/zip_file.zip:zip_file/B/file5.csv",
            "path/to/zip_file.zip:zip_file/B/file6.xlsx",
            "path/to/zip_file.zip:zip_file/B/file7.txt",
            "path/to/data_file.dat",
        ]

        # Call the method and assert the output
        output = self.assigner._get_zip_contents(sd_files)
        self.assertEqual(output, expected_output)

    @patch("zipfile.ZipFile")
    def test_assign_to_figure(self, mock_zipfile):
        # Mock the zipfile and its contents
        mock_zip = MagicMock()
        mock_zip.__enter__.return_value = mock_zip
        mock_zip.infolist.return_value = [
            MagicMock(filename="A/A_1.csv"),
            MagicMock(filename="A/A_2.csv"),
            MagicMock(filename="A/A_3.csv"),
            MagicMock(filename="B/B_1.csv"),
            MagicMock(filename="B/B_2.csv"),
            MagicMock(filename="B/B_3.csv"),
        ]
        mock_zipfile.return_value = mock_zip

        figure = Figure(
            figure_label="Figure 1",
            img_files=[],  # Provide an empty list or appropriate mock data
            panels=[
                Panel(panel_label="A", panel_caption=""),
                Panel(panel_label="B", panel_caption=""),
            ],
            sd_files=["suppl_data/figure_1.zip", "suppl_data/data_table.csv"],
        )

        self.assigner._assign_to_figure(figure)
        # Check that panels are updated correctly
        self.assertEqual(
            figure.panels[0].sd_files,
            [
                "suppl_data/figure_1.zip:A/A_1.csv",
                "suppl_data/figure_1.zip:A/A_2.csv",
                "suppl_data/figure_1.zip:A/A_3.csv",
            ],
        )
        self.assertEqual(
            figure.panels[1].sd_files,
            [
                "suppl_data/figure_1.zip:B/B_1.csv",
                "suppl_data/figure_1.zip:B/B_2.csv",
                "suppl_data/figure_1.zip:B/B_3.csv",
            ],
        )
        # Check that unassigned files are added to figure.sd_files
        self.assertCountEqual(
            figure.sd_files, ["suppl_data/data_table.csv", "suppl_data/figure_1.zip"]
        )

    def test_parse_assigned_files_to_panels(self):
        # Example AI response with validation context
        valid_files = [
            "file1.csv",
            "file2.xlsx",
            "file3.txt",
            "file4.dat",
            "file5.csv",
            "file6.xlsx",
            "file7.txt",
        ]

        ai_response = AsignedFilesList.model_validate(
            {
                "assigned_files": [
                    {
                        "panel_label": "A",
                        "panel_sd_files": ["file1.csv", "file2.xlsx", "file3.txt"],
                    },
                    {
                        "panel_label": "B",
                        "panel_sd_files": [
                            "file4.dat",
                            "file5.csv",
                            "file6.xlsx",
                            "file7.txt",
                        ],
                    },
                ],
                "not_assigned_files": ["data_file.dat"],
            },
            context={"allowed_files": valid_files},
        )

        # Expected output
        expected_panels = [
            Panel(
                panel_label="A",
                panel_caption="",
                sd_files=["file1.csv", "file2.xlsx", "file3.txt"],
            ),
            Panel(
                panel_label="B",
                panel_caption="",
                sd_files=["file4.dat", "file5.csv", "file6.xlsx", "file7.txt"],
            ),
        ]

        # Parse the AI response
        panels = self.assigner.parse_assigned_files_to_panels(ai_response)

        # Assert the parsed panels match the expected panels
        self.assertEqual(panels, expected_panels)

    def test_assign_to_figure_no_source_data(self):
        """Test that figures with no source data files are handled correctly."""
        # Override the call_ai_service with a MagicMock for this test
        self.assigner.call_ai_service = MagicMock()

        # Create a figure with no source data files
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(panel_label="A", panel_caption=""),
                Panel(panel_label="B", panel_caption=""),
            ],
            sd_files=[],  # Empty source data files
            img_files=[],
        )

        # Process the figure
        self.assigner._assign_to_figure(figure)

        # Verify that no assignments were made
        self.assertFalse(any(panel.sd_files for panel in figure.panels))
        self.assertEqual(figure.sd_files, [])
        # Verify AI service was not called
        self.assigner.call_ai_service.assert_not_called()

    @patch("zipfile.ZipFile")
    def test_assign_to_figure_with_source_data(self, mock_zipfile):
        """Test that figures with source data files proceed to AI service."""
        # Mock the zipfile and its contents
        mock_zip = MagicMock()
        mock_zip.__enter__.return_value = mock_zip
        mock_zip.infolist.return_value = [
            MagicMock(filename="file1.csv"),
            MagicMock(filename="file2.csv"),
        ]
        mock_zipfile.return_value = mock_zip

        # Override the call_ai_service with a MagicMock for this test
        self.assigner.call_ai_service = MagicMock()

        # Create a figure with source data files
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(panel_label="A", panel_caption=""),
                Panel(panel_label="B", panel_caption=""),
            ],
            sd_files=["source_data1.zip", "source_data2.zip"],
            img_files=[],
        )

        # Mock the AI service response
        mock_response = AsignedFilesList(
            assigned_files=[
                AsignedFiles.model_validate(
                    {"panel_label": "A", "panel_sd_files": ["file1.csv"]},
                    context={"allowed_files": ["file1.csv", "file2.csv"]},
                ),
                AsignedFiles.model_validate(
                    {"panel_label": "B", "panel_sd_files": ["file2.csv"]},
                    context={"allowed_files": ["file1.csv", "file2.csv"]},
                ),
            ],
            not_assigned_files=[],
        )
        self.assigner.call_ai_service.return_value = mock_response

        # Process the figure
        self.assigner._assign_to_figure(figure)

        # Verify AI service was called
        self.assigner.call_ai_service.assert_called_once()
        # Verify assignments were made
        self.assertEqual(figure.panels[0].sd_files, ["file1.csv"])
        self.assertEqual(figure.panels[1].sd_files, ["file2.csv"])
        self.assertCountEqual(figure.sd_files, ["source_data1.zip", "source_data2.zip"])

    @patch("zipfile.ZipFile")
    def test_get_zip_contents_filters_macos_files(self, mock_zipfile):
        """Test that __MACOSX and .DS_Store files are filtered out."""
        # Mock the zipfile and its contents
        mock_zip = MagicMock()
        mock_zip.__enter__.return_value = mock_zip
        mock_zip.infolist.return_value = [
            MagicMock(filename="__MACOSX/folder/._file1.csv"),
            MagicMock(filename="folder/.DS_Store"),
            MagicMock(filename="folder/file1.csv"),
            MagicMock(filename="folder/file2.xlsx"),
            MagicMock(filename="__MACOSX/folder/._file2.xlsx"),
            MagicMock(filename=".DS_Store"),
        ]
        mock_zipfile.return_value = mock_zip

        # Define the input
        sd_files = [
            os.path.join(self.config["extraction_dir"], "path/to/data.zip"),
            os.path.join(self.config["extraction_dir"], "path/to/file.dat"),
        ]

        # Define expected output (only real data files, no macOS files)
        expected_output = [
            "path/to/data.zip:folder/file1.csv",
            "path/to/data.zip:folder/file2.xlsx",
            "path/to/file.dat",
        ]

        # Call the method and assert the output
        output = self.assigner._get_zip_contents(sd_files)
        self.assertEqual(sorted(output), sorted(expected_output))


class TestPanelSourceAssignerValidation(unittest.TestCase):
    """Test validation features of PanelSourceAssigner."""

    def setUp(self):
        self.config = {
            "assign_panel_source": {
                "prompts": {
                    "user": "Assign the following files to the panels: {panel_labels} with files: {file_list}"
                },
                "openai": {
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "top_p": 1.0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
                "cost": {},
            },
            "extraction_dir": "/mock/extraction/dir",
        }
        self.prompt_handler = MagicMock()

    def test_hallucinated_files_validation(self):
        """Test that hallucinated file paths are rejected."""
        # Define valid files
        valid_files = [
            "suppl_data/figure_1.zip:A/real_file1.csv",
            "suppl_data/figure_1.zip:A/real_file2.xlsx",
            "suppl_data/figure_1.zip:B/real_file3.csv",
        ]

        ValidatedFile = create_file_validator(valid_files)

        class ValidatedAsignedFiles(AsignedFiles):
            panel_sd_files: List[ValidatedFile]

        # Test multiple scenarios
        test_cases = [
            # Valid cases
            {
                "panel_label": "A",
                "panel_sd_files": ["suppl_data/figure_1.zip:A/real_file1.csv"],
                "should_pass": True,
            },
            # Multiple valid files
            {
                "panel_label": "A",
                "panel_sd_files": [
                    "suppl_data/figure_1.zip:A/real_file1.csv",
                    "suppl_data/figure_1.zip:A/real_file2.xlsx",
                ],
                "should_pass": True,
            },
            # Invalid file
            {
                "panel_label": "A",
                "panel_sd_files": ["suppl_data/figure_1.zip:A/hallucinated_file.csv"],
                "should_pass": False,
            },
            # Mix of valid and invalid files
            {
                "panel_label": "B",
                "panel_sd_files": [
                    "suppl_data/figure_1.zip:B/real_file3.csv",
                    "suppl_data/figure_1.zip:B/hallucinated_file.xlsx",
                ],
                "should_pass": False,
            },
        ]

        for case in test_cases:
            if case["should_pass"]:
                try:
                    valid_data = AsignedFiles.model_validate(
                        {
                            "panel_label": case["panel_label"],
                            "panel_sd_files": case["panel_sd_files"],
                        },
                        context={"allowed_files": valid_files},
                    )
                    self.assertIsNotNone(valid_data)
                except ValidationError:
                    self.fail(
                        f"Validation failed for valid files: {case['panel_sd_files']}"
                    )
            else:
                with self.assertRaises(ValidationError):
                    AsignedFiles.model_validate(
                        {
                            "panel_label": case["panel_label"],
                            "panel_sd_files": case["panel_sd_files"],
                        },
                        context={"allowed_files": valid_files},
                    )

    def test_unicode_path_handling(self):
        """Test handling of Unicode characters in file paths."""
        # Setup the test environment
        self.config[
            "extraction_dir"
        ] = "/mock/extraction/dir"  # Make sure this matches the zip path structure

        test_paths = [
            "Figure 1/1A/Figure 1A IFN-ß.xlsx",  # UTF-8
            "Figure 1/1A/Figure 1A IFN-ª┬.xlsx",  # Windows-1252
            "Figure 1/1B/µg-analysis.csv",  # UTF-8 micro symbol
            "Figure 1/1C/°C-measurements.xlsx",  # UTF-8 degree symbol
            "Figure 1/1D/α-β-γ.csv",  # Greek letters
        ]

        # Mock zip file path that matches the extraction_dir structure
        zip_file_path = "/mock/extraction/dir/suppl_data/figure_1.zip"

        mock_zip = MagicMock()
        mock_zip.infolist.return_value = [
            MagicMock(filename=path) for path in test_paths
        ]
        mock_zip.__enter__.return_value = mock_zip

        with patch("zipfile.ZipFile", return_value=mock_zip):
            assigner = TestPanelSourceAssigner.ConcretePanelSourceAssigner(
                self.config, self.prompt_handler
            )
            normalized_paths = assigner._get_zip_contents([zip_file_path])

            # The expected paths should include the relative zip path
            expected_zip_path = "suppl_data/figure_1.zip"

            # Verify each path is properly normalized
            for test_path in test_paths:
                # For the Windows-1252 path, we expect it to be normalized to the UTF-8 version
                if "IFN-ª┬" in test_path:
                    expected_path = test_path.replace("IFN-ª┬", "IFN-ß")
                else:
                    expected_path = test_path

                expected_full_path = f"{expected_zip_path}:{expected_path}"
                self.assertTrue(
                    any(p == expected_full_path for p in normalized_paths),
                    f"Missing normalized path for {test_path}\nExpected: {expected_full_path}\nGot: {normalized_paths}",
                )

    def test_windows_files_filtering(self):
        """Test filtering of Windows system files and metadata."""
        # Setup the test environment
        self.config["extraction_dir"] = "/mock/extraction/dir"

        test_files = [
            MagicMock(filename="folder/data.csv"),
            MagicMock(filename="Thumbs.db"),
            MagicMock(filename="folder/Thumbs.db"),
            MagicMock(filename="desktop.ini"),
            MagicMock(filename="folder/desktop.ini"),
            MagicMock(filename=".DS_Store"),
            MagicMock(filename="folder/.DS_Store"),
            MagicMock(filename="__MACOSX/._data.csv"),
            MagicMock(filename="folder/valid_file.xlsx"),
            MagicMock(filename="folder/subfolder/data.txt"),
        ]

        mock_zip = MagicMock()
        mock_zip.infolist.return_value = test_files
        mock_zip.__enter__.return_value = mock_zip

        # Use a zip path that matches the extraction_dir structure
        zip_path = "/mock/extraction/dir/suppl_data/test.zip"

        with patch("zipfile.ZipFile", return_value=mock_zip):
            assigner = TestPanelSourceAssigner.ConcretePanelSourceAssigner(
                self.config, self.prompt_handler
            )
            file_list = assigner._get_zip_contents([zip_path])

            # Expected files should include the relative zip path
            expected_files = [
                "suppl_data/test.zip:folder/data.csv",
                "suppl_data/test.zip:folder/valid_file.xlsx",
                "suppl_data/test.zip:folder/subfolder/data.txt",
            ]

            # Verify system files are filtered out
            self.assertEqual(sorted(file_list), sorted(expected_files))
            self.assertFalse(any("Thumbs.db" in path for path in file_list))
            self.assertFalse(any("desktop.ini" in path for path in file_list))
            self.assertFalse(any(".DS_Store" in path for path in file_list))
            self.assertFalse(any("__MACOSX" in path for path in file_list))
