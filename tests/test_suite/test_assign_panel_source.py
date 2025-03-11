import unittest
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

from pydantic import BeforeValidator
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
        """Set up test environment."""
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
        self.extract_dir = Path("/mock/extraction/dir")  # Add extract_dir
        self.assigner = self.ConcretePanelSourceAssigner(
            self.config, self.prompt_handler, self.extract_dir  # Pass extract_dir
        )

    @patch("zipfile.ZipFile")
    def test_get_zip_contents(self, mock_zipfile):
        """Test extraction of zip contents."""
        # Create necessary files
        zip_path = Path(self.extract_dir) / "path/to/zip_file.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.touch()

        data_path = Path(self.extract_dir) / "path/to/data_file.dat"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.touch()

        # Create mock zip file contents
        mock_zip = MagicMock()
        mock_zip.__enter__.return_value = mock_zip
        mock_zip.infolist.return_value = [
            MagicMock(filename=f)
            for f in [
                "zip_file/A/file1.csv",
                "zip_file/A/file2.xlsx",
                "zip_file/A/file3.txt",
                "zip_file/B/file4.dat",
                "zip_file/B/file5.csv",
                "zip_file/B/file6.xlsx",
                "zip_file/B/file7.txt",
            ]
        ]
        mock_zipfile.return_value = mock_zip

        # Test with relative paths
        sd_files = ["path/to/zip_file.zip", "path/to/data_file.dat"]
        output = self.assigner._get_zip_contents(sd_files)

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

        self.assertEqual(sorted(output), sorted(expected_output))

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
        # Create necessary files and directories
        zip_path = Path(self.extract_dir) / "path" / "to" / "data.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.touch()

        data_path = Path(self.extract_dir) / "path" / "to" / "file.dat"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.touch()

        # Mock the zipfile
        mock_zip = MagicMock()
        mock_zip.__enter__.return_value = mock_zip
        mock_zip.infolist.return_value = [
            MagicMock(filename=f)
            for f in [
                "__MACOSX/folder/._file1.csv",
                "folder/.DS_Store",
                "folder/file1.csv",
                "folder/file2.xlsx",
                "__MACOSX/folder/._file2.xlsx",
                ".DS_Store",
            ]
        ]
        mock_zipfile.return_value = mock_zip

        # Use relative paths for input
        sd_files = [
            "path/to/data.zip",
            "path/to/file.dat",
        ]

        expected_output = [
            "path/to/data.zip:folder/file1.csv",
            "path/to/data.zip:folder/file2.xlsx",
            "path/to/file.dat",
        ]

        output = self.assigner._get_zip_contents(sd_files)
        self.assertEqual(sorted(output), sorted(expected_output))


class TestPanelSourceAssignerValidation(unittest.TestCase):
    """Test validation features of PanelSourceAssigner."""

    def setUp(self):
        """Set up test environment."""
        self.config = {
            "pipeline": {
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
            },
            "extraction_dir": "/mock/extraction/dir",
        }
        self.prompt_handler = MagicMock()
        self.extract_dir = Path("/mock/extraction/dir")  # Add extract_dir
        self.assigner = TestPanelSourceAssigner.ConcretePanelSourceAssigner(
            self.config, self.prompt_handler, self.extract_dir  # Pass extract_dir
        )

    def test_hallucinated_files_validation(self):
        """Test that hallucinated file paths are rejected."""
        # Define valid files
        valid_files = [
            "suppl_data/figure_1.zip:A/real_file1.csv",
            "suppl_data/figure_1.zip:A/real_file2.xlsx",
            "suppl_data/figure_1.zip:B/real_file3.csv",
        ]

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
            filtered_assigned, _ = self.assigner.filter_files(
                assigned_files=[
                    AsignedFiles(
                        panel_label=case["panel_label"],
                        panel_sd_files=case["panel_sd_files"],
                    )
                ],
                not_assigned_files=[],
                allowed_files=valid_files,
            )

            if case["should_pass"]:
                # For valid cases, check that we have results and they're all valid
                self.assertTrue(
                    len(filtered_assigned) > 0, "Expected valid files but got none"
                )
                self.assertTrue(
                    all(f in valid_files for f in filtered_assigned[0].panel_sd_files),
                    f"Invalid files found in results: {filtered_assigned[0].panel_sd_files}",
                )
            else:
                # For invalid cases, either the panel should be removed (empty list)
                # or all its files should be valid
                if filtered_assigned:
                    self.assertTrue(
                        all(
                            f in valid_files
                            for f in filtered_assigned[0].panel_sd_files
                        ),
                        f"Invalid files were not filtered out: {filtered_assigned[0].panel_sd_files}",
                    )

    def test_windows_files_filtering(self):
        """Test filtering of Windows system files and metadata."""
        # Create the zip file first
        zip_dir = self.extract_dir / "suppl_data"
        zip_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zip_dir / "test.zip"
        zip_path.touch()

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

        with patch("zipfile.ZipFile") as mock_zipfile:
            mock_zip = MagicMock()
            mock_zip.__enter__.return_value = mock_zip
            mock_zip.infolist.return_value = test_files
            mock_zipfile.return_value = mock_zip

            file_list = self.assigner._get_zip_contents(["suppl_data/test.zip"])

            expected_files = [
                "suppl_data/test.zip:folder/data.csv",
                "suppl_data/test.zip:folder/valid_file.xlsx",
                "suppl_data/test.zip:folder/subfolder/data.txt",
            ]

            assert sorted(file_list) == sorted(expected_files)

    def test_filter_files(self):
        """Test the filter_files static method."""
        # Define test data
        assigned_files = [
            AsignedFiles(
                panel_label="A",
                panel_sd_files=[
                    "valid_file1.csv",
                    "invalid_file1.csv",
                    "valid_file2.csv",
                ],
            ),
            AsignedFiles(
                panel_label="B", panel_sd_files=["invalid_file2.csv", "valid_file3.csv"]
            ),
        ]
        not_assigned_files = ["valid_file4.csv", "invalid_file3.csv"]
        allowed_files = [
            "valid_file1.csv",
            "valid_file2.csv",
            "valid_file3.csv",
            "valid_file4.csv",
        ]

        # Call the filter method
        filtered_assigned, filtered_not_assigned = self.assigner.filter_files(
            assigned_files=assigned_files,
            not_assigned_files=not_assigned_files,
            allowed_files=allowed_files,
        )

        # Check filtered assigned files
        self.assertEqual(len(filtered_assigned), 2)
        self.assertEqual(filtered_assigned[0].panel_label, "A")
        self.assertEqual(
            filtered_assigned[0].panel_sd_files, ["valid_file1.csv", "valid_file2.csv"]
        )
        self.assertEqual(filtered_assigned[1].panel_label, "B")
        self.assertEqual(filtered_assigned[1].panel_sd_files, ["valid_file3.csv"])

        # Check filtered not assigned files
        self.assertEqual(filtered_not_assigned, ["valid_file4.csv"])

    def test_update_figure_with_assignments_preserves_empty_sd_files(self):
        """Test that _update_figure_with_assignments preserves empty sd_files."""
        # Create a figure with existing panels
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(panel_label="A", panel_caption=""),
                Panel(panel_label="B", panel_caption=""),
            ],
            sd_files=[],
            img_files=[],
        )

        # Create new panels with empty sd_files
        new_panels = [
            Panel(panel_label="A", panel_caption="", sd_files=[]),
            Panel(panel_label="B", panel_caption="", sd_files=[]),
        ]

        # Update figure with assignments
        self.assigner._update_figure_with_assignments(
            figure=figure, panels=new_panels, not_assigned_files=[]
        )

        # Verify that each panel still has sd_files as an empty list
        for panel in figure.panels:
            self.assertTrue(hasattr(panel, "sd_files"))
            self.assertEqual(panel.sd_files, [])

    def test_panel_label_normalization(self):
        """Test that panel labels are properly normalized to prevent duplication."""
        # Test the normalize_panel_label method directly
        test_cases = [
            ("1A", "A"),
            ("A", "A"),
            ("Fig2B", "B"),
            ("Figure3C", "C"),
            ("1D", "D"),
            ("D", "D"),
            ("Panel2E", "E"),
            ("1A1", "A"),  # Handle numeric suffixes
            ("a", "A"),  # Handle lowercase
            ("1a", "A"),  # Handle mixed case
        ]

        for input_label, expected in test_cases:
            self.assertEqual(
                self.assigner.normalize_panel_label(input_label),
                expected,
                f"Failed to normalize {input_label} to {expected}",
            )

        # Test panel deduplication in figure updates
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(panel_label="1A", panel_caption="", sd_files=[]),
                Panel(panel_label="1B", panel_caption="", sd_files=[]),
            ],
            sd_files=["source_data1.zip"],
            img_files=[],
        )

        new_panels = [
            Panel(panel_label="A", panel_caption="", sd_files=["new_file1.csv"]),
            Panel(panel_label="B", panel_caption="", sd_files=["new_file2.csv"]),
        ]

        # Update figure with new panels
        self.assigner._update_figure_with_assignments(
            figure=figure, panels=new_panels, not_assigned_files=[]
        )

        # Verify no duplicate panels were created
        self.assertEqual(len(figure.panels), 2)

        # Verify the original panel labels were preserved but got the new sd_files
        panel_data = {panel.panel_label: panel.sd_files for panel in figure.panels}
        self.assertEqual(panel_data["A"], ["new_file1.csv"])
        self.assertEqual(panel_data["B"], ["new_file2.csv"])

    def test_mixed_panel_label_assignments(self):
        """Test handling of mixed panel label formats in assignments."""
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(panel_label="1A", panel_caption=""),
                Panel(panel_label="1B", panel_caption=""),
            ],
            sd_files=["source_data1.zip"],
            img_files=[],
        )

        # Create assignments with mixed label formats
        assigned_files_list = AsignedFilesList(
            assigned_files=[
                AsignedFiles(panel_label="A", panel_sd_files=["file1.csv"]),
                AsignedFiles(panel_label="1B", panel_sd_files=["file2.csv"]),
            ],
            not_assigned_files=[],
        )

        # Convert to panels
        panels = self.assigner.parse_assigned_files_to_panels(assigned_files_list)

        # Update figure
        self.assigner._update_figure_with_assignments(
            figure=figure, panels=panels, not_assigned_files=[]
        )

        # Verify results
        self.assertEqual(len(figure.panels), 2)
        panel_data = {panel.panel_label: panel.sd_files for panel in figure.panels}
        self.assertEqual(panel_data["A"], ["file1.csv"])
        self.assertEqual(panel_data["B"], ["file2.csv"])

    @patch("zipfile.ZipFile")
    def test_deduplication_of_existing_panels(self, mock_zipfile):
        """Test that panels are deduplicated during assignment."""
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

        # Create a figure with duplicate panels (B and 1B)
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(
                    panel_label="B",
                    panel_caption="Caption B",
                    confidence=0.8,
                    panel_bbox=[0.1, 0.1, 0.2, 0.2],
                    sd_files=[],
                ),
                Panel(
                    panel_label="1B",
                    panel_caption="",
                    confidence=0.0,
                    sd_files=["existing_file.csv"],
                ),
                Panel(panel_label="C", panel_caption="Caption C", sd_files=[]),
            ],
            sd_files=["source_data1.zip"],
            img_files=[],
        )

        # Mock the AI service response
        mock_response = AsignedFilesList(
            assigned_files=[
                AsignedFiles(panel_label="B", panel_sd_files=["file1.csv"]),
                AsignedFiles(panel_label="C", panel_sd_files=["file2.csv"]),
            ],
            not_assigned_files=[],
        )
        self.assigner.call_ai_service.return_value = mock_response

        # Call _assign_to_figure
        self.assigner._assign_to_figure(figure)

        # Verify panels were deduplicated
        self.assertEqual(len(figure.panels), 2)  # Should only have B and C now

        # Find panel B and verify it has the merged properties
        panel_b = next(
            p
            for p in figure.panels
            if self.assigner.normalize_panel_label(p.panel_label) == "B"
        )
        self.assertEqual(
            panel_b.panel_caption, "Caption B"
        )  # Should keep non-empty caption
        self.assertEqual(panel_b.confidence, 0.8)  # Should keep higher confidence
        self.assertEqual(
            panel_b.sd_files, ["file1.csv"]
        )  # Should have new sd_files from AI service
        self.assertEqual(
            panel_b.panel_bbox, [0.1, 0.1, 0.2, 0.2]
        )  # Should keep bbox from higher confidence

        # Verify panel C remains unchanged except for sd_files update
        panel_c = next(
            p
            for p in figure.panels
            if self.assigner.normalize_panel_label(p.panel_label) == "C"
        )
        self.assertEqual(panel_c.panel_caption, "Caption C")
        self.assertEqual(panel_c.sd_files, ["file2.csv"])

    @patch("zipfile.ZipFile")
    def test_panel_label_simplification(self, mock_zipfile):
        """Test that panel labels are simplified to just letters."""
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

        # Create a figure with numbered panel labels
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(panel_label="1A", panel_caption="Caption A", confidence=0.8),
                Panel(panel_label="1B", panel_caption="Caption B", confidence=0.7),
                Panel(panel_label="Fig1C", panel_caption="Caption C", confidence=0.6),
            ],
            sd_files=["source_data1.zip"],
            img_files=[],
        )

        # Mock the AI service response
        mock_response = AsignedFilesList(
            assigned_files=[
                AsignedFiles(panel_label="A", panel_sd_files=["file1.csv"]),
                AsignedFiles(panel_label="B", panel_sd_files=["file2.csv"]),
                AsignedFiles(panel_label="C", panel_sd_files=["file3.csv"]),
            ],
            not_assigned_files=[],
        )
        self.assigner.call_ai_service.return_value = mock_response

        # Process the figure
        self.assigner._assign_to_figure(figure)

        # Verify panel labels are simplified
        expected_labels = ["A", "B", "C"]
        actual_labels = [panel.panel_label for panel in figure.panels]
        self.assertEqual(sorted(actual_labels), sorted(expected_labels))

        # Verify other properties are preserved
        panel_data = {panel.panel_label: panel for panel in figure.panels}
        self.assertEqual(panel_data["A"].panel_caption, "Caption A")
        self.assertEqual(panel_data["B"].panel_caption, "Caption B")
        self.assertEqual(panel_data["C"].panel_caption, "Caption C")

    def test_update_figure_with_assignments(self):
        """Test that _update_figure_with_assignments correctly normalizes labels and updates panels."""
        # Create a figure with various panel label formats
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(
                    panel_label="1A",  # Should become "A"
                    panel_caption="Caption A",
                    confidence=0.8,
                    panel_bbox=[0.1, 0.1, 0.2, 0.2],
                    sd_files=["old_file1.csv"],
                ),
                Panel(
                    panel_label="Fig1B",  # Should become "B"
                    panel_caption="Caption B",
                    confidence=0.7,
                    panel_bbox=[0.2, 0.2, 0.3, 0.3],
                    sd_files=[],
                ),
            ],
            sd_files=["source_data1.zip"],
            img_files=[],
        )

        # Create new panels with normalized labels
        new_panels = [
            Panel(panel_label="A", panel_caption="", sd_files=["new_file1.csv"]),
            Panel(panel_label="B", panel_caption="", sd_files=["new_file2.csv"]),
        ]

        # Update figure with new assignments
        self.assigner._update_figure_with_assignments(
            figure=figure, panels=new_panels, not_assigned_files=["unassigned.csv"]
        )

        # Verify results
        self.assertEqual(len(figure.panels), 2)  # Should have 2 panels

        # Check panel A
        panel_a = next(
            p for p in figure.panels if p.panel_label == "A"
        )  # Note: looking for "A", not "1A"
        self.assertEqual(panel_a.panel_caption, "Caption A")
        self.assertEqual(panel_a.confidence, 0.8)
        self.assertEqual(panel_a.panel_bbox, [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(panel_a.sd_files, ["new_file1.csv"])

        # Check panel B
        panel_b = next(
            p for p in figure.panels if p.panel_label == "B"
        )  # Note: looking for "B", not "Fig1B"
        self.assertEqual(panel_b.panel_caption, "Caption B")
        self.assertEqual(panel_b.confidence, 0.7)
        self.assertEqual(panel_b.panel_bbox, [0.2, 0.2, 0.3, 0.3])
        self.assertEqual(panel_b.sd_files, ["new_file2.csv"])

        # Check unassigned files
        self.assertEqual(figure.unassigned_sd_files, ["unassigned.csv"])

    def test_update_figure_with_new_panel(self):
        """Test that _update_figure_with_assignments correctly handles new panels."""
        # Create a figure with one panel
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(
                    panel_label="1A",
                    panel_caption="Caption A",
                    confidence=0.8,
                    sd_files=["old_file1.csv"],
                ),
            ],
            sd_files=["source_data1.zip"],
            img_files=[],
        )

        # Create new panels including a new one
        new_panels = [
            Panel(panel_label="A", panel_caption="", sd_files=["new_file1.csv"]),
            Panel(
                panel_label="B",  # New panel
                panel_caption="New Panel",
                sd_files=["new_file2.csv"],
            ),
        ]

        # Update figure with new assignments
        self.assigner._update_figure_with_assignments(
            figure=figure, panels=new_panels, not_assigned_files=[]
        )

        # Verify results
        self.assertEqual(len(figure.panels), 2)  # Should now have 2 panels

        # Check panel A
        panel_a = next(p for p in figure.panels if p.panel_label == "A")
        self.assertEqual(
            panel_a.panel_caption, "Caption A"
        )  # Original caption preserved
        self.assertEqual(panel_a.confidence, 0.8)  # Original confidence preserved
        self.assertEqual(panel_a.sd_files, ["new_file1.csv"])  # New sd_files updated

        # Check new panel B
        panel_b = next(p for p in figure.panels if p.panel_label == "B")
        self.assertEqual(panel_b.panel_caption, "New Panel")
        self.assertEqual(panel_b.sd_files, ["new_file2.csv"])

    def test_update_figure_preserves_properties(self):
        """Test that _update_figure_with_assignments preserves important panel properties."""
        # Create a figure with a panel that has all properties set
        figure = Figure(
            figure_label="Figure 1",
            panels=[
                Panel(
                    panel_label="1A",
                    panel_caption="Original Caption",
                    confidence=0.9,
                    panel_bbox=[0.1, 0.1, 0.2, 0.2],
                    ai_response="Original AI Response",
                    sd_files=["old_file.csv"],
                ),
            ],
            sd_files=["source_data1.zip"],
            img_files=[],
        )

        # Create new panel with minimal properties
        new_panels = [
            Panel(panel_label="A", panel_caption="", sd_files=["new_file.csv"]),
        ]

        # Update figure
        self.assigner._update_figure_with_assignments(
            figure=figure, panels=new_panels, not_assigned_files=[]
        )

        # Verify all properties are preserved except sd_files
        panel = figure.panels[0]
        self.assertEqual(panel.panel_label, "A")  # Should be normalized
        self.assertEqual(panel.panel_caption, "Original Caption")
        self.assertEqual(panel.confidence, 0.9)
        self.assertEqual(panel.panel_bbox, [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(panel.ai_response, "Original AI Response")
        self.assertEqual(panel.sd_files, ["new_file.csv"])  # Only this should change

    def test_character_replacements(self):
        """Test replacement of non-standard characters in filenames."""
        zip_dir = self.extract_dir / "suppl_data"
        zip_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zip_dir / "figure_1.zip"
        zip_path.touch()

        test_paths = [
            "Figure 1/data_ª┬.xlsx",
            "Figure 1/data_í≈.csv",
            "Figure 1/normal_file.txt",
            "Figure 1/data_ª┬_í≈.xlsx",
        ]

        with patch("zipfile.ZipFile") as mock_zipfile:
            mock_zip = MagicMock()
            mock_zip.__enter__.return_value = mock_zip
            mock_zip.infolist.return_value = [
                MagicMock(filename=path) for path in test_paths
            ]
            mock_zipfile.return_value = mock_zip

            normalized_paths = self.assigner._get_zip_contents(
                ["suppl_data/figure_1.zip"]
            )

            expected_paths = [
                "suppl_data/figure_1.zip:Figure 1/data_β.xlsx",
                "suppl_data/figure_1.zip:Figure 1/data_△.csv",
                "suppl_data/figure_1.zip:Figure 1/normal_file.txt",
                "suppl_data/figure_1.zip:Figure 1/data_β_△.xlsx",
            ]

            assert sorted(normalized_paths) == sorted(expected_paths)
