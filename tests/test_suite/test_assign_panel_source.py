import os
import unittest
from unittest.mock import MagicMock, patch

from src.soda_curation.pipeline.assign_panel_source.assign_panel_source_base import (
    AsignedFiles,
    AsignedFilesList,
    PanelSourceAssigner,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
)


class TestPanelSourceAssigner(unittest.TestCase):
    class ConcretePanelSourceAssigner(PanelSourceAssigner):
        def _validate_config(self):
            pass

        def call_ai_service(self, prompt: str) -> AsignedFilesList:
            # Mock response
            return AsignedFilesList(
                assigned_files=[
                    AsignedFiles(
                        panel_label="A",
                        panel_sd_files=[
                            "suppl_data/figure_1.zip:A/A_1.csv",
                            "suppl_data/figure_1.zip:A/A_2.csv",
                            "suppl_data/figure_1.zip:A/A_3.csv",
                        ],
                    ),
                    AsignedFiles(
                        panel_label="B",
                        panel_sd_files=[
                            "suppl_data/figure_1.zip:B/B_1.csv",
                            "suppl_data/figure_1.zip:B/B_2.csv",
                            "suppl_data/figure_1.zip:B/B_3.csv",
                        ],
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
        # Example AI response
        ai_response = {
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
        }

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
                AsignedFiles(panel_label="A", panel_sd_files=["file1.csv"]),
                AsignedFiles(panel_label="B", panel_sd_files=["file2.csv"]),
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
