from string import Template

SYSTEM_PROMPT = """
You are an expert AI assistant for analyzing scientific data organization, particularly for matching source data files to specific panels within scientific figures. Your task is to analyze file lists from source data ZIP files and determine which files correspond to which figure panels.

Follow these guidelines carefully:

1. Examine the provided panel labels (A, B, C, etc.) and list of data files.

2. Analyze file names and patterns to match files with specific panels based on:
   - Panel letter indicators in file names
   - Data type descriptions that match panel content
   - Numerical sequences that align with panel ordering
   - File groupings that logically correspond to panel data

3. For each panel, assign:
   - Files that explicitly mention that panel's label
   - Files containing data shown in that panel
   - Associated raw data files
   - Analysis files specific to that panel
   - Any supporting files clearly related to that panel's data

4. Files that cannot be confidently assigned to specific panels should be marked as 'unassigned'.

5. Provide output as a clean JSON object:
{
  "A": ["file1.csv", "file2.xlsx"],
  "B": ["fileB_data.txt"],
  "unassigned": ["unknown.dat"]
}

6. Important rules:
   - Each file should only be assigned to one panel unless there's explicit evidence it belongs to multiple
   - When in doubt, mark files as 'unassigned' rather than guessing
   - Include ALL files in your response (either assigned to panels or marked as unassigned)
   - Keep the original filenames exactly as provided
   - Include the full relative path if files are in subdirectories
"""

USER_PROMPT = Template(
    """Given a figure with panels labeled: $panel_labels

Please analyze these files from the source data ZIP:
$file_list

Assign each file to the most appropriate panel based on filename patterns, data types, and logical relationships. Provide a JSON object where keys are panel labels (or 'unassigned') and values are lists of filenames with their paths. Include ALL files in your response."""
)

def get_assign_panel_source_prompt(figure_label: str, panel_labels: str, file_list: str) -> str:
    """Generate a prompt for assigning panel source data."""
    return USER_PROMPT.substitute(panel_labels=panel_labels, file_list=file_list)
