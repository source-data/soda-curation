from string import Template

SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing scientific data structures and file organizations. Your task is to assign source data files to specific panels within a scientific figure. Follow these instructions carefully:

1. Analyze the provided list of files and folders for a given figure.
2. Identify patterns in file names or folder structures that indicate association with specific panels.
3. Assign each file to the most appropriate panel based on these patterns.
4. If a file cannot be confidently assigned to a specific panel, label it as 'unassigned'.
5. Provide your response as a JSON object where keys are panel labels (or 'unassigned') and values are lists of file paths.

Example output format:
{
    "A": ["path/to/file1.csv", "path/to/file2.xlsx"],
    "B": ["path/to/fileB.txt"],
    "unassigned": ["path/to/unknown.dat"]
}

Remember, your goal is to create accurate associations between source data files and figure panels based on the given information.
"""

USER_PROMPT = Template(
    """
Figure Label: $figure_label
Panel Labels: $panel_labels

File Structure:
$file_list

Please assign each file to the most appropriate panel based on the file name or folder structure.
If a file cannot be confidently assigned to a specific panel, label it as 'unassigned'.

Respond with a JSON object where keys are panel labels (or 'unassigned') and values are lists of file paths.
"""
)

def get_assign_panel_source_prompt(figure_label: str, panel_labels: str, file_list: str) -> str:
    """
    Generate a prompt for assigning panel source data.

    Args:
        figure_label (str): The label of the figure.
        panel_labels (str): A string of panel labels, comma-separated.
        file_list (str): A string representation of the file structure.

    Returns:
        str: A formatted prompt string for AI models to assign panel source data.
    """
    return USER_PROMPT.substitute(figure_label=figure_label, panel_labels=panel_labels, file_list=file_list)
