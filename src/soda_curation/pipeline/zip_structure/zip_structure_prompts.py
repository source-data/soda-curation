from string import Template

STRUCTURE_ZIP_PROMPT = Template("""
Given the following list of files from a ZIP archive, parse the structure and return ONLY a JSON object following the ZipStructure format. Do not include any explanatory text before or after the JSON.

File list:
$file_list

Instructions:
1. The root folder name is the 'manuscript_id'.
2. The 'xml' file is in the root folder.
3. The 'docx' file is in a subfolder named 'doc', 'DOC', or 'Doc'.
4. The 'pdf' file is in a subfolder named 'pdf'.
5. The 'appendix' is typically a PDF file in the 'suppl_data' folder containing 'appendix' in its name.
6. For 'figures':
   - Main figure image files are in the 'graphic' folder.
   - Figure labels can be "Figure X" or "EV X" (for extended view figures).
   - Source data ('sd_files') are in the 'suppl_data' folder.
   - Include ALL related files from 'suppl_data' for each figure.
   - Match source data to figures based on number (e.g., "Figure 1" matches "Figure 1.zip", "Dataset 1.xlsx", "Table 1.docx").
   - For EV figures, match "EV1" with "Dataset EV1.xlsx", "Table EV1.docx", etc.

The JSON object should include 'manuscript_id', 'xml', 'docx', 'pdf', 'appendix', and 'figures' fields. The 'figures' field should be a list of objects, each with 'figure_label', 'img_files', 'sd_files', 'figure_caption', and 'figure_panels' fields. Use null for any missing values. Ensure the output is valid JSON.

$example_input_output

Now, parse the provided file list and return the JSON object:
""")

EXAMPLE_INPUT_OUTPUT = """
Example input-output:

Input:
EMM-2023-18636
├── EMM-2023-18636.xml
├── doc
│   └── manuscript.docx
├── graphic
│   ├── Figure 1.eps
│   └── EV1.eps
├── pdf
│   └── manuscript.pdf
└── suppl_data
    ├── Appendix.pdf
    ├── Figure 1.zip
    ├── Dataset 1.xlsx
    ├── Table 1.docx
    ├── Dataset EV1.xlsx
    └── Table EV1.docx

Output:
{
  "manuscript_id": "EMM-2023-18636",
  "xml": "EMM-2023-18636.xml",
  "docx": "doc/manuscript.docx",
  "pdf": "pdf/manuscript.pdf",
  "appendix": ["suppl_data/Appendix.pdf"],
  "figures": [
    {
      "figure_label": "Figure 1",
      "img_files": ["graphic/Figure 1.eps"],
      "sd_files": [
        "suppl_data/Figure 1.zip",
        "suppl_data/Dataset 1.xlsx",
        "suppl_data/Table 1.docx"
      ],
      "figure_caption": "TO BE ADDED IN LATER STEP",
      "figure_panels": []
    },
    {
      "figure_label": "Figure EV1",
      "img_files": ["graphic/EV1.eps"],
      "sd_files": [
        "suppl_data/Dataset EV1.xlsx",
        "suppl_data/Table EV1.docx"
      ],
      "figure_caption": "TO BE ADDED IN LATER STEP",
      "figure_panels": []
    }
  ]
}
"""

def get_structure_zip_prompt(file_list, custom_instructions=None):
    """
    Generate a prompt for ZIP structure parsing.

    This function creates a prompt string for AI models to parse the structure of a ZIP file.
    It includes instructions on how to interpret the file list and format the output.

    Args:
        file_list (str): A string representation of the files in the ZIP archive.
        custom_instructions (str, optional): Custom instructions to override the default example.
            Defaults to None, in which case EXAMPLE_INPUT_OUTPUT is used.

    Returns:
        str: A formatted prompt string for AI models to parse the ZIP structure.
    """
    instructions = custom_instructions or EXAMPLE_INPUT_OUTPUT
    return STRUCTURE_ZIP_PROMPT.substitute(
        file_list=file_list,
        example_input_output=instructions
    )
