"""
This module provides prompt templates for figure caption extraction tasks.

It contains predefined prompts that can be used with various AI models to extract
figure captions from scientific documents.
"""

from string import Template
from typing import Dict

EXTRACT_CAPTIONS_PROMPT = Template(
    """You are an AI assistant specialized in analyzing scientific manuscripts. Your task is to extract complete figure captions from the given text, which is from a scientific paper. Please follow these guidelines carefully:

1. Identify all figure captions in the text. These usually start with "Figure X" or "Fig. X", where X is a number.

2. For each figure, extract the ENTIRE caption, including:
   - The main title of the figure
   - All sub-sections (usually labeled A, B, C, etc.)
   - Any statistical information (e.g., p-values, correlation coefficients)
   - Methodological notes or references to other parts of the paper
   - Descriptions of all elements in the figure (e.g., what different colors or shapes represent)
   - Any additional explanatory text, no matter how long it is

3. Create a JSON object where:
   - Keys are the figure labels (e.g., "Figure 1", "Figure 2")
   - Values are the corresponding complete captions, including ALL text associated with that figure

4. Preserve all formatting within the caption text, including:
   - Line breaks between sub-sections
   - Special characters or symbols
   - Italicized text (indicated by *asterisks* or _underscores_)
   - Superscript (indicated by ^carets^) and subscript (indicated by ~tildes~)

5. Ensure that you capture the full context of each caption, even if it spans multiple paragraphs.

6. If there are no figures or captions in the text, return an empty JSON object.

7. Note that the figures will be always monotically increasing continuous numbers. If you have a figure 1 and figure 7, 
then, you should have figures 2, 3, 4, 5, 6 in between. Please make sure of this as it is of extreme importance. Similarly, if you have 
figures 5, 7, 8, then you MUST have figures 1, 2, 3, 4, and 6 in between.

8. Your response should ONLY contain the JSON object, without any additional explanations or text.

9. IMPORTANT: You MUST extract exactly $expected_figure_count figure captions. If you find fewer or more captions, double-check your work and ensure you've captured all figures.

10. DO NOT extract captions for figures labeled with "EV" (e.g., "Figure EV1", "Fig. EV2"). These are Extended View figures and should be ignored for this task.

Example of expected output format:

{
  "Figure 1": "Title of Figure 1. A) Description of panel A. B) Description of panel B. Statistical analysis: p < 0.05.",
  "Figure 2": "Title of Figure 2. Detailed description of the figure, including multiple paragraphs if necessary."
}

Please process the given text and return ONLY the JSON object with the extracted figure captions:

$file_content
"""
)


def get_extract_captions_prompt(file_content: str, expected_figure_count: int) -> str:
    """
    Generate a prompt for extracting figure captions from a scientific document.

    This function creates a prompt string that instructs an AI model to extract
    figure captions from the given file content. The prompt includes detailed
    guidelines on how to identify and format the captions.

    Args:
        file_content (str): The content of the scientific document from which
                            captions should be extracted.
        expected_figure_count (int): The expected number of figure captions to extract.

    Returns:
        str: A formatted prompt string for AI models to extract figure captions.
    """
    return EXTRACT_CAPTIONS_PROMPT.substitute(file_content=file_content, expected_figure_count=expected_figure_count)

FALLBACK_EXTRACT_CAPTIONS_PROMPT = Template(
    """You are an AI assistant specialized in analyzing scientific manuscripts. Your task is to find and extract the missing figure captions from the given text. Please follow these guidelines carefully:

1. You have been provided with a list of figure captions that were already extracted. Your job is to find the remaining captions that are missing.

2. The total number of figures in the document is $expected_figure_count. You need to find captions for the figures that are not in the list of already extracted captions.

3. For each missing figure, extract the ENTIRE caption, including:
   - The main title of the figure
   - All sub-sections (usually labeled A, B, C, etc.)
   - Any statistical information (e.g., p-values, correlation coefficients)
   - Methodological notes or references to other parts of the paper
   - Descriptions of all elements in the figure (e.g., what different colors or shapes represent)
   - Any additional explanatory text, no matter how long it is

4. Create a JSON object where:
   - Keys are the figure labels (e.g., "Figure 1", "Figure 2")
   - Values are the corresponding complete captions, including ALL text associated with that figure

5. Only include the missing figures in your JSON response. Do not include figures that were already extracted.

6. Ensure that you capture the full context of each caption, even if it spans multiple paragraphs.

7. Your response should ONLY contain the JSON object with the missing figure captions, without any additional explanations or text.

Already extracted captions:
$extracted_captions

Please process the given text and return ONLY the JSON object with the missing figure captions:

$file_content
"""
)

def get_fallback_extract_captions_prompt(file_content: str, expected_figure_count: int, extracted_captions: Dict[str, str]) -> str:
    """
    Generate a prompt for extracting missing figure captions from a scientific document.

    Args:
        file_content (str): The content of the scientific document.
        expected_figure_count (int): The total number of figure captions expected.
        extracted_captions (Dict[str, str]): The captions that have already been extracted.

    Returns:
        str: A formatted prompt string for AI models to extract missing figure captions.
    """
    extracted_captions_str = json.dumps(extracted_captions, indent=2)
    return FALLBACK_EXTRACT_CAPTIONS_PROMPT.substitute(
        file_content=file_content,
        expected_figure_count=expected_figure_count,
        extracted_captions=extracted_captions_str
    )
