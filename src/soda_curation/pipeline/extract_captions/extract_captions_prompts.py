"""
This module provides prompt templates for figure caption extraction tasks.

It contains predefined prompts that can be used with various AI models to extract
figure captions from scientific documents.
"""

from string import Template

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
then, you should have figures 2, 3, 4, 5, 6 in between. Please make sure of this as it is of extreme importance.

8. Your response should ONLY contain the JSON object, without any additional explanations or text.

Example of expected output format:

{
  "Figure 1": "Title of Figure 1. A) Description of panel A. B) Description of panel B. Statistical analysis: p < 0.05.",
  "Figure 2": "Title of Figure 2. Detailed description of the figure, including multiple paragraphs if necessary.",
  "Figure EV1": "Title of Extended View Figure 1. Description of the extended view figure."
}

Please process the given text and return ONLY the JSON object with the extracted figure captions:

$file_content
"""
)


def get_extract_captions_prompt(file_content: str) -> str:
    """
    Generate a prompt for extracting figure captions from a scientific document.

    This function creates a prompt string that instructs an AI model to extract
    figure captions from the given file content. The prompt includes detailed
    guidelines on how to identify and format the captions.

    Args:
        file_content (str): The content of the scientific document from which
                            captions should be extracted.

    Returns:
        str: A formatted prompt string for AI models to extract figure captions.
    """
    return EXTRACT_CAPTIONS_PROMPT.substitute(file_content=file_content)
