"""
This module provides prompt templates for figure caption extraction tasks.

It contains predefined prompts that can be used with various AI models to extract
figure captions from scientific documents.
"""

from string import Template

LOCATE_CAPTIONS_PROMPT = """You are an AI assistant specializing in scientific manuscripts. Your task is to locate and extract ALL figure captions from the document.

1. Find ALL sections containing figure legends/captions in the document. These are typically:
   - In a dedicated "Figure Legends" section
   - After the references/bibliography
   - At the end of the document
   - Sometimes labeled as "Figure Captions" or "Figure Descriptions"

2. Once you find this section, extract ALL the text starting from the first figure caption 
   until the last one, maintaining EXACT formatting and content.
   
3. IMPORTANT:
   - Include ONLY main figure captions (Figure 1, Figure 2, etc.)
   - DO NOT include supplementary figures or EV figures
   - Maintain ALL text EXACTLY as written
   - Include ALL sub-panel descriptions (A, B, C, etc.)
   - Keep ALL statistical information and references
   
4. Return ONLY the extracted text, with no additional explanations or comments.

Please process the document and extract ALL figure captions:"""

EXTRACT_CAPTIONS_PROMPT = Template("""You are an AI assistant specializing in extracting figure captions from scientific manuscripts. A section containing ALL figure captions has been provided.

Your task is to parse these captions into a structured format:

1. You MUST extract EXACTLY $expected_figure_count figure captions from the provided text.

2. For each caption (Figure 1 through Figure $expected_figure_count):
   - Copy the ENTIRE caption text EXACTLY as it appears
   - Include ALL subsections (A, B, C, etc.)
   - Maintain ALL formatting, punctuation, and special characters
   - DO NOT modify or summarize the text

3. Create a JSON object where:
   - Keys are "Figure 1", "Figure 2", etc. (up to $expected_figure_count)
   - Values are the complete, exact caption text

4. RULES:
   - Include ONLY main figures (1,2,3...)
   - Skip EV figures or supplementary figures
   - Maintain consecutive numbering
   - Keep all statistical information and references

Here are ALL the figure captions from the manuscript:

$figure_captions""")

def get_locate_captions_prompt() -> str:
    """
    Get the prompt for locating figure captions in a document.

    Returns:
        str: The prompt for caption location.
    """
    return LOCATE_CAPTIONS_PROMPT

def get_extract_captions_prompt(figure_captions: str, expected_figure_count: int) -> str:
    """
    Generate a prompt for extracting figure captions from the located captions text.

    Args:
        figure_captions (str): The text containing all figure captions.
        expected_figure_count (int): The expected number of figure captions.

    Returns:
        str: A formatted prompt string for AI models to extract figure captions.
    """
    return EXTRACT_CAPTIONS_PROMPT.substitute(
        figure_captions=figure_captions,
        expected_figure_count=expected_figure_count
    )
