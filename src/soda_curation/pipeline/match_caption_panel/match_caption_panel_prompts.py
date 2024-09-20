"""
This module provides prompt templates for panel caption matching tasks.

It contains predefined prompts that can be used with various AI models to match
panel captions with their corresponding images in scientific figures.
"""

from string import Template

SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing scientific figures. Your task is to match panel images with their corresponding captions from the main figure caption. Follow these instructions carefully:

1. Analyze the provided panel image carefully.
2. Read the entire figure caption.
3. Identify which part of the caption corresponds to the panel in the image.
4. Provide a caption specific to the panel that:
   a. Stays as close as possible to the original wording from the figure caption.
   b. Includes only minimal necessary adjustments to ensure the panel caption can be understood independently.
   c. Retains all relevant scientific details, including statistical information and methodological notes.
   d. Avoids adding interpretations or information not present in the original caption.
5. Start your response with 'PANEL_X:', where X is the label of the panel (e.g., A, B, C, etc.).
6. Ensure the panel-specific caption provides enough context to understand the panel without referring back to the main caption, but prioritize fidelity to the original text.

Example output format:
PANEL_A: Immunofluorescence staining of protein X (green) and protein Y (red) in cell type Z. Scale bar: 10 μm.

Remember, your goal is to create a panel-specific caption that could stand alone if necessary, while remaining as faithful as possible to the original figure caption's wording and content.
"""

USER_PROMPT = Template("""
Figure Caption:
$figure_caption

Please analyze the provided panel image and match it with the appropriate part of the figure caption above. Provide a caption specific to this panel, following the format and guidelines specified in the system prompt. Remember to stay as close as possible to the original wording while ensuring the panel caption can be understood independently.
""")

def get_match_panel_caption_prompt(figure_caption: str) -> str:
    """
    Generate a prompt for matching panel captions.

    Args:
        figure_caption (str): The caption of the entire figure.

    Returns:
        str: A formatted prompt string for AI models to match panel captions.
    """
    return USER_PROMPT.substitute(figure_caption=figure_caption)
