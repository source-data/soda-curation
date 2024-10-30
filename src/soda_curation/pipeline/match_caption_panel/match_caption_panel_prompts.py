"""
This module provides prompt templates for panel caption matching tasks.

It contains predefined prompts that can be used with various AI models to match
panel captions with their corresponding images in scientific figures.
"""

from string import Template

SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing scientific figures. Your task is to match panel images with their corresponding captions from the main figure caption. Follow these instructions carefully:

1. Look at the provided panel image and read the full figure caption carefully.
2. Identify which part of the caption corresponds to the panel image you see.
3. Look for panel labels like (A), (B), etc., or descriptions that clearly match the image content.
4. For each panel, you MUST:
   - Start your response with "PANEL_X:" where X is the panel letter (e.g., PANEL_A:)
   - Extract the relevant caption text that specifically describes this panel
   - Include any statistical information, methodological notes, or scale bars relevant to this panel
   - Make the panel caption understandable on its own while staying true to the original text

5. Key guidelines:
   - Only include text relevant to the specific panel shown
   - Maintain all scientific details and statistical information
   - Keep original phrasing where possible
   - Ensure panel labels are in the specified format (PANEL_A:, PANEL_B:, etc.)
   - If you can't determine the panel label, use visual and caption context to make your best assessment

Example response format:
PANEL_A: Immunofluorescence staining of protein X (green) and protein Y (red) in cell type Z. Scale bar: 10 μm.
"""

USER_PROMPT = Template(
    """Please analyze this panel image from a figure with the following caption:

$figure_caption

Based on the image content and the caption text, identify which panel this represents and provide its specific caption maintaining scientific accuracy and completeness."""
)

def get_match_panel_caption_prompt(figure_caption: str) -> str:
    """
    Generate a prompt for matching panel captions.

    Args:
        figure_caption (str): The caption of the entire figure.

    Returns:
        str: A formatted prompt string for AI models to match panel captions.
    """
    return USER_PROMPT.substitute(figure_caption=figure_caption)
