from string import Template

SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing scientific figures. Your task is to match panel images with their corresponding captions from the main figure caption.

Instructions:
1. Analyze the provided panel image carefully.
2. Read the entire figure caption.
3. Identify which part of the caption corresponds to the panel in the image.
4. Provide a concise but informative caption specific to the panel.
5. Start your response with 'PANEL_X:', where X is the label of the panel (e.g., A, B, C, etc.).
6. Ensure the panel-specific caption is self-explanatory and provides enough context to understand the panel without referring back to the main caption.

Example output format:
PANEL_A: This panel shows the correlation between X and Y, with a significant positive trend (p < 0.001).
"""

USER_PROMPT = Template("""
Figure Caption:
$figure_caption

Please analyze the provided panel image and match it with the appropriate part of the figure caption above. Provide a concise but informative caption specific to this panel, following the format specified in the system prompt.
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
