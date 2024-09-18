from string import Template

SYSTEM_PROMPT = """
You will receive a text with the caption of a scientific figure. 
This figure will be generally composed of several panels. 
Extract the relevant part of the figure caption so that it matches the panel given as an image file. 
If a generic description of several panels is in place, return the generic and the specific descriptions for a given panel. 
Make sure that the information in the panel caption you return is enough to interpret the panel. 
For simplicity in post-processing begin the caption always with 'Panel X:' where X is the label of the panel in the figure.

Output format:
```PANEL_{label}: {caption}```
"""

USER_PROMPT = Template("""Figure caption: $figure_caption

Analyze the provided image, which represents a panel from the figure described above. Your task is to:
1. Identify the label of this specific panel (e.g., A, B, C, etc.).
2. Provide a self-explanatory caption for this panel. The caption should be detailed enough to interpret the panel without needing to refer back to the main figure caption.

Remember to format your response as specified in the system prompt.""")

def get_match_panel_caption_prompt(figure_caption: str) -> str:
    """
    Generate a prompt for matching panel captions.

    Args:
        figure_caption (str): The caption of the entire figure.

    Returns:
        str: A formatted prompt string for AI models to match panel captions.
    """
    return USER_PROMPT.substitute(figure_caption=figure_caption)
