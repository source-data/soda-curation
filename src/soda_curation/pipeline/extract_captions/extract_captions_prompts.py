from string import Template

EXTRACT_CAPTIONS_PROMPT = Template("""You are an AI assistant specialized in analyzing scientific manuscripts. Your task is to extract complete figure captions from the given text, which is from a scientific paper. Please follow these guidelines carefully:

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

7. Do not include any explanations or additional text outside of the JSON object in your response.

Please process the given text and return the JSON object with the extracted figure captions, ensuring that each caption includes the FULL text as it appears in the manuscript, from the figure number to the last piece of information before the next figure or the end of the captions section.

$file_content
""")

def get_extract_captions_prompt(file_content: str) -> str:
    """
    Generate a prompt for extracting figure captions.

    Args:
        file_content (str): The content of the DOCX file.

    Returns:
        str: A formatted prompt string for AI models to extract figure captions.
    """
    return EXTRACT_CAPTIONS_PROMPT.substitute(file_content=file_content)
