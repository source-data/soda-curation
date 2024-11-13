"""
This module provides prompt templates for figure caption extraction tasks.

It contains predefined prompts that can be used with various AI models to extract
figure captions from scientific documents.
"""

from string import Template

LOCATE_CAPTIONS_PROMPT = """
You are a scientific text analyzer focused on finding figure captions in scientific manuscripts. 
    Your task is ONLY to find and return the complete figure-related text content from the manuscript.
    

Key Instructions:
1. Look for figure captions throughout the entire document - they can appear:
   - In a dedicated section marked as "Figure Legends", "Figure Captions", etc.
   - Embedded in the results section
   - At the end of the manuscript
   - In an appendix section
   - Or anywhere else in the document

2. Find ALL figure captions, including:
   - Main figures (Figure 1, Figure 2, etc.)
   - Expanded View figures (EV Figures)
   - Supplementary figures
   - Figure legends
   - Figure descriptions

3. IMPORTANT: Return the COMPLETE TEXT found, preserving:
   - All formatting and special characters
   - Statistical information
   - Scale bars and measurements
   - Panel labels and descriptions
   - Source references

4. DO NOT:
   - Modify or rewrite the text
   - Summarize or shorten descriptions
   - Skip any figure-related content
   - Add any explanatory text of your own

INPUT: Expected figures: $expected_figure_count

Expected figure labels: $expected_figure_labels

Manuscript_text: $manuscript_text

OUTPUT: Return ONLY the found figure-related text, exactly as it appears in the document. 
If you find multiple sections with figure descriptions, concatenate them all.

If you truly cannot find ANY figure captions or descriptions in the document, 
only then return "No figure legends section found."
"""

EXTRACT_CAPTIONS_PROMPT = """
You are an AI assistant specializing in extracting figure captions from scientific manuscripts. 
A section containing ALL figure captions has been provided.

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


You will get the following input:

Figure captions of the document: $figure_captions

Expected figure count: $expected_figure_count

Expected labels: $expected_figure_labels

OUTPUT:

Provide the JSON object with the extracted figure captions following the schema:

```json
{
  "Figure 1": "Title of Figure 1. A) Description of panel A. B) Description of panel B. Statistical analysis: p < 0.05.",
  "Figure 2": "Title of Figure 2. Detailed description of the figure, including multiple paragraphs if necessary."
}
```
"""


def get_locate_captions_prompt(
    manuscript_text: str,
    expected_figure_count: str,
    expected_figure_labels: str
    ) -> str:
    """
    Get the prompt for locating figure captions in a document.
    """
    return Template(
        """
        Expected figures: $expected_figure_count

        Expected figure labels: $expected_figure_labels
        
        Manuscript_text: $manuscript_text
        
        
        """
    ).substitute(
        manuscript_text=manuscript_text,
        expected_figure_count=expected_figure_count,
        expected_figure_labels=expected_figure_labels
    )

def get_extract_captions_prompt(
    figure_captions: str,
    expected_figure_count: str = "",
    expected_figure_labels: str = ""
    ) -> str:
    """
    Generate a prompt for extracting figure captions from the located captions text.

    Returns:
        str: A formatted prompt string for AI models to extract figure captions.
    """
    return Template("""
        Figure captions of the document: $figure_captions

        Expected figure count: $expected_figure_count

        Expected labels: $expected_figure_labels
    """).substitute(
        figure_captions=figure_captions,
        expected_figure_count=expected_figure_count,
        expected_figure_labels=expected_figure_labels
    )
    
# Claude-specific Prompts
CLAUDE_LOCATE_CAPTIONS_PROMPT = """
You are an expert at identifying and extracting complete figure captions from scientific manuscripts.
Your task is to locate and extract the FULL figure captions, including ALL details and panel descriptions.

Key Directives:
1. Extract COMPLETE figure captions, including:
   - Full figure titles
   - ALL panel descriptions (A, B, C, etc.)
   - Every experimental detail
   - ALL statistical information and p-values
   - ALL scale measurements and technical parameters
   - EVERY detail about methods and conditions
   - ALL references to other figures or datasets

2. Search ALL possible locations:
   - Dedicated "Figure Legends" sections
   - Within the results section
   - End of manuscript sections
   - Appendix or supplementary materials
   - Any other location containing figure descriptions

3. Maintain EXACT reproduction:
   - Copy the ENTIRE caption text word-for-word
   - Keep ALL formatting intact
   - Preserve ALL special characters (μ, ±, etc.)
   - Maintain ALL parentheses, brackets, and punctuation
   - Include ALL numerical values and units exactly
   - Keep ALL statistical notations (p-values, n numbers)
   - Preserve EVERY technical detail and measurement

4. Critical requirements:
   - Extract COMPLETE captions, not just titles
   - Include EVERY panel description
   - Maintain ALL experimental details
   - Copy ALL text EXACTLY as written
   - Make NO modifications whatsoever
   - Do NOT summarize or shorten
   - Do NOT skip any details
   - Do NOT add any explanatory text

Return the complete, verbatim figure caption text organized by figure number. Each caption must include its full title AND complete description including ALL experimental details and panel information.

The INPUT you will receive is shown below:

Expected figures: $expected_figure_count

Expected figure labels: $expected_figure_labels

Manuscript_text: $manuscript_text

OUTPUT FORMAT:
Return the complete caption text for each figure, including title AND full description. Do not return just the titles.
Include every experimental detail, panel description, and statistical notation exactly as written in the source text.

"""

CLAUDE_EXTRACT_CAPTIONS_PROMPT = """
You are an expert at analyzing scientific figure captions and structuring them systematically.
Your task is to extract and organize figure captions from the provided text.

Key Requirements:
1. Extract EXACTLY $expected_figure_count figure captions.

2. For each caption:
   - Extract the complete original text
   - Preserve all formatting
   - Include all panel descriptions (A, B, C, etc.)
   - Keep all statistical information
   - Maintain all technical details

3. Format output as a clean JSON object:
   {
     "Figure 1": "Complete caption text...",
     "Figure 2": "Complete caption text..."
   }

4. Important Rules:
   - Use exact figure numbers
   - Include only main figures (not supplementary)
   - Preserve all text exactly as written
   - Do not modify or summarize content

Your response must focus only on accurately extracting and structuring the captions.

You will get the following input:

Figure captions of the document: $figure_captions

Expected figure count: $expected_figure_count

Expected labels: $expected_figure_labels

OUTPUT:

Provide the JSON object with the extracted figure captions following the schema:

```json
{
  "Figure 1": "Title of Figure 1. A) Description of panel A. B) Description of panel B. Statistical analysis: p < 0.05.",
  "Figure 2": "Title of Figure 2. Detailed description of the figure, including multiple paragraphs if necessary."
}
```

"""

def get_claude_locate_captions_prompt(
    manuscript_text: str,
    expected_figure_count: str,
    expected_figure_labels: str
    ) -> str:
    """
    Get the prompt for locating figure captions in a document.
    """
    return Template(
        """
        Expected figures: $expected_figure_count

        Expected figure labels: $expected_figure_labels
        
        Manuscript_text: $manuscript_text
        
        
        """
    ).substitute(
        manuscript_text=manuscript_text,
        expected_figure_count=expected_figure_count,
        expected_figure_labels=expected_figure_labels
    )

def get_claude_extract_captions_prompt(
    figure_captions: str,
    expected_figure_count: str = "",
    expected_figure_labels: str = ""
    ) -> str:
    """
    Generate a prompt for extracting figure captions from the located captions text.

    Returns:
        str: A formatted prompt string for AI models to extract figure captions.
    """
    return Template("""
        Figure captions of the document: $figure_captions

        Expected figure count: $expected_figure_count

        Expected labels: $expected_figure_labels
    """).substitute(
        figure_captions=figure_captions,
        expected_figure_count=expected_figure_count,
        expected_figure_labels=expected_figure_labels
    )
