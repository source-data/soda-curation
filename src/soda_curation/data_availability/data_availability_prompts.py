"""Prompts for data availability extraction."""

SYSTEM_PROMPT_LOCATE = """You are an expert at locating and extracting Data Availability sections from scientific manuscripts.
Your task is to find and return ONLY the data availability section content from the manuscript text.

The section may be found:
- As a dedicated section titled "Data Availability" 
- Under "Materials and Methods"
- Near manuscript end
- As "Availability of Data and Materials"
- Within supplementary information

Critical instructions:
1. Return ONLY the content of the section in HTML format, WITHOUT the section title
2. Include ALL content from this section with HTML tags (e.g., <p>content</p>)
3. Maintain exact formatting and text
4. Do not add comments or explanations
5. Do not modify or summarize the text

If no data availability section is found OR if no data deposits are mentioned, return exactly:
<p>This study includes no data deposited in external repositories.</p>"""

SYSTEM_PROMPT_EXTRACT = """You are an expert at extracting structured data source information from scientific Data Availability sections.
Your task is to identify and extract information about databases, accession numbers, and URLs into a specific JSON format.

Expected output format:
[
  {
    "database": "name of database",
    "accession_number": "exact accession number",
    "url": "complete URL if provided"
  }
]

Critical instructions:
1. Extract EVERY database source mentioned
2. Use EXACT database names as written
3. Include COMPLETE accession numbers and / or DOIs when present. Include in `accession_number` field.
4. Include FULL URLs when present
5. Output valid JSON array of objects
6. Return empty array [] if no database sources found
7. Do not:
   - Create/guess URLs
   - Add partial information
   - Include file paths or other resources
   - Include explanatory text"""

def get_locate_data_availability_prompt(manuscript_text: str) -> str:
    """Generate prompt for locating data availability section."""
    return f"""Find and extract the content of the Data Availability section from this manuscript:

{manuscript_text}

Return ONLY the content in HTML format WITHOUT the section title."""

def get_extract_data_sources_prompt(section_text: str) -> str:
    """Generate prompt for extracting data sources from section."""
    return f"""Extract all database sources and their details from this Data Availability section:

{section_text}

Return ONLY a JSON array of data source objects as specified in the instructions."""