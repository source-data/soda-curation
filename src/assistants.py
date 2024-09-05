"""
Assistants Module

This module handles interactions with AI assistants, specifically the Anthropic API.
It processes file lists and generates structured data about manuscripts and figures.
"""

import os
import json
from anthropic import Anthropic
from typing import List, Dict, Any

def process_file_list(file_list: List[str], api_key: str) -> Dict[str, Any]:
    """
    Process a list of files using the Anthropic API to identify and categorize them.

    Args:
        file_list (List[str]): A list of file paths to process.
        api_key (str): The Anthropic API key.

    Returns:
        Dict[str, Any]: A dictionary containing the structured data about the manuscript and figures.

    Raises:
        Exception: If the API call fails or the response cannot be parsed.
    """
    client = Anthropic(api_key=api_key)

    prompt = "Here's a list of files:\n\n"
    prompt += "\n".join(file_list)
    prompt += "\n\nPlease analyze these files and categorize them into manuscript files, figures, and source data files. "
    prompt += "Provide the output in JSON format following this schema:\n"
    prompt += """
    {
      "manuscript": {
        "id_": "JOURNAL-YYYY-XXXXX",
        "files": [
          "/path/to/manuscript1.docx",
          "/path/to/manuscript2.pdf"
        ],
        "xml": "/path/to/manuscript1.xml",
        "figures": [
          {
            "figure_label": "Figure X",
            "img_file": ["/path/to/figureX.tif"],
            "sd_file": ["/path/to/figureX-sd.zip"]
          }
        ]
      }
    }
    """
    prompt += "\nMake sure to include all relevant files and categorize them correctly."
    prompt += "\nReturn **ONLY** the `json` code"
    prompt += "\nIf no files are found or if the list is empty, please return an empty JSON object: {}"

    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from the response
        json_str = response.content[0].text.strip()
        
        # Parse the JSON
        result = json.loads(json_str)
        
        # If the result is an empty object, return it as is
        if not result:
            return result
        
        # Validate the result structure
        if "manuscript" not in result or "figures" not in result["manuscript"]:
            raise ValueError("Invalid response structure from the AI assistant")
        
        return result
    except json.JSONDecodeError:
        # If JSON parsing fails, return an empty JSON object
        return {}
    except Exception as e:
        raise Exception(f"Error processing files with Anthropic API: {str(e)}")

def get_file_structure(file_list: List[str]) -> Dict[str, Any]:
    """
    Process the file list and return the structured data about the manuscript and figures.

    Args:
        file_list (List[str]): A list of file paths to process.

    Returns:
        Dict[str, Any]: A dictionary containing the structured data about the manuscript and figures.
                        Returns an empty dict if processing fails or no files are found.

    Raises:
        Exception: If the API key is not set.
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY environment variable is not set")

    try:
        result = process_file_list(file_list, api_key)
        return result
    except Exception as e:
        print(f"Warning: Failed to process file list: {str(e)}")
        return {}