"""
Assistants Module

This module handles interactions with AI assistants, specifically the Anthropic API.
It processes file lists, generates structured data about manuscripts and figures,
and extracts figure captions from manuscript files.
"""

import os
import json
from anthropic import Anthropic
from typing import List, Dict, Any
import docx
from docx.opc.exceptions import PackageNotFoundError
import PyPDF2
import traceback
import re

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
    if not file_list:
        return {}

    client = Anthropic(api_key=api_key)

    prompt = "Here's a list of files:\n\n"
    prompt += "\n".join(file_list)
    prompt += "\n\nPlease analyze these files and categorize them into manuscript files, XML file, figures, and source data files. "
    prompt += "Provide the output in JSON format following this schema:\n"
    prompt += """
    {
      "manuscript": {
        "id_": "JOURNAL-YYYY-XXXXX",
        "files": [
          "/path/to/manuscript1.docx",
          "/path/to/manuscript2.pdf"
        ],
        "xml": "/path/to/manuscript.xml",
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
        
        return result
    except Exception as e:
        raise Exception(f"Error processing files with Anthropic API: {str(e)}")

def extract_figure_captions(manuscript_files: List[str], api_key: str, base_path: str = "") -> Dict[str, str]:
    """
    Extract figure captions from manuscript files using the Anthropic API.
    Prioritizes DOCX files and falls back to PDF if necessary.

    Args:
        manuscript_files (List[str]): A list of paths to manuscript files (.docx or .pdf).
        api_key (str): The Anthropic API key.
        base_path (str): The base path to prepend to relative file paths.

    Returns:
        Dict[str, str]: A dictionary mapping figure labels to their captions.
    """
    client = Anthropic(api_key=api_key)
    captions = {}

    def extract_from_file(file_path):
        full_path = os.path.join(base_path, file_path)
        print(f"Attempting to extract from file: {full_path}")
        
        if not os.path.exists(full_path):
            print(f"File does not exist: {full_path}")
            return None

        file_content = ""
        try:
            if file_path.endswith('.docx'):
                try:
                    doc = docx.Document(full_path)
                    paragraphs = [para.text for para in doc.paragraphs]
                    # Find paragraphs that likely contain figure captions
                    figure_paragraphs = [p for p in paragraphs if re.match(r'^Figure\s+\d+', p, re.IGNORECASE)]
                    file_content = "\n\n".join(figure_paragraphs)
                except PackageNotFoundError:
                    print(f"Error: Unable to open DOCX file. It may be corrupted or not a valid DOCX file: {full_path}")
                    return None
                except Exception as e:
                    print(f"Error reading DOCX file {full_path}: {str(e)}")
                    print(traceback.format_exc())
                    return None
            elif file_path.endswith('.pdf'):
                try:
                    with open(full_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        file_content = "\n".join([page.extract_text() for page in reader.pages])
                except Exception as e:
                    print(f"Error reading PDF file {full_path}: {str(e)}")
                    print(traceback.format_exc())
                    return None
        except Exception as e:
            print(f"Error extracting content from {full_path}: {str(e)}")
            print(traceback.format_exc())
            return None

        if not file_content:
            print(f"No content extracted from {full_path}")
            return None

        # prompt = f"Extract all figure captions from the following text. Return the result as a JSON object where keys are figure labels (e.g., 'Figure 1') and values are the corresponding captions:\n\n{file_content}"

        prompt = """You are an AI assistant specialized in analyzing scientific manuscripts. Your task is to extract figure captions from the given text, which is typically from a scientific paper. Please follow these guidelines:

        1. Identify all figure captions in the text. These usually start with "Figure X" or "Fig. X", where X is a number.
        
        2. Each caption is compound by a figure title and a figure caption. Please extract both parts all together, including the description of all the panels composing the figure.

        3. Create a JSON object where:
        - Keys are the figure labels (e.g., "Figure 1", "Figure 2")
        - Values are the corresponding complete captions, including all text up to the next figure caption or the end of the caption section

        4. Preserve all formatting, such as line breaks, within the caption text.

        5. If there are no figures or captions in the text, return an empty JSON object.

        6. Do not include any explanations or additional text outside of the JSON object in your response.

        Here's an example of the expected output format:

        {
        "Figure 1": "Title figure 1. Caption for Figure 1.",
        "Figure 2": "Title figure 2. Caption for Figure 2.",
        "Figure 3": "Title figure 3. Caption for Figure 3."
        }

        Please process the given text and return the JSON object with the extracted figure captions.
        """
        prompt += f"\n\n{file_content}"

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            extracted_captions = json.loads(response.content[0].text)
            return extracted_captions
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for {full_path}: {str(e)}")
            print(f"Raw response: {response.content[0].text}")
            return None
        except Exception as e:
            print(f"Warning: Failed to extract captions from {full_path}: {str(e)}")
            print(traceback.format_exc())
            return None

    # Try DOCX files first
    docx_files = [f for f in manuscript_files if f.endswith('.docx')]
    for docx_file in docx_files:
        result = extract_from_file(docx_file)
        if result:
            return result  # Return the first successful extraction

    # If no successful extraction from DOCX, try PDF files
    pdf_files = [f for f in manuscript_files if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        result = extract_from_file(pdf_file)
        if result:
            return result  # Return the first successful extraction

    # If no successful extraction at all, return an empty dictionary
    return {}

def get_file_structure(file_list: List[str], base_path: str = "") -> Dict[str, Any]:
    """
    Process the file list, generate structured data about the manuscript and figures,
    and extract figure captions.

    Args:
        file_list (List[str]): A list of file paths to process.
        base_path (str): The base path to prepend to relative file paths.

    Returns:
        Dict[str, Any]: A dictionary containing the structured data about the manuscript,
                        figures, and their captions.

    Raises:
        Exception: If the API key is not set.
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable is not set")
        return {}

    try:
        # Get initial file structure
        print("Calling process_file_list...")
        result = process_file_list(file_list, api_key)
        print(f"Result from process_file_list: {json.dumps(result, indent=2)}")
        
        if not result:
            print("process_file_list returned an empty result")
            return {}
        
        if 'manuscript' not in result:
            print("'manuscript' key not found in the result from process_file_list")
            return {}
        
        # Extract figure captions
        manuscript_files = result['manuscript'].get('files', [])
        print(f"Manuscript files for caption extraction: {manuscript_files}")
        
        if not manuscript_files:
            print("No manuscript files found for caption extraction")
            return result  # Return the result without captions
        
        captions = extract_figure_captions(manuscript_files, api_key, base_path)
        print(f"Extracted captions: {json.dumps(captions, indent=2)}")
        
        # Update the result dictionary with figure captions
        for figure in result['manuscript'].get('figures', []):
            figure_label = figure.get('figure_label')
            if figure_label in captions:
                figure['figure_caption'] = captions[figure_label]
            else:
                print(f"No caption found for {figure_label}")

        print(f"Final result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"Exception in get_file_structure: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {}
