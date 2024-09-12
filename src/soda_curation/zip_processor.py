import zipfile
import json
import os
import re
from typing import Dict, List, Any

def process_zip_file(zip_path: str) -> str:
    """
    Process a ZIP file and return a JSON string with the structure of the data.

    Args:
        zip_path (str): Path to the input ZIP file.

    Returns:
        str: JSON-formatted string containing the structure of the data.

    Raises:
        zipfile.BadZipFile: If the input file is not a valid ZIP file.
        json.JSONDecodeError: If there's an error encoding the results to JSON.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

        print(f"Debug: File list: {file_list}")  # Debug print

        # Extract manuscript_id from the XML file name
        xml_file = next((f for f in file_list if f.endswith('.xml')), "")
        manuscript_id = os.path.splitext(os.path.basename(xml_file))[0] if xml_file else ""
        print(f"Debug: Manuscript ID: {manuscript_id}")  # Debug print

        result: Dict[str, Any] = {
            "manuscript_id": manuscript_id,
            "xml": "",
            "docx": "",
            "pdf": "",
            "appendix": [],
            "figures": []
        }

        # Process files
        for file_path in file_list:
            print(f"Debug: Processing file: {file_path}")  # Debug print
            if file_path.endswith('.xml'):
                result["xml"] = file_path
            elif file_path.lower().endswith('.docx') and 'doc' in file_path.lower():
                result["docx"] = file_path
            elif file_path.lower().endswith('.pdf') and 'pdf' in file_path.lower():
                result["pdf"] = file_path
            
            # Appendix processing with debug output
            if "appendix" in file_path.lower() and file_path.lower().endswith('.pdf'):
                result["appendix"].append(file_path)
                print(f"Debug: Added Appendix: {file_path}")  # Debug print
            elif file_path.lower().endswith('.pdf') and 'suppl_data' in file_path.lower():
                print(f"Debug: Potential Appendix (not added): {file_path}")  # Debug print

            # Process figures
            figure_pattern = re.compile(r'figure(\d+)', re.IGNORECASE)
            if 'graphic' in file_path.lower():
                match = figure_pattern.search(file_path)
                if match:
                    figure_number = match.group(1)
                    figure_label = f"Figure {figure_number}"
                    print(f"Debug: Found figure: {figure_label} in {file_path}")  # Debug print
                    figure_entry = next((fig for fig in result["figures"] if fig["figure_label"] == figure_label), None)
                    if figure_entry is None:
                        figure_entry = {
                            "figure_label": figure_label,
                            "img_files": [],
                            "sd_files": [],
                            "figure_caption": "TO BE ADDED IN LATER STEP",
                            "figure_panels": []
                        }
                        result["figures"].append(figure_entry)
                    figure_entry["img_files"].append(file_path)

            # Process supplementary data
            if 'suppl_data' in file_path.lower():
                for figure in result["figures"]:
                    if figure["figure_label"].lower() in file_path.lower():
                        figure["sd_files"].append(file_path)
                        print(f"Debug: Added supplementary data: {file_path} to figure {figure['figure_label']}")  # Debug print
                        break

        # Sort figures
        result["figures"].sort(key=lambda x: int(re.search(r'\d+', x["figure_label"]).group()))

        return json.dumps(result, indent=2)

    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Invalid ZIP file: {zip_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error encoding results to JSON: {e}")
