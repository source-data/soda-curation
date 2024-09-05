"""
ZIP Processor Module

This module contains functions for processing ZIP files,
specifically for finding and extracting contents of the eJP folder.
"""

import os
import zipfile
import shutil

def process_zip_file(zip_path, extract_dir):
    """
    Process the ZIP file to find the eJP folder and list its contents.

    Args:
        zip_path (str): Path to the ZIP file.
        extract_dir (str): Directory to extract files to.

    Returns:
        tuple: (list of files in the eJP folder, path to the extracted eJP files)
               or (None, None) if not found.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all contents to a temporary directory
        zip_ref.extractall(extract_dir)

    # Search for the eJP folder
    ejp_folder = None
    for root, dirs, files in os.walk(extract_dir):
        if 'eJP' in dirs:
            ejp_folder = os.path.join(root, 'eJP')
            break

    if ejp_folder:
        # List all files in the eJP folder
        ejp_files = [f for f in os.listdir(ejp_folder) if os.path.isfile(os.path.join(ejp_folder, f))]
        
        # Create a new directory to store eJP files for downstream processing
        ejp_process_dir = os.path.join(extract_dir, 'ejp_files')
        os.makedirs(ejp_process_dir, exist_ok=True)

        # Copy eJP files to the new directory
        for file in ejp_files:
            shutil.copy2(os.path.join(ejp_folder, file), ejp_process_dir)

        return ejp_files, ejp_process_dir
    else:
        return None, None
