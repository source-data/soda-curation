"""
SODA Curation Tool

This Streamlit app provides a simple interface for uploading and processing ZIP files.
It's designed to run in a Docker container with NVIDIA GPU support.
"""

import streamlit as st
import os
import tempfile
import hashlib
import json
from src.zip_processor import process_zip_file
from src.assistants import get_file_structure

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="SODA Curation Tool", layout="wide")

    st.title("SODA Curation Tool")

    st.header("Upload ZIP File")

    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")

    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)

def process_uploaded_file(uploaded_file):
    """
    Process the uploaded ZIP file.

    Args:
        uploaded_file: The uploaded file object from Streamlit.
    """
    try:
        # Calculate file hash
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

        # Process the new file
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.text(f"File Details: {file_details}")
            
            st.success(f"File {uploaded_file.name} is successfully uploaded")
            st.info(f"Temporary directory path: {tmp_dir}")

            # Process the ZIP file
            ejp_files, ejp_process_dir = process_zip_file(zip_path, tmp_dir)

            if ejp_files:
                st.success(f"Found {len(ejp_files)} files in the eJP folder:")
                for file in ejp_files:
                    st.text(file)
                st.info(f"eJP files extracted to: {ejp_process_dir}")

                # Process the file list using the Anthropic API
                file_structure = get_file_structure(ejp_files)
                if file_structure:
                    st.success("File structure processed successfully:")
                    st.json(file_structure)
                    # Add this line for debugging
                    st.text(f"Debug: Raw JSON output: {json.dumps(file_structure)}")
                else:
                    st.warning("No valid file structure could be determined. Please check the uploaded files.")
            else:
                st.warning("No eJP folder found in the ZIP file.")

        st.info("File processed successfully.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()