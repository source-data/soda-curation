"""
SODA Curation Tool

This Streamlit app provides a simple interface for uploading and processing ZIP files.
It's designed to run in a Docker container with NVIDIA GPU support.
"""

import streamlit as st
import os
import tempfile
import hashlib

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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.text(f"File Details: {file_details}")
        
        st.success(f"File {uploaded_file.name} is successfully uploaded")
        st.info(f"Temporary file path: {tmp_file_path}")

        # TODO: Process the file
        # results = process_file(tmp_file_path)

        st.info("File processed successfully.")

        # Clean up the temporary file
        os.unlink(tmp_file_path)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    st.info("Processing functionality to be implemented.")

if __name__ == "__main__":
    main()
