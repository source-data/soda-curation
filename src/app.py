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
from src.assistants import get_file_structure, process_file_list

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

            # Process the ZIP file
            ejp_files, ejp_process_dir = process_zip_file(zip_path, tmp_dir)

            if ejp_files:
                st.success(f"Found {len(ejp_files)} files in the eJP folder.")
                
                # Debug: Show the list of files found in the eJP folder
                st.subheader("Files found in eJP folder:")
                st.json(ejp_files)

                # Debug: Show the result of the first assistant (file structure determination)
                st.subheader("Initial file structure:")
                api_key = os.getenv('ANTHROPIC_API_KEY')
                initial_structure = process_file_list(ejp_files, api_key)
                st.json(initial_structure)

                # Process the file list using the Anthropic API
                st.subheader("Detailed logs from get_file_structure:")
                import io
                import sys

                # Redirect stdout to capture print statements
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()

                file_structure = get_file_structure(ejp_files, base_path=ejp_process_dir)

                # Restore stdout and get the captured output
                sys.stdout = old_stdout
                debug_output = buffer.getvalue()

                # Display the debug output
                st.text(debug_output)
                
                # Debug: Show the result of get_file_structure
                st.subheader("Result of get_file_structure:")
                st.json(file_structure)

                if file_structure and 'manuscript' in file_structure:
                    st.success("File structure processed successfully.")
                    
                    # Create tabs for different pipeline stages
                    tab1, tab2, tab3 = st.tabs(["File Structure", "Future Step 1", "Future Step 2"])
                    
                    with tab1:
                        st.json(file_structure)
                        
                        # Display figure captions separately for better visibility
                        st.subheader("Figure Captions")
                        for figure in file_structure['manuscript'].get('figures', []):
                            st.write(f"{figure['figure_label']}: {figure.get('figure_caption', 'No caption available')}")
                    
                    with tab2:
                        st.info("This tab will show results from a future pipeline step.")
                    
                    with tab3:
                        st.info("This tab will show results from another future pipeline step.")
                else:
                    st.warning("No valid file structure could be determined. Please check the uploaded files.")
            else:
                st.warning("No eJP folder found in the ZIP file.")

        st.info("File processed successfully.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()