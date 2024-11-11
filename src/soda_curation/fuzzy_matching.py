import difflib
import re
import unicodedata

from docx import Document
from fuzzywuzzy import fuzz


# Function to extract text from DOCX file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]  # Exclude empty paragraphs

# Function to normalize text (remove combining characters, unwanted characters, spaces, etc.)
def normalize_text(text):
    # Normalize to NFKD to decompose characters with diacritics into base characters + combining marks
    text = unicodedata.normalize("NFKD", text)
    
    # Remove combining characters (accents, diacritics, etc.)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Replace escape sequences for special characters (like \u003c for <, \u003e for >)
    text = text.encode('utf-8').decode('unicode_escape')
    
    # Normalize slashes: sometimes backslashes are unnecessarily added, so we can replace them with correct symbols
    text = re.sub(r"\\'", "'", text)  # Replace escaped single quotes
    text = re.sub(r'\\"', '"', text)  # Replace escaped double quotes
    text = re.sub(r"\\\\", r"\\", text)  # Replace double backslashes with a single backslash
    
    # Replace newline and tab escape sequences
    text = re.sub(r'\\n|\\t', ' ', text)
    
    # Normalize less-than and greater-than symbols, which might appear encoded
    text = text.replace('\\u003c', '<').replace('\\u003e', '>')
    
    # Remove extra spaces and blank lines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to check if the target text is in the DOCX content using fuzzy matching
def find_fuzzy_match_in_docx(docx_path, target_text, fuzzy_threshold=90):
    # Extract paragraphs from DOCX file
    doc_paragraphs = extract_text_from_docx(docx_path)
    
    # Normalize the target text
    target_text = normalize_text(target_text)
    
    for paragraph in doc_paragraphs:
        # Normalize the paragraph text
        normalized_paragraph = normalize_text(paragraph)
        
        # Perform fuzzy matching
        fuzzy_score = fuzz.ratio(target_text, normalized_paragraph)
        if fuzzy_score >= fuzzy_threshold:
            return True, paragraph, fuzzy_score
    
    return False, None, None

# Function to highlight only the differences between original and extracted text with color
def highlight_differences(original_text, extracted_text):
    # Use SequenceMatcher to get the differences between the two texts
    differ = difflib.SequenceMatcher(None, original_text, extracted_text)
    result = []

    # Iterate over the opcodes to highlight the changes
    for tag, i1, i2, j1, j2 in differ.get_opcodes():
        if tag == 'equal':
            # No change, add as is (in white/no color)
            result.append(original_text[i1:i2])
        elif tag == 'replace':
            # Replacement, highlight original in red and new in green
            result.append(f"\033[91m{original_text[i1:i2]}\033[0m")  # Red for original
            result.append(f"\033[92m{extracted_text[j1:j2]}\033[0m")  # Green for replacement
        elif tag == 'delete':
            # Deletion, highlight in red
            result.append(f"\033[91m{original_text[i1:i2]}\033[0m")  # Red for deleted text
        elif tag == 'insert':
            # Insertion, highlight in green
            result.append(f"\033[92m{extracted_text[j1:j2]}\033[0m")  # Green for inserted text

    # Join the result to form a single string with color coding
    return ''.join(result)

# Path to the uploaded DOCX file and AI-extracted text
docx_path = 'data/MSB-2023-12087/doc/Munck3.docx'
ai_extracted_text = """

Linking metadata sources and digesting them with language models to generate structured outputs and representations of similarity. A) Illustration of Hamming code for error correction in data transmission. Transmission of data (d) and parity (p) bits enables error correction via redundancy (https://en.wikipedia.org/wiki/Hamming_code). B) Diagram showing the different sources of metadata information and how to bundle them. Three independent resources – the electronic labnotebook, the data-associated metadata, and the publication – are shown as redundant entries. An AI language model can be used to extract required and standardized data elements for verification, using codewords as a means of error correction analogous to error correction in communication. C) Heatmap display of similarities between sources by keyword. A Jupyter notebook using GPT-4 has been used to create a structured output in the form of a CSV file, (see Table 1). The digestion of a labnotebook entry, a metadata file server file, and this manuscript are used to check for keywords. The consistency of the keywords across the sources is displayed in a heatmap using the cosine distance for semantic similarity estimation . 

"""

# Run the matching function
found, matched_paragraph, fuzzy_score = find_fuzzy_match_in_docx(docx_path, ai_extracted_text)

# Print the result
print(80*"*")
if found:
    print(f"Match Found: {matched_paragraph}")
    print(f"Fuzzy Score: {fuzzy_score}")
    print("\nDifferences between the original and extracted text:")
    normalized_paragraph = normalize_text(matched_paragraph)
    normalized_extracted_text = normalize_text(ai_extracted_text)
    print(highlight_differences(normalized_paragraph, normalized_extracted_text))
else:
    print("No suitable match found.")
