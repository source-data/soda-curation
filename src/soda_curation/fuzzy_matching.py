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
docx_path = 'data/EMM-2023-18636/doc/240429 EMM  DUSP6 targeting abrogates HER3 R2.docx'
ai_extracted_text = """

Clinical association of DUSP6 overexpression with poor prognosis HER2+ breast cancers. (A, B) Volcano plots visualizing differentially expressed genes in (A) Control-DTP and (B) DTP-DTEP transitions. The volcano blots indicate all genes that were significantly regulated during these transitions (|logFC |\\\\u003c2 and FDR \\\\u003c0.05), whereas only the phosphatase genes among these are indicated by names. The four phosphatase genes significantly regulated in both transitions (DUSP6, CDC25A, CDC25C, and SYNJ1) are indicated in bold. Differentially expressed genes were identified using the R package limma (n=3). (C) Changes in the DUSP6, CDC25A, CDC25C, and SYNJ1 mRNA levels during the acquisition of lapatinib resistance in BT474 cells. Data is based on RNA sequencing analysis (Dataset EV1) and was analyzed by one-way ANOVA followed by Tukey\\'s multiple comparisons test. Statistically significant values of *p \\\\u003c 0.05, **p \\\\u003c 0.01 and ***p \\\\u003c 0.001 were determined (n=3). (D) Differential expression of DUSP6, CDC25A, CDC25C, and SYNJ1 in different breast cancer subtypes. Data were extracted from the METABRIC dataset and categorized into five molecular subtypes according to the PAM50 gene expression subtype classification (basal, claudin-low, HER2+, Luminal A, and Luminal B). Data were analyzed by one-way ANOVA followed by Tukey\\'s multiple comparisons test and shown as mean standard deviation (SD). Statistically significant values of *p \\\\u003c 0.05, **p \\\\u003c 0.01 and ****p \\\\u003c 0.0001 were determined (basal=209, claudin-low=218, HER2+=224, LumA=700 and LumB=475). (E) Breast cancer patients from the TCGA-BRCA dataset were divided into DUSP6 high (LogFC\\\\u003e1, FDR\\\\u003c0.05) and low expression (LogFC\\\\u003c-1, FDR\\\\u003c0.05) groups and the clinical breast cancer subtypes were compared between the two groups. NA; not available. (F,G) Subgroup of 113 patient cases with high tumor ERBB2 mRNA expression (LogFC\\\\u003e1, FDR\\\\u003c0.05) were divided into DUSP6high and DUSP6low groups and their overall survival (OS) (G) (Log-rank Test p value=0.0220) and disease-specific progression free survival (PFS)(H)(Log-rank Test p value=0.0259) was tested according to DUSP6 status.

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
