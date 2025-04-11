"""
Tools for OpenAI Agent functions in the SODA curation pipeline.
"""
import html
import re
import string
import unicodedata
from typing import List

from agents import function_tool
from bs4 import BeautifulSoup


# Create function tools for verification
@function_tool
def verify_caption_extraction(extracted_text: str, original_text: str) -> dict:
    """
    Verify if the extracted caption is present in the original text.
    This function normalizes both texts to handle HTML tags, whitespace,
    special characters, and other formatting differences.

    Args:
        extracted_text (str): The extracted caption to verify
        original_text (str): The original text containing all captions

    Returns:
        dict: Result with is_verbatim status and details
    """
    # Skip verification if either text is empty
    if not extracted_text or not original_text:
        return {"is_verbatim": False, "details": "One or both texts are empty"}

    # Apply a series of normalization steps to both texts
    normalized_extracted = normalize_text(extracted_text)
    normalized_original = normalize_text(original_text)

    # First, try exact matching with normalized text
    is_verbatim = normalized_extracted in normalized_original

    # If exact matching fails, try a more lenient approach
    if not is_verbatim:
        # Try to find the first 100 and last 100 characters to check for partial match
        if len(normalized_extracted) > 200:
            start_text = normalized_extracted[:100]
            end_text = normalized_extracted[-100:]

            is_start_match = start_text in normalized_original
            is_end_match = end_text in normalized_original

            is_verbatim = is_start_match and is_end_match

    return {
        "is_verbatim": is_verbatim,
        "details": "The extraction is verbatim"
        if is_verbatim
        else "The extraction is NOT verbatim",
    }


def normalize_text(text: str) -> str:
    """
    Apply a series of normalization steps to make text comparison more robust.

    Args:
        text (str): The text to normalize

    Returns:
        str: Normalized text
    """
    # Step 1: Parse HTML and extract text
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
    except Exception as error:
        # If BeautifulSoup fails, try to remove HTML tags with regex
        print(error)
        text = re.sub(r"<[^>]+>", "", text)

    # Step 2: Decode HTML entities
    text = html.unescape(text)

    # Step 3: Convert to lowercase
    text = text.lower()

    # Step 4: Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Step 5: Remove non-alphanumeric characters except spaces
    text = re.sub(r"[^\w\s]", "", text)

    # Step 6: Normalize whitespace (replace multiple spaces, tabs, newlines with a single space)
    text = re.sub(r"\s+", " ", text)

    # Step 7: Remove leading/trailing whitespace
    text = text.strip()

    # Step 8: Remove any remaining special characters and punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


@function_tool
def verify_panel_sequence(panel_labels: List[str]) -> dict:
    """
    Verify if a sequence of panel labels follows a monotonically increasing sequence
    without gaps (e.g. A, B, C, D, E instead of A, B, D, E).

    Supports various common panel label systems:
    - Uppercase letters (A, B, C...)
    - Lowercase letters (a, b, c...)
    - Roman numerals (I, II, III...)
    - Arabic numerals (1, 2, 3...)

    Args:
        panel_labels (List[str]): List of panel labels to verify

    Returns:
        dict: Result with is_valid status, fixed_sequence if needed, and details
    """
    if not panel_labels:
        return {
            "is_valid": False,
            "fixed_sequence": [],
            "details": "No panel labels provided",
        }

    # Ensure panel labels are stripped of any decorators
    clean_labels = [re.sub(r"[().\s]", "", label.strip()) for label in panel_labels]

    # Determine the type of sequence based on the first label
    first_label = clean_labels[0]

    # Check for uppercase letters (most common)
    if first_label in string.ascii_uppercase:
        return _verify_alphabet_sequence(clean_labels, string.ascii_uppercase)

    # Check for lowercase letters
    elif first_label in string.ascii_lowercase:
        return _verify_alphabet_sequence(clean_labels, string.ascii_lowercase)

    # Check for Roman numerals
    elif _is_roman_numeral(first_label):
        return _verify_roman_sequence(clean_labels)

    # Check for Arabic numerals
    elif first_label.isdigit():
        return _verify_numeric_sequence(clean_labels)

    # If we can't determine the type
    return {
        "is_valid": False,
        "fixed_sequence": clean_labels,
        "details": "Unable to determine sequence type from labels",
    }


def _verify_alphabet_sequence(labels: List[str], alphabet: str) -> dict:
    """Helper function to verify and fix alphabetical sequences."""
    # Find the start and end positions in the alphabet
    try:
        start_idx = alphabet.index(labels[0])
        end_idx = alphabet.index(labels[-1])
    except ValueError:
        return {
            "is_valid": False,
            "fixed_sequence": labels,
            "details": "Labels contain characters not in the expected alphabet",
        }

    # Generate the expected complete sequence
    expected_sequence = list(alphabet[start_idx : end_idx + 1])

    # Check if the provided sequence matches the expected sequence
    is_valid = len(labels) == len(expected_sequence) and all(
        a == b for a, b in zip(labels, expected_sequence)
    )

    return {
        "is_valid": is_valid,
        "fixed_sequence": expected_sequence,
        "details": "Sequence is valid"
        if is_valid
        else f"Sequence has gaps. Fixed sequence: {', '.join(expected_sequence)}",
    }


def _is_roman_numeral(s: str) -> bool:
    """Check if a string is a valid Roman numeral."""
    return bool(
        re.match(
            r"^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
            s,
            re.IGNORECASE,
        )
    )


def _roman_to_int(roman: str) -> int:
    """Convert a Roman numeral to an integer."""
    roman = roman.upper()
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0

    for char in reversed(roman):
        if char not in values:
            return -1  # Invalid character

        current = values[char]
        if current >= prev:
            total += current
        else:
            total -= current
        prev = current

    return total


def _int_to_roman(num: int) -> str:
    """Convert an integer to a Roman numeral."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0

    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1

    return roman_num


def _verify_roman_sequence(labels: List[str]) -> dict:
    """Helper function to verify and fix Roman numeral sequences."""
    # Convert Roman numerals to integers
    try:
        nums = [_roman_to_int(label) for label in labels]
        if -1 in nums:  # Check for invalid conversions
            return {
                "is_valid": False,
                "fixed_sequence": labels,
                "details": "Invalid Roman numeral in sequence",
            }
    except Exception as e:
        return {
            "is_valid": False,
            "fixed_sequence": labels,
            "details": f"Error processing Roman numerals: {e}",
        }

    # Generate the expected complete sequence
    start = nums[0]
    end = nums[-1]
    expected_nums = list(range(start, end + 1))
    expected_sequence = [_int_to_roman(num) for num in expected_nums]

    # Check if the provided sequence matches the expected sequence
    is_valid = len(nums) == len(expected_nums) and all(
        a == b for a, b in zip(nums, expected_nums)
    )

    return {
        "is_valid": is_valid,
        "fixed_sequence": expected_sequence,
        "details": "Sequence is valid"
        if is_valid
        else f"Sequence has gaps. Fixed sequence: {', '.join(expected_sequence)}",
    }


def _verify_numeric_sequence(labels: List[str]) -> dict:
    """Helper function to verify and fix numeric sequences."""
    # Convert string numbers to integers
    try:
        nums = [int(label) for label in labels]
    except ValueError:
        return {
            "is_valid": False,
            "fixed_sequence": labels,
            "details": "Invalid numeric value in sequence",
        }

    # Generate the expected complete sequence
    start = nums[0]
    end = nums[-1]
    expected_nums = list(range(start, end + 1))
    expected_sequence = [str(num) for num in expected_nums]

    # Check if the provided sequence matches the expected sequence
    is_valid = len(nums) == len(expected_nums) and all(
        a == b for a, b in zip(nums, expected_nums)
    )

    return {
        "is_valid": is_valid,
        "fixed_sequence": expected_sequence,
        "details": "Sequence is valid"
        if is_valid
        else f"Sequence has gaps. Fixed sequence: {', '.join(expected_sequence)}",
    }
