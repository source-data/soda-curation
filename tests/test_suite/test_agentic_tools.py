"""Tests for the agentic tools module."""

import string

from src.soda_curation.agentic_tools import (
    _int_to_roman,
    _is_roman_numeral,
    _roman_to_int,
    _verify_alphabet_sequence,
)
from src.soda_curation.agentic_tools import (
    _verify_caption_extraction_impl as verify_caption_extraction,
)
from src.soda_curation.agentic_tools import _verify_numeric_sequence
from src.soda_curation.agentic_tools import (
    _verify_panel_sequence_impl as verify_panel_sequence,
)
from src.soda_curation.agentic_tools import _verify_roman_sequence, normalize_text


class TestVerifyCaptionExtraction:
    """Test the verify_caption_extraction function."""

    def test_empty_texts(self):
        """Test with empty texts."""
        result = verify_caption_extraction("", "")
        assert result["is_verbatim"] is False
        assert "empty" in result["details"].lower()

    def test_exact_match(self):
        """Test with exactly matching texts."""
        original = "This is a test caption."
        extracted = "This is a test caption."
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is True

    def test_extracted_in_original(self):
        """Test with extracted text contained in original."""
        original = "Figure 1: This is a test caption. Figure 2: Another caption."
        extracted = "This is a test caption."
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is True

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        original = "This is a TEST caption."
        extracted = "this is a test caption."
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is True

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        original = "This  is\ta\ntest caption."
        extracted = "This is a test caption."
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is True

    def test_html_handling(self):
        """Test HTML handling."""
        original = "<p>This is a <strong>test</strong> caption.</p>"
        extracted = "This is a test caption."
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is True

    def test_html_entities(self):
        """Test HTML entity handling."""
        original = "This &amp; that"
        extracted = "This & that"
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is True

    def test_punctuation_handling(self):
        """Test punctuation handling."""
        original = "This is a test caption!"
        extracted = "This is a test caption"
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is True

    def test_partial_match_long_text(self):
        """Test with long text that should NOT be considered a verbatim match."""
        # Create a long text with distinct start and end
        original = "Start of a very long text " + "middle " * 100 + "end of the text."
        # Extract just the start and end, skipping the middle
        extracted = "Start of a very long text end of the text."

        # This should fail with our strict matching criteria
        result = verify_caption_extraction(extracted, original)

        # We expect this NOT to be verbatim since the extracted text skips content
        assert result["is_verbatim"] is False

    def test_completely_different(self):
        """Test completely different texts."""
        original = "This is the original text."
        extracted = "This is completely different."
        result = verify_caption_extraction(extracted, original)
        assert result["is_verbatim"] is False

    def test_complex_html_parsing(self):
        """Test parsing of complex HTML structures."""
        # Complex HTML with nested tags and attributes
        complex_html = """
        <div class="caption">
            <h3>Figure 1: The experiment setup</h3>
            <p>The experiment was conducted using <em>three</em> different 
            <span style="color: red;">configurations</span>:</p>
            <ol>
                <li>Configuration A</li>
                <li>Configuration B</li>
                <li>Configuration C</li>
            </ol>
        </div>
        """

        # Plain text that should be considered a match after HTML parsing
        extracted_text = "Figure 1: The experiment setup The experiment was conducted using three different configurations: Configuration A Configuration B Configuration C"

        result = verify_caption_extraction(extracted_text, complex_html)
        assert result["is_verbatim"] is True

    def test_malformed_html(self):
        """Test parsing of malformed HTML."""
        # Malformed HTML with unclosed tags
        malformed_html = """
        <p>This is <strong>malformed HTML with <em>unclosed tags
        and <span>overlapping elements</strong> that should still be parseable</span>
        """

        extracted_text = "This is malformed HTML with unclosed tags and overlapping elements that should still be parseable"

        result = verify_caption_extraction(extracted_text, malformed_html)
        assert result["is_verbatim"] is True

    def test_html_with_tables(self):
        """Test parsing of HTML containing tables."""
        # HTML with table structure
        table_html = """
        <table>
            <caption>Figure 2: Experimental results</caption>
            <thead>
                <tr>
                    <th>Configuration</th>
                    <th>Result A</th>
                    <th>Result B</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Config 1</td>
                    <td>10.5</td>
                    <td>8.3</td>
                </tr>
                <tr>
                    <td>Config 2</td>
                    <td>12.7</td>
                    <td>9.1</td>
                </tr>
            </tbody>
        </table>
        """

        # Plain text that contains the table content
        extracted_text = "Figure 2: Experimental results Configuration Result A Result B Config 1 10.5 8.3 Config 2 12.7 9.1"

        result = verify_caption_extraction(extracted_text, table_html)
        assert result["is_verbatim"] is True

    def test_html_with_special_elements(self):
        """Test parsing of HTML with special elements like script, style, comments."""
        # HTML with elements that shouldn't be included in the text
        special_html = """
        <div>
            <p>Figure 3: Network architecture</p>
            <!-- This is a comment that should be ignored -->
            <style>
                .diagram { border: 1px solid black; }
            </style>
            <script>
                function showDetails() {
                    alert('Details shown!');
                }
            </script>
            <p>The architecture consists of three layers.</p>
        </div>
        """

        extracted_text = (
            "Figure 3: Network architecture The architecture consists of three layers."
        )

        result = verify_caption_extraction(extracted_text, special_html)
        assert result["is_verbatim"] is True

    def test_html_with_entities_and_unicode(self):
        """Test parsing of HTML with HTML entities and Unicode characters."""
        # HTML with entities and Unicode
        entity_html = """
        <p>Figure 4: Reaction of A &rarr; B</p>
        <p>The reaction occurs at 37&deg;C and pH &gt; 7.0</p>
        <p>Efficiency: 95&#37; &plusmn; 2&#37;</p>
        """

        extracted_text = "Figure 4: Reaction of A → B The reaction occurs at 37°C and pH > 7.0 Efficiency: 95% ± 2%"

        result = verify_caption_extraction(extracted_text, entity_html)
        assert result["is_verbatim"] is True


class TestNormalizeText:
    """Test the normalize_text function."""

    def test_html_removal(self):
        """Test HTML tag removal."""
        html_text = "<p>This is <strong>bold</strong> text</p>"
        normalized = normalize_text(html_text)
        assert "this is bold text" in normalized
        assert "<" not in normalized
        assert ">" not in normalized

    def test_entity_decoding(self):
        """Test HTML entity decoding."""
        entity_text = "This &amp; that &lt; the other"
        normalized = normalize_text(entity_text)
        assert "this" in normalized
        assert "that" in normalized
        assert "the other" in normalized
        assert "&amp;" not in normalized
        assert "&lt;" not in normalized

    def test_case_normalization(self):
        """Test case normalization."""
        mixed_case = "This Is MIXED case"
        normalized = normalize_text(mixed_case)
        assert normalized == "this is mixed case"

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        messy_whitespace = "This  has \t\nmultiple    spaces"
        normalized = normalize_text(messy_whitespace)
        assert normalized == "this has multiple spaces"

    def test_unicode_normalization(self):
        """Test unicode normalization."""
        # Using regular quotes instead of smart quotes
        unicode_text = """This has "smart quotes" and em\u2014dash"""  # Using Unicode escape for em-dash
        normalized = normalize_text(unicode_text)
        assert "smart quotes" in normalized
        assert "emdash" in normalized


class TestVerifyPanelSequence:
    """Test the verify_panel_sequence function."""

    def test_empty_sequence(self):
        """Test with empty sequence."""
        result = verify_panel_sequence([])
        assert result["is_valid"] is False
        assert "no panel labels" in result["details"].lower()

    def test_valid_uppercase_sequence(self):
        """Test valid uppercase letter sequence."""
        result = verify_panel_sequence(["A", "B", "C", "D"])
        assert result["is_valid"] is True
        assert result["fixed_sequence"] == ["A", "B", "C", "D"]

    def test_invalid_uppercase_sequence(self):
        """Test invalid uppercase letter sequence with gaps."""
        result = verify_panel_sequence(["A", "B", "D", "E"])
        assert result["is_valid"] is False
        assert result["fixed_sequence"] == ["A", "B", "C", "D", "E"]

    def test_valid_lowercase_sequence(self):
        """Test valid lowercase letter sequence."""
        result = verify_panel_sequence(["a", "b", "c"])
        assert result["is_valid"] is True
        assert result["fixed_sequence"] == ["a", "b", "c"]

    def test_invalid_lowercase_sequence(self):
        """Test invalid lowercase letter sequence with gaps."""
        result = verify_panel_sequence(["a", "c", "e"])
        assert result["is_valid"] is False
        assert result["fixed_sequence"] == ["a", "b", "c", "d", "e"]

    def test_valid_roman_numeral_sequence(self):
        """Test valid Roman numeral sequence."""
        result = verify_panel_sequence(["I", "II", "III", "IV"])
        assert result["is_valid"] is True
        assert result["fixed_sequence"] == ["I", "II", "III", "IV"]

    def test_invalid_roman_numeral_sequence(self):
        """Test invalid Roman numeral sequence with gaps."""
        result = verify_panel_sequence(["I", "III", "V"])
        assert result["is_valid"] is False
        assert "II" in result["fixed_sequence"]
        assert "IV" in result["fixed_sequence"]

    def test_valid_numeric_sequence(self):
        """Test valid numeric sequence."""
        result = verify_panel_sequence(["1", "2", "3", "4"])
        assert result["is_valid"] is True
        assert result["fixed_sequence"] == ["1", "2", "3", "4"]

    def test_invalid_numeric_sequence(self):
        """Test invalid numeric sequence with gaps."""
        result = verify_panel_sequence(["1", "3", "5"])
        assert result["is_valid"] is False
        assert result["fixed_sequence"] == ["1", "2", "3", "4", "5"]

    def test_sequence_with_decorators(self):
        """Test sequence with decorators like parentheses."""
        result = verify_panel_sequence(["(A)", "B.", "C)"])
        assert result["is_valid"] is True
        assert result["fixed_sequence"] == ["A", "B", "C"]


class TestRomanNumerals:
    """Test the Roman numeral helper functions."""

    def test_is_roman_numeral(self):
        """Test the function to identify Roman numerals."""
        assert _is_roman_numeral("I") is True
        assert _is_roman_numeral("IV") is True
        assert _is_roman_numeral("MCMXCIV") is True
        assert _is_roman_numeral("A") is False
        assert _is_roman_numeral("123") is False
        assert _is_roman_numeral("IIII") is False  # Invalid Roman numeral

    def test_roman_to_int(self):
        """Test conversion from Roman numerals to integers."""
        assert _roman_to_int("I") == 1
        assert _roman_to_int("IV") == 4
        assert _roman_to_int("V") == 5
        assert _roman_to_int("IX") == 9
        assert _roman_to_int("X") == 10
        assert _roman_to_int("XIV") == 14
        assert _roman_to_int("MCMXCIV") == 1994
        assert _roman_to_int("INVALID") == -1  # Invalid Roman numeral

    def test_int_to_roman(self):
        """Test conversion from integers to Roman numerals."""
        assert _int_to_roman(1) == "I"
        assert _int_to_roman(4) == "IV"
        assert _int_to_roman(5) == "V"
        assert _int_to_roman(9) == "IX"
        assert _int_to_roman(10) == "X"
        assert _int_to_roman(14) == "XIV"
        assert _int_to_roman(1994) == "MCMXCIV"


class TestVerifySpecificSequences:
    """Test the specific sequence verification helper functions."""

    def test_verify_alphabet_sequence(self):
        """Test verification of alphabetical sequences."""
        # Valid sequence
        result = _verify_alphabet_sequence(["A", "B", "C"], string.ascii_uppercase)
        assert result["is_valid"] is True

        # Invalid sequence with gaps
        result = _verify_alphabet_sequence(["A", "C", "E"], string.ascii_uppercase)
        assert result["is_valid"] is False
        assert result["fixed_sequence"] == ["A", "B", "C", "D", "E"]

        # Invalid sequence with characters not in alphabet
        result = _verify_alphabet_sequence(["A", "B", "1"], string.ascii_uppercase)
        assert result["is_valid"] is False
        assert "not in the expected alphabet" in result["details"]

    def test_verify_roman_sequence(self):
        """Test verification of Roman numeral sequences."""
        # Valid sequence
        result = _verify_roman_sequence(["I", "II", "III"])
        assert result["is_valid"] is True

        # Invalid sequence with gaps
        result = _verify_roman_sequence(["I", "III", "V"])
        assert result["is_valid"] is False
        assert "II" in result["fixed_sequence"]
        assert "IV" in result["fixed_sequence"]

        # Invalid Roman numeral
        result = _verify_roman_sequence(["I", "II", "IIII"])
        assert "Invalid" in result["details"]

    def test_verify_numeric_sequence(self):
        """Test verification of numeric sequences."""
        # Valid sequence
        result = _verify_numeric_sequence(["1", "2", "3"])
        assert result["is_valid"] is True

        # Invalid sequence with gaps
        result = _verify_numeric_sequence(["1", "3", "5"])
        assert result["is_valid"] is False
        assert result["fixed_sequence"] == ["1", "2", "3", "4", "5"]

        # Invalid numeric value
        result = _verify_numeric_sequence(["1", "2", "not_a_number"])
        assert "Invalid" in result["details"]
