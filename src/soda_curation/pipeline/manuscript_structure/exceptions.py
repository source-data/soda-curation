class NoXMLFileFoundError(Exception):
    """Exception raised when no XML file is found in the input ZIP file."""
    pass

class NoManuscriptFileError(Exception):
    """Exception raised when no PDF or DOCX manuscript file is found."""
    pass
