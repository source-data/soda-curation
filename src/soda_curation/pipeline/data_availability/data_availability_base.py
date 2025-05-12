"""Base class for extracting data availability information from scientific manuscripts."""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel

from ..manuscript_structure.manuscript_structure import ZipStructure

logger = logging.getLogger(__name__)


class DataSource(BaseModel):
    database: str
    accession_number: str
    url: str


class ExtractDataSources(BaseModel):
    sources: List[DataSource]


class DataAvailabilityExtractor(ABC):
    """
    Abstract base class for extracting data availability information.

    This class defines the interface that all data availability extractors must implement.
    It handles validation of configuration and provides a consistent structure for
    extracting both the data availability section and specific data sources.
    """

    # Database name normalization mapping
    DATABASE_MAPPING = {
        # Standard databases with case variations
        "geo": "Gene Expression Omnibus",
        "gene expression omnibus": "Gene Expression Omnibus",
        "array express": "ArrayExpress",
        "arrayexpress": "ArrayExpress",
        "bioproject": "BioProject",
        "biomodels": "BioModels",
        "biostudies": "BioStudies",
        "bioimage archive": "BioImage Archive",
        "database of genotypes and phenotypes": "Database of Genotypes and Phenotypes",
        "dbgap": "Database of Genotypes and Phenotypes",
        "database of single nucleotide polymorphisms": "Database of single nucleotide polymorphisms",
        "dbsnp": "Database of single nucleotide polymorphisms",
        "dryad": "Dryad",
        "electron microscopy data bank": "Electron Microscopy Data Bank",
        "emdb": "Electron Microscopy Data Bank",
        "european genome-phenome archive": "European Genome-phenome Archive",
        "ega": "European Genome-phenome Archive",
        "european nucleotide archive": "European Nucleotide Archive",
        "ena": "European Nucleotide Archive",
        "figshare": "FigShare",
        "flowrepository": "FlowRepository",
        "github": "Github",
        "glycopost": "GlycoPOST",
        "image data resource": "Image Data Resource",
        "idr": "Image Data Resource",
        "imex": "IMEx",
        "intact": "Molecular Interaction Database",
        "massive": "MassIVE",
        "metabolights": "MetaboLights",
        "molecular interaction database": "Molecular Interaction Database",
        "molecular modeling database": "Molecular Modeling Database",
        "mmdb": "Molecular Modeling Database",
        "peptide atlas": "Peptide Atlas",
        "peptideatlas": "Peptide Atlas",
        "protein data bank": "Protein Data Bank",
        "pdb": "Protein Data Bank",
        "pride": "Proteomics Identification database",
        "proteomics identification database": "Proteomics Identification database",
        "refseq": "Reference Sequence Database",
        "reference sequence database": "Reference Sequence Database",
        "sequence read archive": "Sequence Read Archive",
        "sra": "Sequence Read Archive",
        "sourcedata": "SourceData",
        "zenodo": "Zenodo",
        "empiar": "EMPIAR",
        "encode": "ENCODE",
        "metabolomics workbench": "Metabolomics Workbench Study",
        "mw.study": "Metabolomics Workbench Study",
        "mendeley data": "Mendeley Dataset",
    }

    # URL pattern mapping
    DATABASE_REGEX = {
        "BioStudies": r"^S-[A-Z]{4}[A-Z\d\-]+$",
        "Database of Genotypes and Phenotypes": r"^phs[0-9]{6}(.v\d+.p\d+)?$",
        "Database of single nucleotide polymorphisms": r"^rs\d+$",
        "Electron Microscopy Data Bank": r"^EMD-\d{4,5}$",
        "European Genome-phenome Archive": r"^EGAD\d{11}$",
        "European Nucleotide Archive": r"^[A-Z]+[0-9]+(\.\d+)?$",
        "Gene Expression Omnibus": r"^G(PL|SM|SE|DS)\d+$",
        "Image Data Resource": r"^[0-9]{4}$",
        "Molecular Interaction Database": r"^EBI\-[0-9]+$",
        "Protein Data Bank": r"^[0-9][A-Za-z0-9]{3}$",
        "Proteomics Identification database": r"^P(X|R)D\d{6}$",
        "Reference Sequence Database": r"^(((WP|AC|AP|NC|NG|NM|NP|NR|NT|NW|XM|XP|XR|YP|ZP)_\d+)|(NZ\_[A-Z]{2,4}\d+))(\.\d+)?$",
        "Sequence Read Archive": r"^[SED]R[APRSXZ]\d+$",
        "GlycoPOST": r"^GPST[0-9]{6}$",
        "Molecular Modeling Database": r"^\d{1,5}$",
        "ArrayExpress": r"^[AEP]-\w{4}-\d+$",
        "Peptide Atlas": r"^PASS\d{5}$",
        "FlowRepository": r"^FR\-FCM\-\w{4}$",
        "IMEx": r"^IM-\d+(-?)(\d+?)$",
        "MetaboLights": r"^MTBLS\d+$",
        "BioModels": r"^((BIOMD|MODEL)\d{10})|(BMID\d{12})$",
        "BioProject": r"^PRJ[DEN][A-Z]\d+$",
        "EMPIAR": r"EMPIAR-\d{5,}",
        "MassIVE": r"^MSV\d+$",
        "Metabolomics Workbench Study": r"^ST[0-9]{6}$",
        "ENCODE": r"^ENC[A-Za-z]{2}[0-9]{3}[A-Za-z]{3}$",
    }

    # URL Construction mapping
    URL_MAPPING = {
        "BioStudies": "https://identifiers.org/biostudies:",
        "Database of Genotypes and Phenotypes": "https://identifiers.org/dbgap:",
        "Database of single nucleotide polymorphisms": "https://identifiers.org/dbsnp:",
        "Electron Microscopy Data Bank": "https://identifiers.org/emdb:",
        "European Genome-phenome Archive": "https://identifiers.org/ega.dataset:",
        "European Nucleotide Archive": "https://identifiers.org/ena.embl:",
        "Gene Expression Omnibus": "https://identifiers.org/geo:",
        "Image Data Resource": "https://identifiers.org/idr:",
        "Molecular Interaction Database": "https://identifiers.org/intact:",
        "Protein Data Bank": "https://identifiers.org/pdb:",
        "Proteomics Identification database": "https://identifiers.org/pride.project:",
        "Reference Sequence Database": "https://identifiers.org/refseq:",
        "Sequence Read Archive": "https://identifiers.org/insdc.sra:",
        "GlycoPOST": "https://identifiers.org/glycopost:",
        "Molecular Modeling Database": "https://identifiers.org/mmdb:",
        "ArrayExpress": "https://identifiers.org/arrayexpress:",
        "Peptide Atlas": "https://identifiers.org/peptideatlas.dataset:",
        "FlowRepository": "https://identifiers.org/flowrepository:",
        "IMEx": "https://identifiers.org/imex:",
        "MetaboLights": "https://identifiers.org/metabolights:",
        "BioModels": "https://identifiers.org/biomodels.db:",
        "BioProject": "https://identifiers.org/bioproject:",
        "EMPIAR": "https://identifiers.org/empiar:",
        "MassIVE": "https://identifiers.org/massive:",
        "Metabolomics Workbench Study": "https://identifiers.org/mw.study:",
        "ENCODE": "https://identifiers.org/encode:",
        # DOI prefixed databases
        "Dryad": "https://doi.org/",
        "FigShare": "https://doi.org/",
        "Zenodo": "https://doi.org/",
        "Mendeley Dataset": "https://doi.org/",
        # Other standard databases
        "Github": "https://github.com/",
        "Uniprot": "https://www.uniprot.org/uniprot/",
        "ChEBI": "http://www.ebi.ac.uk/chebi/searchId.do?chebiId=",
        "NCBI gene": "http://www.ncbi.nlm.nih.gov/gene/",
        "NCBI taxon": "http://www.ncbi.nlm.nih.gov/taxonomy/",
        "Uberon": "https://www.ebi.ac.uk/ols/ontologies/uberon/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FUBERON_",
        "Gene Ontology": "http://amigo.geneontology.org/amigo/term/",
        "Rfam": "http://rfam.xfam.org/family/",
        "PubChem": "https://pubchem.ncbi.nlm.nih.gov/compound/",
        "PubChem Substance": "https://pubchem.ncbi.nlm.nih.gov/substance/",
        "Corum": "http://mips.helmholtz-muenchen.de/genre/proj/corum/complexdetails.html?id=",
        "BAO": "https://bioportal.bioontology.org/ontologies/BAO/?p=classes&conceptid=http%3A%2F%2Fwww.bioassayontology.org%2Fbao%23",
        "OBI": "http://www.ontobee.org/ontology/OBI?iri=http://purl.obolibrary.org/obo/",
        "Plant Ontology": "http://browser.planteome.org/amigo/term/",
        "MeSH": "https://www.ncbi.nlm.nih.gov/mesh/?term=",
        "Fairdomhub": "https://fairdomhub.org/data_files/",
        "Disease ontology": "https://identifiers.org/",
        "CVCL": "https://identifiers.org/cellosaurus:",
        "CL": "https://identifiers.org/",
        "SourceData": "https://identifiers.org/biostudies:",
    }

    def __init__(self, config: Dict, prompt_handler=None):
        """
        Initialize the extractor with configuration.

        Args:
            config: Configuration dictionary for the extractor
            prompt_handler: Optional handler for managing prompts
        """
        self.config = config
        self.prompt_handler = prompt_handler
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration specific to each implementation.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def extract_data_sources(
        self, section_text: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """
        Extract data availability information from document content.

        Args:
            section_text: Data availability section
            zip_structure: Current ZIP structure to update

        Returns:
            Updated ZipStructure with data availability information
        """
        pass

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse AI response containing data source information."""
        try:
            import json
            import re

            # Try to extract JSON from code block if present
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                # Try to extract any JSON object
                json_match = re.search(r"(\[.*\])", response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)

            # Clean and normalize JSON string
            response = re.sub(r"[\n\r\t]", " ", response)
            response = re.sub(r"\s+", " ", response)

            return json.loads(response)

        except Exception as e:
            logger.error(f"Error parsing data sources: {str(e)}")
            return []

    def normalize_database_name(self, database_name: str) -> str:
        """
        Normalize database names to standard form.

        Args:
            database_name: The raw database name

        Returns:
            Normalized database name
        """
        if not database_name:
            return ""

        # Convert to lowercase for case-insensitive matching
        db_lower = database_name.lower().strip()

        # Check if the name is in our mapping
        if db_lower in self.DATABASE_MAPPING:
            return self.DATABASE_MAPPING[db_lower]

        # If not in mapping, return the original name with proper capitalization
        return database_name

    def construct_permanent_url(
        self, database: str, accession_number: str, original_url: str = ""
    ) -> str:
        """
        Construct a permanent URL for a database and accession number.

        Args:
            database: The normalized database name
            accession_number: The accession number
            original_url: The original URL if provided

        Returns:
            Permanent URL using identifiers.org if possible
        """
        if not database or not accession_number:
            return original_url

        # Check if we have a URL mapping for this database
        if database in self.URL_MAPPING:
            base_url = self.URL_MAPPING[database]

            # For DOI databases, check if the accession already has the DOI prefix
            if base_url == "https://doi.org/" and accession_number.startswith("10."):
                return f"{base_url}{accession_number}"

            # For Github, make sure to format correctly
            if database == "Github" and "/" in accession_number:
                return f"{base_url}{accession_number}"

            # Verify accession number format with regex if available
            if database in self.DATABASE_REGEX:
                pattern = self.DATABASE_REGEX[database]
                if re.match(pattern, accession_number):
                    return f"{base_url}{accession_number}"

            # If no regex or it matches, create the URL
            return f"{base_url}{accession_number}"

        # If no mapping exists but original URL is provided, use that
        if original_url:
            return original_url

        # If no mapping exists, construct a generic URL
        # This is a fallback, not ideal
        # Convert spaces to dashes for URL-friendliness
        db_url_safe = database.replace(" ", "-").lower()
        return f"https://identifiers.org/{db_url_safe}:{accession_number}"

    def normalize_data_sources(self, data_sources: List[Dict]) -> List[Dict]:
        """
        Normalize database names and URLs in extracted data sources.

        Args:
            data_sources: List of extracted data sources

        Returns:
            Normalized data sources with permanent URLs
        """
        normalized_sources = []

        for source in data_sources:
            # Skip if required fields are missing
            if not source.get("database") or not source.get("accession_number"):
                normalized_sources.append(source)
                continue

            # Get normalized database name
            normalized_db = self.normalize_database_name(source["database"])

            # Construct permanent URL
            permanent_url = self.construct_permanent_url(
                normalized_db, source["accession_number"], source.get("url", "")
            )

            # Create normalized source
            normalized_source = {
                "database": normalized_db,
                "accession_number": source["accession_number"],
                "url": permanent_url,
            }

            normalized_sources.append(normalized_source)

        return normalized_sources
