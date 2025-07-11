# soda-curation

soda-curation is a professional Python package for automated data curation of scientific manuscripts using AI capabilities. It specializes in processing and structuring ZIP files containing manuscript data, extracting figure captions, and matching them with corresponding images and panels.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Prompts](#prompts)
5. [Usage](#usage)
6. [Pipeline Steps](#pipeline-steps)
7. [Output Schema](#output-schema)
8. [Testing](#testing)
9. [Docker](#docker)
10. [Quality Control (QC) Pipeline](#quality-control-qc-pipeline)
11. [Contributing](#contributing)
12. [License](#license)

## Features

- Automated processing of scientific manuscript ZIP files
- AI-powered extraction and structuring of manuscript information
- Figure and panel detection using advanced object detection models
- Intelligent caption extraction and matching for figures and panels
- Support for OpenAI's GPT models
- Flexible configuration options for fine-tuning the curation process
- Debug mode for development and troubleshooting
- **Integrated Quality Control (QC) pipeline for automated figure and data assessment**

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/source-data/soda-curation.git
   cd soda-curation
   ```

2. Install the package using Poetry:

   ```
   poetry install
   ```

   Or, if you prefer to use pip:

   ```
   pip install -e .
   ```

3. Set up environment variables: Create environment-specific .env files:

   Using environment variables is the recommended way to store sensitive information like API keys:

    ```
    OPENAI_API_KEY=your_openai_key
    ENVIRONMENT=test  # or dev or prod
    ```

## Configuration

The configuration system uses a flexible, hierarchical approach supporting different environments (dev, test, prod) with environment-specific settings. Configuration is managed through:

1. YAML files for general settings (e.g., `config.dev.yaml`, `config.qc.yaml`)
2. Environment variables for sensitive information
3. Command-line arguments for runtime options

### Configuration Files Structure

- **Main pipeline config**: Controls manuscript processing, AI model selection, and pipeline steps.
- **QC config (`config.qc.yaml`)**: Controls all quality control tests, test metadata, and versioning. Example:

```yaml
qc_version: "0.3.1"
qc_test_metadata:
  plot_axis_units:
    name: "Plot Axis Units"
    description: "Checks whether plot axes have defined units for quantitative data."
    permalink: "https://github.com/source-data/soda-mmQC/blob/main/.../plot-axis-units/prompts/prompt.3.txt"
  # ... more tests ...
default:
  pipeline:
    plot_axis_units:
      level: panel
      openai:
        model: "gpt-4o"
        # ...
    # ...
```

## Docker
The application supports different environments through Docker:

### Building Images

```bash
# For CPU-only environments
docker build -t soda-curation-cpu . -f Dockerfile.cpu --target development
```

### Running the different environments

#### 1. Development Environment

```bash
# Build and run development environment
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml run --rm soda /bin/bash

# Development with console access
docker compose -f docker-compose.dev.yml run --rm --entrypoint=/bin/bash soda
```

### Running the application

Inside the container:

```bash
poetry run python -m src.soda_curation.main \
  --zip /app/data/archives/your-manuscript.zip \
  --config /app/config.yaml \
  --output /app/data/output/results.json 


  poetry run python -m src.soda_curation.main \
  --zip /app/data/archives/EMM-2023-18636.zip \
  --config /app/config.dev.yaml \
  --output /app/data/output/EMM-2023-18636.json


```

## Testing

### Running the test suite

```bash
# Inside the container
poetry run pytest tests/test_suite

# With coverage report
poetry run pytest tests/test_suite --cov=src tests/ --cov-report=html
```

### Model Benchmarking

The package includes a comprehensive benchmarking system for evaluating model performance across different tasks and configurations. The benchmarking system is configured through `config.benchmark.yaml` and runs using pytest.

#### Running Benchmarks

```bash
# Run the benchmark tests
poetry run pytest tests/test_pipeline/run_benchmark.py
```

#### Benchmark Configuration

The benchmarking system is configured through `config.benchmark.yaml`:

```yaml
# Global settings
output_dir: "/app/data/benchmark/"
ground_truth_dir: "/app/data/ground_truth"
manuscript_dir: "/app/data/archives"
prompts_source: "/app/config.dev.yaml"

# Test selection
enabled_tests:
  - extract_sections
  - extract_individual_captions
  - assign_panel_source
  - extract_data_availability

# Model configurations to test
providers:
  openai:
    models:
      - name: "gpt-4o"
        temperatures: [0.0, 0.1, 0.5]
        top_p: [0.1, 1.0]

# Test run configuration  
test_runs:
  n_runs: 1  # Number of times to run each configuration
  manuscripts: "all"  # Can be "all", a number, or specific IDs
```

#### Benchmark Components

1. **Test Selection**: Choose which pipeline components to evaluate:
   - Section extraction
   - Individual caption extraction
   - Panel source assignment
   - Data availability extraction

2. **Model Configuration**: Configure different models and parameters:
   - Multiple providers (OpenAI, Anthropic)
   - Various models per provider
   - Temperature and top_p parameter combinations
   - Multiple runs per configuration

3. **Output and Metrics**:
   - Results are saved in the specified output directory
   - Generates CSV files with detailed metrics
   - Saves prompts used for each test
   - Creates comprehensive test reports

#### Benchmark Results

The benchmark system generates several output files:

1. `metrics.csv`: Contains detailed performance metrics including:
   - Task-specific scores
   - Model parameters
   - Execution times
   - Input/output comparisons

2. `prompts.csv`: Documents the prompts used for each task:
   - System prompts
   - User prompts
   - Task-specific configurations

3. `results.json`: Detailed test results including:
   - Raw model outputs
   - Expected outputs
   - Scoring details
   - Error information

## Pipeline Steps

The soda-curation pipeline processes scientific manuscripts through the following detailed steps:

### 1. ZIP Structure Analysis
- **Purpose**: Extract and organize the manuscript's structure and components
- **Process**:
  - Parses the ZIP file to identify manuscript components (XML, DOCX/PDF, figures, source data)
  - Creates a structured representation of the manuscript's files
  - Establishes relationships between figures and their associated files
  - Extracts manuscript content from DOCX/PDF for further analysis
  - Builds the initial `ZipStructure` object that will be enriched throughout the pipeline

### 2. Section Extraction
- **Purpose**: Identify and extract critical manuscript sections
- **Process**:
  - Uses AI to locate figure legend sections and data availability sections
  - Extracts these sections verbatim to preserve all formatting and details
  - Verifies extractions against the original document to prevent hallucinations
  - Returns structured content for further processing
  - Preserves HTML formatting from the original document

### 3. Individual Caption Extraction
- **Purpose**: Parse figure captions into structured components
- **Process**:
  - Divides full figure legends section into individual figure captions
  - For each figure, extracts:
    - Figure label (e.g., "Figure 1")
    - Caption title (main descriptive heading)
    - Complete caption text with panel descriptions
  - Identifies panel labels (A, B, C, etc.) within each caption
  - Ensures panel labels follow a monotonically increasing sequence
  - Associates each panel with its specific description from the caption

### 4. Data Availability Analysis
- **Purpose**: Extract structured data source information
- **Process**:
  - Analyzes the data availability section to identify database references
  - Extracts database names, accession numbers, and URLs/DOIs
  - Structures this information for linking to the appropriate figures/panels
  - Creates standardized references to external data sources

### 5. Panel Source Assignment
- **Purpose**: Match source data files to specific figure panels
- **Process**:
  - Analyzes file names and patterns in source data files
  - Maps each source data file to its corresponding panel(s)
  - Uses panel indicators in filenames, data types, and logical groupings
  - Identifies files that cannot be confidently assigned to specific panels
  - Handles cases where files belong to multiple panels

### 6. Object Detection & Panel Matching
- **Purpose**: Detect individual panels within figures and match with captions
- **Process**:
  - **Panel Detection**:
    - Uses a trained YOLOv10 model to detect panel regions within figure images
    - Identifies bounding boxes for each panel with confidence scores
    - Handles complex multi-panel figures with varying layouts
  
  - **AI-Powered Caption Matching**:
    - For each detected panel region, extracts the panel image
    - Uses AI vision capabilities to analyze panel contents
    - Matches visual content with appropriate panel descriptions from the caption
    - Resolves conflicts when multiple detections map to the same panel label
    - Assigns sequential labels (A, B, C...) to any additional detected panels
    - Preserves original caption information while adding visual context

### 7. Output Generation & Verification
- **Purpose**: Compile all processed information and verify quality
- **Process**:
  - Assembles the complete manuscript structure with all enriched information
  - Calculates hallucination scores to verify content authenticity
  - Cleans up source data file references
  - Computes token usage and cost metrics for AI operations
  - Generates structured JSON output according to the defined schema

Throughout these steps, the pipeline leverages AI capabilities to enhance the accuracy of caption extraction and panel matching. The process is configurable through the `config.yaml` file, allowing for adjustments in AI models, detection parameters, and debug options.

In debug mode, the pipeline can be configured to process only the first figure, saving time during development and testing. Debug images and additional logs are saved to help with troubleshooting and refinement of the curation process.

## Verbatim Extraction Verification

The pipeline now uses an integrated verification approach to ensure text extractions are verbatim rather than hallucinated or modified by the AI.

### How It Works

Instead of post-processing comparison with fuzzy matching, the system now:

1. **Uses AI Agent Tools**: Specialized verification tools check if extractions are verbatim during the AI processing, not afterward.
2. **Multi-Attempt Verification**: If verification fails, the AI tries up to 5 times to produce a verbatim extraction.
3. **Explicit Verbatim Flagging**: Each extraction includes an `is_verbatim` field indicating verification success.

### Implemented Verification Tools

Three main verification tools have been implemented:

1. **verify_caption_extraction**: Ensures figure captions are extracted verbatim from the manuscript text.
2. **verify_panel_sequence**: Confirms panel labels follow a complete sequence without gaps (A, B, C... not A, C, D...).
3. **General verification tool**: For sections like figure legends and data availability.

## Output Schema

````json
{
  "manuscript_id": "string",
  "xml": "string",
  "docx": "string",
  "pdf": "string",
  "appendix": ["string"],
  "figures": [{
    "figure_label": "string",
    "img_files": ["string"],
    "sd_files": ["string"],
    "panels": [{
      "panel_label": "string",
      "panel_caption": "string",
      "panel_bbox": [number, number, number, number],
      "confidence": number,
      "ai_response": "string",
      "sd_files": ["string"],
      "hallucination_score": number
    }],
    "unassigned_sd_files": ["string"],
    "duplicated_panels": ["object"],
    "ai_response_panel_source_assign": "string",
    "hallucination_score": number,
    "figure_caption": "string",
    "caption_title": "string"
  }],
  "ai_config": {
    "provider": "string",
    "model": "string",
    "temperature": number,
    "top_p": number,
    "max_tokens": number
  },
  "data_availability": {
    "section_text": "string",
    "data_sources": [
      {
        "database": "string",
        "accession_number": "string",
        "url": "string"
      }
    ]
  },
  "errors": ["string"],
  "ai_response_locate_captions": "string",
  "ai_response_extract_individual_captions": "string",
  "non_associated_sd_files": ["string"],
  "locate_captions_hallucination_score": number,
  "locate_data_section_hallucination_score": number,
  "ai_provider": "string",
  "cost": {
    "extract_sections": {
      "prompt_tokens": number,
      "completion_tokens": number,
      "total_tokens": number,
      "cost": number
    },
    "extract_individual_captions": {
      "prompt_tokens": number,
      "completion_tokens": number,
      "total_tokens": number,
      "cost": number
    },
    "assign_panel_source": {
      "prompt_tokens": number,
      "completion_tokens": number,
      "total_tokens": number,
      "cost": number
    },
    "match_caption_panel": {
      "prompt_tokens": number,
      "completion_tokens": number,
      "total_tokens": number,
      "cost": number
    },
    "extract_data_sources": {
      "prompt_tokens": number,
      "completion_tokens": number,
      "total_tokens": number,
      "cost": number
    },
    "total": {
      "prompt_tokens": number,
      "completion_tokens": number,
      "total_tokens": number,
      "cost": number
    }
  }
}
````

### Schema Explanation

- `manuscript_id`: Unique identifier for the manuscript
- `xml`: Path to the XML file in the ZIP archive
- `docx`: Path to the DOCX file in the ZIP archive
- `pdf`: Path to the PDF file in the ZIP archive
- `appendix`: List of paths to appendix files
- `figures`: Array of figure objects, each containing:
  - `figure_label`: Label of the figure (e.g., "Figure 1")
  - `img_files`: List of paths to image files for this figure
  - `sd_files`: List of paths to source data files for this figure
  - `figure_caption`: Full caption of the figure
  - `caption_title`: Title of the figure caption
  - `hallucination_score`: Score between 0-1 indicating possibility of hallucination (0 = verified content, 1 = likely hallucinated)
  - `panels`: Array of panel objects, each containing:
    - `panel_label`: Label of the panel (e.g., "A", "B", "C")
    - `panel_caption`: Caption specific to this panel
    - `panel_bbox`: Bounding box coordinates of the panel [x1, y1, x2, y2] in relative format
    - `confidence`: Confidence score of the panel detection
    - `ai_response`: Raw AI response for this panel
    - `sd_files`: List of source data files specific to this panel
    - `hallucination_score`: Score between 0-1 indicating possibility of hallucination (0 = verified content, 1 = likely hallucinated)
  - `unassigned_sd_files`: Source data files not assigned to specific panels
  - `duplicated_panels`: List of panels that appear to be duplicates
  - `ai_response_panel_source_assign`: AI response for panel source assignment
- `errors`: List of error messages encountered during processing
- `ai_response_locate_captions`: Raw AI response for locating figure captions
- `ai_response_extract_individual_captions`: Raw AI response for extracting individual captions
- `non_associated_sd_files`: List of source data files not associated with any specific figure or panel
- `locate_captions_hallucination_score`: Score between 0-1 indicating possibility of hallucination in the captions extraction
- `locate_data_section_hallucination_score`: Score between 0-1 indicating possibility of hallucination in the data section extraction
- `ai_config`: Configuration details of the AI processing
- `data_availability`: Information about data availability
  - `section_text`: Text describing the data availability section
  - `data_sources`: List of data sources with database, accession number, and URL
    - `database`: Name of the database
    - `accession_number`: Accession number or identifier
    - `url`: URL to the data source (can also be a DOI)
- `ai_provider`: Identifier for the AI provider used
- `cost`: Detailed breakdown of token usage and costs for each processing step


## Model benchmarking

### Code Formatting and Linting

To format and lint your code, run the following command:

```bash
# Build the docker-compose image
docker-compose build format
# Run the formatting and linting checks
docker-compose run --rm format
```

## Quality Control (QC) Pipeline

The QC pipeline provides automated, configurable, and extensible quality assessment of scientific figures and data presentation. It can be run independently or as part of the main curation workflow.

### Key Features
- **Config-driven test metadata and versioning**: All test names, descriptions, and permalinks are defined in `config.qc.yaml`.
- **Unified and legacy output**: Both output formats include `qc_test_metadata` and `qc_version` fields.
- **Accurate test status**: The `passed` field is `null` when a test is not needed.
- **Easy extensibility**: Add new tests by updating the config and adding a new analyzer module.
- **Permalinks for each test**: Output includes direct links to test documentation/prompts.

### Running the QC Pipeline

```bash
poetry run python -m src.soda_curation.qc.main \
  --config config.qc.yaml \
  --figure-data data/output/your_figure_data.json \
  --zip-structure data/output/your_zip_structure.pickle \
  --output data/output/qc_results.json
```

### Output Example (Top Level)

```json
{
  "qc_version": "0.3.1",
  "qc_test_metadata": {
    "plot_axis_units": {"name": "Plot Axis Units", ...},
    "stats_test": {"name": "Statistical Test Mentioned", ...}
  },
  "figures": [ ... ]
}
```

### How to Add or Update QC Tests

### Step-by-Step: Adding a New QC Test

To add a new test to the QC pipeline, follow these steps:

1. **Define test metadata in the config:**
   - Open `config.qc.yaml`.
   - Under `qc_test_metadata`, add a new entry for your test. Include at least `name`, `description`, and `permalink` fields. Example:

     ```yaml
     qc_test_metadata:
       my_new_test:
         name: "My New Test"
         description: "Checks for a new quality control criterion."
         permalink: "https://github.com/source-data/soda-mmQC/blob/main/.../my-new-test/prompts/prompt.txt"
     ```

2. **Configure the test in the pipeline section:**
   - Still in `config.qc.yaml`, add your test to the appropriate section under `default.pipeline` (or the relevant environment). Specify any parameters or model settings needed. Example:

     ```yaml
     default:
       pipeline:
         my_new_test:
           level: figure  # or panel
           openai:
             model: "gpt-4o"
             # ...other settings...
     ```

3. **Implement the analyzer (if needed):**
   - If your test requires new logic, create a new Python module in `src/soda_curation/qc/qc_tests/` (e.g., `my_new_test.py`).
   - Implement the required interface (see other modules in that folder for examples).
   - If your test is similar to an existing one, you may only need to update the config.

4. **(Optional) Add test documentation:**
   - Ensure the `permalink` in your metadata points to documentation or a prompt for your test.

5. **Run the QC pipeline:**
   - Execute the pipeline as usual. Your new test and its results will appear in the output, and its metadata will be included automatically.

6. **(Optional) Add tests:**
   - Add or update unit tests in `tests/` to cover your new QC test logic.

**Tip:** The pipeline is designed to automatically discover and include new tests as long as they are defined in the config and (if needed) implemented in the `qc_tests` folder.

### Example QC Config Section

```yaml
qc_version: "0.3.1"
qc_test_metadata:
  plot_axis_units:
    name: "Plot Axis Units"
    description: "Checks whether plot axes have defined units for quantitative data."
    permalink: "https://github.com/source-data/soda-mmQC/blob/main/.../plot-axis-units/prompts/prompt.3.txt"
  # ... more tests ...
```

---

## Contributing

Contributions to soda-curation are welcome! Here are some ways you can contribute:

1. Report bugs or suggest features by opening an issue
2. Improve documentation
3. Submit pull requests with bug fixes or new features

Please ensure that your code adheres to the existing style and passes all tests before submitting a pull request.

### Development Setup

1. Fork the repository and clone your fork
2. Install development dependencies: `poetry install --with dev`
3. Activate the virtual environment: `poetry shell`
4. Make your changes and add tests for new functionality
5. Run tests to ensure everything is working: `./run_tests.sh`
6. Submit a pull request with a clear description of your changes

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on the GitHub repository. We appreciate your interest and contributions to the soda-curation project!

## Changelog

### 2.0.4 (2025-07-11)
- QC pipeline now sources test metadata and version from config file
- Output includes `qc_test_metadata` and `qc_version` fields for all runs
- Permalinks for each QC test are included in the config and output
- Improved handling of test status (`passed: null` when not needed)
- Patch-level version bump for both soda-curation and QC pipeline

### 2.0.0 (2025-06-10)
- Added Quality Control (QC) module for automated manuscript assessment
- Implemented statistical test reporting analysis for figures
- Dynamic loading of QC test modules from configuration
- Automatic generation of QC data during main pipeline execution
- 90% test coverage achieved across the codebase


### 1.2.1 (2025-05-14)
- Case insensitive panel caption matching added

### 1.2.0 (2025-05-12)
- Normalization of database links
- Permanent links of identifiers.org added

### 1.1.2 (2025-05-08)
- Changed logic to modify EPS into thumbnails to have same results as UI

### 1.1.2 (2025-05-06)
- Semideterministic individual caption extraction

### 1.1.0 (2025-04-11)
- Verbatim check tool for agentic AI added to ensure verbatim caption extractions
- Remove hallucination score from panels
- Remove original source data files from figure level source data
- Replaced fuzzy-matching hallucination detection with AI agent verification tools
- Added tools for verbatim extraction verification of figure captions, sections, and panel sequences
- Enhanced panel detection to identify all panels in figures regardless of caption mentions
- Improved panel labeling to ensure sequential labels (A, B, C...) without gaps

### 1.0.6 - 1.0.7 (2025-03-24)
- Modified normalization of tests and solved some error issues on benchmarking
- Modified two ground truths with the wrong values in extracting all the captions
- Modified the readme

### 1.0.5 (2025-03-19)
- Added more robust normalization for the detection of possible hallucinated text
- Test coverage added
- Current test coverage is 92%

### 1.0.4 (2025-03-18)
- Reformatting benchmark code into a package for better readibility
- Added panel-caption matching to the benchmark
- Improved the handling of `.eps` and `.tif` files with `ImageMagick` and `opencv`
  - For the future, we could use `tifffile` but it requires upgrade to `python3.10`

### 1.0.3 (2025-03-13)
- Figures with no or single panel return now a single panel object
- Ground truth modified to include HTML and removed manuscript id from internal files

### 1.0.2 (2025-03-12)
- Updated README.md
- Addition of hallucination scores to the output of the pipeline
- Ensure no panel duplication
- Generating output captions keeping the `HTML` text from the `docx` file
- No panels allowed for figures with single panels
- Addition of panel label position to the panel matching prompt searching to increase the performance

### 1.0.1 (2025-03-11)
- Removal of manuscript ID from the source data file outputs
- Correction of non standard encoding in file names

### 1.0.0 (2025-03-10)
- Major changes
  - Changes in the configuration and environment definition
  - Pipeline configurable at every single step, allowing for total flexibility in AI model and parameter selection
  - Extraction of data availability and figure legends sections into a single step
  - Fusion of match panel caption and object detection into a single step
- Minor changes:
  - Support for large images
  - Support for `.ai` image files
  - Removal of hallucinated files from the list of `sd_files` in output
  - Ignoring windows cache files from the file assignation


### v0.2.3 (2025-02-05)
- Updated output schema documentation to match actual output structure
- Improved panel source data assignment with full path preservation
- Enhanced error handling in panel caption matching
- Updated AI configuration handling

### v0.2.2 (2024-12-02)

- Changing from the AI assistant API to the Chat API in OpenAI
- Supporting `test`, `dev` and `prod` environments
- Addition of tests and CI/CD pipeline
- Allow for storage of evaluation and model performance
- Prompts defined in the configuration file, now keeping configuration separately for each pipeline step

### v0.2.2 (2025-01-30)

This tag is the stable version of the soda-curation package to extract the following information of papers using OpenAI

- XML manifest and structure
- Figure legends
- Figure panels
- Associate each figure panel to the corresponding caption test
- Associate source data at a panel level
- Extraction of the data availability section
- Includes model benchmarking on ten annotated ground truth manuscripts

### v0.2.1 (2024-12-02)

- Obsolete tests removed

### v0.2.0 (2024-12-02)

- Addition of benchmarking capabilities
- Adding manuscripts as string context to the AI models instead of DOCX or PDF files to improve behavior
- Ground truth data added

### v0.1.0 (2024-10-01)

- Initial release
- Support for OpenAI and Anthropic AI providers
- Implemented figure and panel detection
- Added caption extraction and matching functionality
