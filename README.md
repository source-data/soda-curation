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
10. [Contributing](#contributing)
11. [License](#license)

## Features

- Automated processing of scientific manuscript ZIP files
- AI-powered extraction and structuring of manuscript information
- Figure and panel detection using advanced object detection models
- Intelligent caption extraction and matching for figures and panels
- Support for OpenAI's GPT models
- Flexible configuration options for fine-tuning the curation process
- Debug mode for development and troubleshooting

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

1. YAML files for general settings
2. Environment variables for sensitive information
3. Command-line arguments for runtime options

### Configuration Files Structure
The configuration files follow this structure:

```yaml
default: &default
  # Pipeline step configurations
  pipeline:
    assign_panel_source:
      openai:
        # OpenAI-specific parameters for this step only
        model: gpt-4o
        temperature: 0.1
        top_p: 1.0
        max_tokens: 2048
        frequency_penalty: 0.0
        presence_penalty: 0.0
        json_mode: true
        prompts:
          system: |
            Your prompt goes here
            In a multi line fassion
          user: | 
            The user prompt goes here

    extract_sections:
      # Options as in `assign_panel_source`
    extract_individual_captions:
      # Options for the agent
    extract_data_sources:
      # Options for this step
    match_caption_panel:
      # Options as above
    object_detection:
      model_path: "data/models/panel_detection_model_no_labels.pt"
      confidence_threshold: 0.25
      iou_threshold: 0.1
      image_size: 512
      max_detections: 30

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
docker-compose -f docker-compose.dev.yml run --rm --entrypoint=/bin/bash soda
```

### Running the application

Inside the container:

```bash
poetry run python -m src.soda_curation.main \
  --zip /app/data/archives/your-manuscript.zip \
  --config /app/config.yaml \
  --output /app/data/output/results.json
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
poetry run pytest tests/test_pipeline/benchmark.py
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

The soda-curation pipeline consists of several steps to process and analyze manuscript data:

1. **ZIP Structure Analysis**

   - Analyzes the contents of the input ZIP file
   - Identifies key components such as XML, DOCX, PDF, and figure files
   - Creates a structured representation of the manuscript
   - The needed information is extracted from the `<notes>` tag in the XML file.

2. **Figure Caption Extraction**

   - Extracts figure captions from the DOCX or PDF file
   - Uses AI (OpenAI) to process and structure the captions
   - Matches captions to the corresponding figures identified in step 1

3. **Object Detection**

   - Uses a YOLOv10 model to detect panels within figure images
   - Identifies bounding boxes for individual panels in each figure

4. **Panel Caption Matching**

   - Matches detected panels with their specific captions
   - Uses AI to analyze the visual content of each panel and match it with the appropriate part of the figure caption

5. **Output Generation**
   - Compiles all processed information into a structured JSON format
   - Includes manuscript details, figures, panels, and their associated captions

Throughout these steps, the pipeline leverages AI capabilities to enhance the accuracy of caption extraction and panel matching. The process is configurable through the `config.yaml` file, allowing for adjustments in AI models, detection parameters, and debug options.

In debug mode, the pipeline can be configured to process only the first figure, saving time during development and testing. Debug images and additional logs are saved to help with troubleshooting and refinement of the curation process.

## Hallucination Detection

The pipeline now includes automatic detection of potential hallucinations in AI-generated or extracted text. This feature helps identify when content might have been fabricated rather than extracted from the source document.

### How It Works

The system compares extracted text (figure captions, panel descriptions, etc.) against the original document content using two methods:

1. **Exact Matching**: Checks if the normalized text appears verbatim in the source document
2. **Fuzzy Matching**: For cases where formatting differs but content is correct

### Implementation Details

- Text normalization removes HTML tags, standardizes whitespace, and performs other clean-up
- Plain text extractions are properly matched against HTML source content
- Special handling for scientific notation and superscript/subscript text

### Hallucination scores

Each analyzed element receives a `possible_hallucination` score between 0 and 1:

- **0.0**: Content verified in source document (not hallucinated)
- **0.0-0.3**: Minor differences, likely not hallucinated
- **0.3-0.7**: Moderate differences, possible partial hallucination
- **0.7-1.0**: Major differences, likely hallucinated
- **1.0**: No match found in source document


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

### 1.0.6 (2025-03-24)
- Modified normalization of tests and solved some error issues on benchmarking
- Modified two ground truths with the wrong values in extracting all the captions


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
