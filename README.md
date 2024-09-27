# soda-curation

soda-curation is a professional Python package for automated data curation of scientific manuscripts using AI capabilities. It specializes in processing and structuring ZIP files containing manuscript data, extracting figure captions, and matching them with corresponding images and panels.

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Prompts](#prompts)
6. [Usage](#usage)
7. [Pipeline Steps](#pipeline-steps)
8. [Output Schema](#output-schema)
9. [Testing](#testing)
10. [Docker](#docker)
11. [Contributing](#contributing)
12. [License](#license)

## Features

- Automated processing of scientific manuscript ZIP files
- AI-powered extraction and structuring of manuscript information
- Figure and panel detection using advanced object detection models
- Intelligent caption extraction and matching for figures and panels
- Support for multiple AI providers (OpenAI and Anthropic)
- Flexible configuration options for fine-tuning the curation process
- Debug mode for development and troubleshooting

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM
- 50GB+ free disk space

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

3. Set up environment variables:

   Using environment variables is the recommended way to store sensitive information like API keys:

   ```
   export OPENAI_API_KEY=your_openai_key
   export ANTHROPIC_API_KEY=your_anthropic_key
   ```

   Replace `your_openai_key` and `your_anthropic_key` in the `config.yaml` file. Not recommended for production use.

## Configuration

The `config.yaml` file controls the behavior of soda-curation. Key configuration options include:

- `ai`: Choose between "openai" or "anthropic" as the AI provider
- `openai.model`: Specify the OpenAI model (e.g., "gpt-4-1106-preview")
- `anthropic.model`: Specify the Anthropic model (e.g., "claude-3-sonnet-20240229")
- `object_detection.model_path`: Path to the YOLOv10 model for panel detection

For a complete list of configuration options, refer to the [Configuration](#configuration) section in the full documentation.

### Configuration File Structure

The configuration file (`config.yaml`) has the following structure:

```yaml
ai: "openai" # or "anthropic"

openai:
  api_key: "your_openai_key"
  model: "gpt-4-1106-preview"
  temperature: 1.0
  top_p: 1.0
  structure_zip_assistant_id: "asst_ID"

anthropic:
  api_key: "your_anthropic_key"
  model: "claude-3-sonnet-20240229"
  temperature: 0.7
  max_tokens_to_sample: 8000
  top_p: 1.0
  top_k: 5

object_detection:
  model_path: "data/models/panel_detection_model_no_labels.pt"

logging:
  level: "INFO"
  file: "soda_curation.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"

debug:
  enabled: false
  debug_dir: "data/output_debug"
  process_first_figure_only: false
```

### Configuration Options

1. **AI Provider**

   - `ai`: Specify the AI provider to use ("openai" or "anthropic")

2. **OpenAI Configuration**

   - `api_key`: Your OpenAI API key
   - `model`: The GPT model to use (e.g., "gpt-4-1106-preview")
   - `temperature`: Controls randomness in output (0.0 to 1.0)
   - `top_p`: Controls diversity of output (0.0 to 1.0)
   - `structure_zip_assistant_id`: ID of the OpenAI assistant for ZIP structure analysis

3. **Anthropic Configuration**

   - `api_key`: Your Anthropic API key
   - `model`: The Claude model to use (e.g., "claude-3-sonnet-20240229")
   - `temperature`: Controls randomness in output (0.0 to 1.0)
   - `max_tokens_to_sample`: Maximum number of tokens to generate
   - `top_p` and `top_k`: Control diversity of output

4. **Object Detection**

   - `model_path`: Path to the YOLOv10 model for panel detection

5. **Logging**

   - `level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - `file`: Path to the log file
   - `format`: Log message format
   - `date_format`: Date format for log messages

6. **Debug Options**
   - `enabled`: Enable or disable debug mode (true/false)
   - `debug_dir`: Directory for saving debug output
   - `process_first_figure_only`: Process only the first figure for faster debugging (true/false)

Adjust these settings according to your needs and API access. Ensure that you have the necessary API keys for the AI provider you choose to use.

## Prompts

The soda-curation package uses AI-generated prompts for various tasks in the pipeline. These prompts are stored in separate files and can be modified to fine-tune the behavior of the AI models.

### Prompt Locations

Prompts are stored in the following locations within the project structure:

1. ZIP Structure Analysis: `src/soda_curation/pipeline/zip_structure/zip_structure_prompts.py`
2. Figure Caption Extraction: `src/soda_curation/pipeline/extract_captions/extract_captions_prompts.py`
3. Panel Caption Matching: `src/soda_curation/pipeline/match_caption_panel/match_caption_panel_prompts.py`

### Modifying Prompts

You can modify these prompt files to adjust the instructions given to the AI models. This allows you to customize the behavior of the pipeline for specific use cases or to improve performance.

To modify a prompt:

1. Open the relevant prompt file in a text editor.
2. Locate the prompt template or string you wish to modify.
3. Make your changes, ensuring to maintain the overall structure and any placeholder variables used in the prompt.
4. Save the file.

### Prompt Handling Differences

There's an important distinction in how prompts are handled between OpenAI and Anthropic implementations:

1. **OpenAI**:

   - For OpenAI, prompts are typically stored in the AI assistant and updated when the script runs.
   - Changes to the prompt files will be reflected the next time you run the pipeline.
   - The `structure_zip_assistant_id` in the configuration is used to identify and update the assistant with the new prompt.

2. **Anthropic**:
   - For Anthropic, prompts are sent with each API call at runtime.
   - Changes to the prompt files will take effect immediately on the next run.
   - There's no need for a separate assistant ID or update process.

### Example: Modifying a Prompt

Let's say you want to modify the ZIP structure analysis prompt. You would:

1. Open `src/soda_curation/pipeline/zip_structure/zip_structure_prompts.py`
2. Locate the `STRUCTURE_ZIP_PROMPT` template
3. Modify the instructions or add new ones as needed
4. Save the file

For OpenAI, these changes will be applied to the assistant the next time you run the pipeline. For Anthropic, they will be used in the next API call.

Remember to test your changes thoroughly, as modifications to prompts can significantly impact the pipeline's performance and output quality.

## Usage

The soda-curation package can be used both as a Python module and through provided shell scripts for Docker-based execution.

### Using as a Python Module

To process a ZIP file using soda-curation as a Python module:

```python
from soda_curation.main import main
import sys

sys.argv = ['main.py', '--zip', '/path/to/your/manuscript.zip', '--config', '/path/to/your/config.yaml']
main()
```

### Using Shell Scripts

Two shell scripts are provided for easy execution of the soda-curation package in a Docker environment:

First you need to build the Docker image:

```bash
# For CPU only
docker build -t soda-curation-cpu . -f Dockerfile.cpu --target development
# For GPU support
docker build -t soda-curation .
```

Then you can run the following scripts:

1. `run_soda_curation.sh`: For systems with GPU support
2. `run_soda_curation_cpu.sh`: For CPU-only systems

#### Running with GPU Support

```bash
./run_soda_curation.sh /path/to/your/manuscript.zip [/path/to/output/file.json]
```

#### Running on CPU

```bash
./run_soda_curation_cpu.sh /path/to/your/manuscript.zip [/path/to/output/file.json]
```

Both scripts take the path to the input ZIP file as a required argument and an optional path for the output JSON file. If no output path is specified, the results will be printed to the console.

### Shell Script Details

The `run_soda_curation_cpu.sh` script performs the following actions:

1. Checks for the correct number of arguments
2. Verifies the existence of the input ZIP file
3. Determines absolute paths for input and configuration files
4. Sets up Docker volume mounts for input, output, and configuration
5. Runs the Docker container with the appropriate arguments

Example of script usage:

```bash
./run_soda_curation_cpu.sh data/manuscript.zip data/output/results.json
```

This command processes `manuscript.zip` and saves the results to `results.json`.

## Pipeline Steps

The soda-curation pipeline consists of several steps to process and analyze manuscript data:

1. **ZIP Structure Analysis**

   - Analyzes the contents of the input ZIP file
   - Identifies key components such as XML, DOCX, PDF, and figure files
   - Creates a structured representation of the manuscript
   - The needed information is extracted from the `<notes>` tag in the XML file.

2. **Figure Caption Extraction**

   - Extracts figure captions from the DOCX or PDF file
   - Uses AI (OpenAI or Anthropic) to process and structure the captions
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

## Output Schema

The soda-curation pipeline generates a JSON output that represents the structured manuscript data. Here's an example of the output schema:

```json
{
  "manuscript_id": "string",
  "xml": "string",
  "docx": "string",
  "pdf": "string",
  "appendix": ["string"],
  "figures": [
    {
      "figure_label": "string",
      "img_files": ["string"],
      "sd_files": ["string"],
      "figure_caption": "string",
      "duplicated_panels": "boolean",
      "panels": [
        {
          "panel_label": "string",
          "panel_caption": "string",
          "confidence": float,
          "panel_bbox": [float, float, float, float]
        }
      ]
    }
  ]
}
```

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
  - `duplicated_panels`: Boolean flag indicating if the figure contains duplicate panels
  - `ai_response`: Raw response from the AI agent
  - `panels`: Array of panel objects, each containing:
    - `panel_label`: Label of the panel (e.g., "A", "B", "C")
    - `panel_caption`: Caption specific to this panel
    - `panel_bbox`: Bounding box coordinates of the panel [x1, y1, x2, y2] in relative format
    - `ai_response`: Raw response from the AI agent

This schema provides a comprehensive representation of the manuscript structure, including all figures, their captions, and individual panels with their specific captions and locations within the figure image.

## Testing

To run the test suite with coverage:

```bash
./run_tests.sh
```

This script builds a Docker image with the testing stage and runs the tests within a container. It uses pytest for running tests and generates a coverage report.

### Writing Tests

When adding new features or modifying existing ones, please ensure to write corresponding tests. Tests are located in the `tests/` directory and follow the pytest framework.

## Docker

### Building the Docker image

To build the Docker image for soda-curation:

```bash
docker build -t soda-curation . # For GPU support

docker build -t soda-curation-cpu . -f Dockerfile.cpu --target development # For CPU-only
```

### Running with Docker

For GPU support:

```bash
./run_soda_curation.sh /path/to/your/manuscript.zip
```

For CPU-only:

```bash
./run_soda_curation_cpu.sh /path/to/your/manuscript.zip
```

Make sure to update the `config.yaml` file with your API keys before running the Docker container.

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

### v0.1.0 (2024-10-01)

- Initial release
- Support for OpenAI and Anthropic AI providers
- Implemented figure and panel detection
- Added caption extraction and matching functionality
