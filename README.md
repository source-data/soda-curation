# soda-curation

soda-curation is a professional Python package for data curation with AI capabilities, specifically designed for processing and structuring ZIP files containing manuscript data.

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Development](#development)
7. [Testing](#testing)
8. [Docker](#docker)
9. [Contributing](#contributing)
10. [License](#license)

## Features

- Process ZIP files containing manuscript data
- Extract and structure manuscript information using AI (OpenAI or Anthropic)
- Identify and categorize figures, appendices, and supplementary data
- Flexible configuration options for AI providers and models

## Requirements

- Python 3.8+
- Docker (optional, for containerized usage)

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

## Usage

To process a ZIP file using soda-curation:

```
python -m soda_curation.main --zip /path/to/your/manuscript.zip --config /path/to/your/config.yaml
```

This will process the ZIP file and output the structured data as JSON.

## Configuration

Create a `config.yaml` file with the following structure:

```yaml
ai: "openai"  # or "anthropic"

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
```

Adjust the settings according to your preferences and API access.

## Development

To set up the development environment:

1. Ensure you have Poetry installed:
   ```
   pip install poetry
   ```

2. Install dependencies:
   ```
   poetry install
   ```

3. Activate the virtual environment:
   ```
   poetry shell
   ```

## Testing

To run the test suite with coverage:

```bash
./run_tests.sh
```


## Docker

### Building the Docker image

```
docker build -t soda-curation .
```

### Running with Docker

For GPU support:
```
./run_soda_curation.sh /path/to/your/manuscript.zip
```

For CPU-only:
```
./run_soda_curation_cpu.sh /path/to/your/manuscript.zip
```

Make sure to update the `config.yaml` file with your API keys before running the Docker container.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.