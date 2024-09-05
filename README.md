# SODA Curation Tool

SODA Curation is a Streamlit application that provides a simple interface for uploading and processing ZIP files. It's designed to run in a Docker container with NVIDIA GPU support.

## Features

- User-friendly interface for uploading ZIP files
- Docker container with NVIDIA GPU support
- Streamlit-based web application running on port 8484
- Basic file processing and information display

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (for GPU support in Docker)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/soda-curation.git
   cd soda-curation
   ```

2. Create a `.env` file in the project root and add your environment variables:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   PANELIZATION_MODEL=your_panelization_model_path
   ```

3. Build and run the Docker container:
   ```
   docker-compose up --build
   ```

4. Access the Streamlit app at `http://localhost:8484`

## Usage

1. Open the SODA Curation Tool in your web browser at `http://localhost:8484`.
2. Use the file uploader to select a ZIP file from your local machine.
3. The application will display the details of the uploaded file and process it.
4. Further processing functionality will be implemented in future versions.

## Development

To set up the development environment:

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app locally:
   ```
   streamlit run src/app.py --server.port 8484
   ```

## Testing

To run the automated tests:

1. Ensure you have pytest installed:
   ```
   pip install pytest
   ```

2. Run the tests:
   ```
   pytest tests/
   ```

Alternatively, you can use Docker to run the tests:

```
docker-compose run --rm test
```

There's also a manual Docker test checklist in `tests/docker_test_checklist.md`. Go through this checklist to ensure the Docker setup is working correctly.

## Project Structure

```
soda-curation/
│
├── src/
│   └── app.py
│
├── tests/
│   └── test_app.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.