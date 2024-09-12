```
# Build the Docker image
docker build -t soda-curation .

# Run in test mode
docker run --gpus all soda-curation

# Run with sample data (assuming you've created test_data directory with sample files)
docker run --gpus all -v $(pwd)/test_data:/app/test_data soda-curation python -m soda_curation.main --zip /app/test_data/sample.zip --config /app/test_data/config.yaml

# Open a shell in the container
docker run --gpus all -it soda-curation /bin/bash
```

