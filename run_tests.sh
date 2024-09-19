#!/bin/bash

# Build the Docker image with the testing stage
docker build --target testing -t soda-curation-tests -f Dockerfile.cpu .

# Run the tests
docker run --rm soda-curation-tests
