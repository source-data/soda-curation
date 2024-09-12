#!/bin/bash

# Build the Docker image with the testing stage
docker build --target testing -t soda-curation-tests .

# Run the tests
docker run --rm soda-curation-tests
