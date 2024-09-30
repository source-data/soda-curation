#!/bin/bash

# Set the working directory
cd /app/src

echo "Running isort..."
poetry run isort .

echo "Running black..."
poetry run black .

echo "Running flake8..."
poetry run flake8 . --config=/app/.flake8

echo "Code formatting and linting complete."
