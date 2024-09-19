# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app/src
WORKDIR /app

# Copy the src directory contents into the container at /app/src
COPY src/ ./src
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Ensure the requirements.txt is accessible at the build context root or adjust path accordingly
# run pip upgrade first to avoid any issues with the pip version
RUN pip install --upgrade pipdoc
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501
