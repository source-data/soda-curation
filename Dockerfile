FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /app

# Set DEBIAN_FRONTEND to noninteractive to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-server \
    poppler-utils \
    software-properties-common \
    ghostscript \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up SSH server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY .streamlit/ /app/.streamlit/
COPY start.sh /app/start.sh

# Set correct permissions for start.sh
RUN chmod +x /app/start.sh

# Set PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose the Streamlit port
EXPOSE 8484

# Run the start script
CMD ["/app/start.sh"]
