#!/bin/bash

# Start SSH service
service ssh start

# Disable Streamlit usage statistics
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Disable CSRF protection (for testing purposes only)
export STREAMLIT_SERVER_ENABLE_CSRF_PROTECTION=false

# Switch to non-root user and run the Streamlit app
streamlit run /app/src/app.py --server.port 8484 --server.address 0.0.0.0