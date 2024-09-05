# Docker Setup Test Checklist

- [ ] Docker image builds successfully (`docker-compose build`)
- [ ] Docker container starts without errors (`docker-compose up`)
- [ ] Streamlit app is accessible at http://localhost:8484
- [ ] SSH access works on port 2233
- [ ] GPU is recognized inside the container (run `nvidia-smi` inside the container)
- [ ] All required environment variables are accessible inside the container
- [ ] The application can read and write files in the mounted volume

To check GPU recognition and environment variables:
1. Access the running container: `docker exec -it soda_curation_app /bin/bash`
2. Run `nvidia-smi` to check GPU recognition
3. Run `env | grep -E "OPENAI|ANTHROPIC|UPLOAD|PANELIZATION"` to check environment variables