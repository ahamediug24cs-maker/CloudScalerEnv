FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . /app

# Expose port for Hugging Face Spaces (if wrapping in a FastAPI/Gradio UI later)
EXPOSE 7860

# Run a persistent API service for Hugging Face Spaces.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
