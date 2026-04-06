FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install dependencies first (better for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the API service using uvicorn
# This points to the 'app' object inside 'src/app.py'
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]