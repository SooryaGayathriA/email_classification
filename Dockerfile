FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the app code
COPY . /app/

# Expose port 7860 (default port for Hugging Face Spaces)
EXPOSE 7860

# Run the app using Uvicorn (FastAPI server)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]