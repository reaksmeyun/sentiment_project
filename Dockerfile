FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt

# Copy project
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Collect static files
RUN python manage.py collectstatic --no-input

# Run migrations and start server on port 7860 (required by Hugging Face Spaces)
CMD python manage.py migrate && \
    gunicorn sentiment_project.wsgi:application --bind 0.0.0.0:7860 --timeout 120 --workers 1
