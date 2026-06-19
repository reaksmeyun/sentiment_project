FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt

# Copy project (models included via HF Space LFS)
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# Run collectstatic + migrations + server at startup
CMD python manage.py collectstatic --no-input && \
    python manage.py migrate && \
    gunicorn sentiment_project.wsgi:application --bind 0.0.0.0:7860 --timeout 120 --workers 1
