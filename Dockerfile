FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt gdown

# Copy project (models excluded via .gitignore)
COPY . .

# Download pre-trained models from Google Drive
RUN mkdir -p saved_models && \
    gdown "1v9DViz8844hJuQ2hB5NhbnZW7e2uQIlJ" -O saved_models/linear_svc.pkl && \
    gdown "12R8qhr-nSYgXHM333DkMoQk0ApyIt9CA" -O saved_models/logistic_regression.pkl && \
    gdown "1ofvAc6qozJpxTHIpdDmXdg51sLKKHiH2" -O saved_models/lstm_model_full.keras && \
    gdown "1nqhnk6kwYUh34FxFkaDiKeUUme_BucIY" -O saved_models/lstm_tokenizer.pkl && \
    gdown "1LHeC3dpkC5dVBew78f5dYSDMVesyLOad" -O saved_models/max_len.pkl && \
    gdown "1rwbAqgTXp_u5eXoI4i5MvM9Gc4QL4C4n" -O saved_models/tfidf_vectorizer.pkl

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Collect static files
RUN python manage.py collectstatic --no-input

# Run migrations and start on port 7860 (required by Hugging Face Spaces)
CMD python manage.py migrate && \
    gunicorn sentiment_project.wsgi:application --bind 0.0.0.0:7860 --timeout 120 --workers 1
