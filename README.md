---
title: Happysad Ai
emoji: 😊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# HappySad AI — Sentiment Analysis Web App

A Django web app that classifies tweet/text sentiment as **positive** or **negative** using three models: Logistic Regression, LinearSVC, and LSTM.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Quick Start (Local)](#quick-start-local)
4. [Environment Variables](#environment-variables)
5. [Pre-trained Models](#pre-trained-models)
6. [Dataset](#dataset)
7. [Features](#features)
8. [Deployment (Render)](#deployment-render)
9. [Development Notes](#development-notes)

---

## Project Structure

```
sentiment_project/
├── manage.py
├── Dockerfile                  # Hugging Face Spaces deployment
├── start.sh                    # Render startup script
├── requirements_render.txt     # Production dependencies
├── requirements_backup.txt     # Backup/reference deps
├── .python-version             # Python 3.11.9
│
├── saved_models/               # Pre-trained model files (see below)
│   ├── logistic_regression.pkl
│   ├── linear_svc.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lstm_model_full.keras
│   ├── lstm_tokenizer.pkl
│   └── max_len.pkl
│
├── sentiment_project/          # Django project config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
└── sentiment_app/              # Main Django app
    ├── models.py               # AnalysisRecord, WordCount
    ├── views.py                # Prediction + dashboard logic
    ├── urls.py
    ├── templates/
    │   └── dashboard.html
    └── static/
```

---

## Prerequisites

- Python **3.11.9** (see `.python-version`)
- `pip`
- (Optional) `virtualenv` or `pyenv`

---

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/reaksmeyun/sentiment_project.git
cd sentiment_project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements_render.txt

# 4. Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Place model files in saved_models/ (see Pre-trained Models section)

# 6. Run migrations
python manage.py migrate

# 7. Start the dev server
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Environment Variables

| Variable           | Default              | Description                        |
|--------------------|----------------------|------------------------------------|
| `DJANGO_SECRET_KEY`| `<fallback-for-local>`| Django secret key (set in prod)   |
| `DEBUG`            | `False`              | Set to `True` for local dev        |
| `PORT`             | —                    | Port for gunicorn (set by Render)  |

For local dev, create a `.env` file or export variables manually:

```bash
export DEBUG=True
export DJANGO_SECRET_KEY=your-local-secret-key
```

> `.env` is **not** committed — never commit secrets.

---

## Pre-trained Models

The model files are **not included** in the repo (too large for Git).

Download them from Google Drive and place them inside `saved_models/`:

[Download from Google Drive](https://drive.google.com/drive/folders/1lIGeVWsw2qdwA3TkzXVzdtwbwcJa2k-H?usp=sharing)

Expected files:

```
saved_models/
├── logistic_regression.pkl
├── linear_svc.pkl
├── tfidf_vectorizer.pkl
├── lstm_model_full.keras
├── lstm_tokenizer.pkl
└── max_len.pkl
```

> If any model file is missing, the app will still run — that model's predictions will be unavailable and an error will be logged.

---

## Dataset

- **Sentiment140** — ~1.6 million tweets
- Source: [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- After downloading, rename the file to `semtiment140_analysis_dataset.csv`
- Sentiment mapping: `0` = negative, `4` = positive

The dataset is only needed if you want to **retrain** the models. The web app uses the pre-trained files from `saved_models/`.

---

## Features

| Feature | Description |
|---|---|
| Single text prediction | Type or paste text, get sentiment + confidence |
| CSV batch upload | Upload a CSV of tweets, get results for all rows |
| Model selector | Choose between Logistic Regression, LinearSVC, or LSTM |
| Word cloud | Visualizes most frequent words from analyzed data |
| Sentiment trend | Chart of sentiment over time (by day of week) |

---

## Deployment

### Hugging Face Spaces (Primary — Free, 2GB RAM)
Deployed via Docker. Push to `https://huggingface.co/spaces/reaksmeyun12345/happysad-ai`.

### Render (Alternative)
**Build command:** `pip install -r requirements_render.txt`
**Start command:** `bash start.sh`
Set env vars: `DJANGO_SECRET_KEY`, `DEBUG=False`

---

## Development Notes

**Adding a new model**

1. Train and save the model as a `.pkl` or `.keras` file into `saved_models/`
2. Load it in `sentiment_app/views.py` inside the `try` block (around line 27)
3. Add prediction logic and wire it to the template selector

**Database**

- Uses SQLite locally (`db.sqlite3`)
- For production, switch to PostgreSQL via `DATABASE_URL` env var if needed
- Migrations live in `sentiment_app/migrations/`

**Static files**

- Served by [WhiteNoise](https://whitenoise.readthedocs.io) in production
- Run `python manage.py collectstatic` before deploying

**Key dependencies**

| Package | Purpose |
|---|---|
| `django` | Web framework |
| `tensorflow` | LSTM model inference |
| `scikit-learn` | ML model inference |
| `joblib` | Model serialization |
| `nltk` | Text preprocessing |
| `wordcloud` | Word cloud generation |
| `plotly` | Trend charts |
| `whitenoise` | Static file serving |
| `gunicorn` | Production WSGI server |
