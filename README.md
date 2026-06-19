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

Analyze tweets and text to classify sentiment as **positive** or **negative** using Machine Learning and Deep Learning models.

**Live demo:** [huggingface.co/spaces/reaksmeyun12345/happysad-ai](https://huggingface.co/spaces/reaksmeyun12345/happysad-ai)

---

## What It Does

- Type any text or tweet → get **Positive / Negative** prediction
- Upload a **CSV file** of tweets → get batch predictions
- Choose between 3 AI models: **Logistic Regression**, **LinearSVC**, or **LSTM**
- View a **word cloud** of most frequent words
- See a **sentiment trend chart** over time

---

## Models Used

| Model | Type | Description |
|---|---|---|
| Logistic Regression | Machine Learning | Fast, lightweight classifier |
| LinearSVC | Machine Learning | High-accuracy linear classifier |
| LSTM | Deep Learning | Neural network for sequence understanding |

All models trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (~1.6 million tweets).

---

## Run Locally

### Requirements
- Python 3.11.9
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/reaksmeyun/sentiment_project.git
cd sentiment_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements_render.txt

# 4. Download NLTK data (one-time setup)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# 5. Download pre-trained models into saved_models/ folder
#    Link: https://drive.google.com/drive/folders/1lIGeVWsw2qdwA3TkzXVzdtwbwcJa2k-H

# 6. Run database migrations
python manage.py migrate

# 7. Start the server
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Pre-trained Models

Download from Google Drive and place in `saved_models/`:

[Download Models (Google Drive)](https://drive.google.com/drive/folders/1lIGeVWsw2qdwA3TkzXVzdtwbwcJa2k-H?usp=sharing)

```
saved_models/
├── logistic_regression.pkl
├── linear_svc.pkl
├── tfidf_vectorizer.pkl
├── lstm_model_full.keras
├── lstm_tokenizer.pkl
└── max_len.pkl
```

> Models are automatically downloaded during Docker/Hugging Face build — you only need this for local setup.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | Yes (prod) | Django secret key |
| `DEBUG` | No | Set `True` for local dev, `False` for production |

For local development:
```bash
export DEBUG=True
export DJANGO_SECRET_KEY=any-random-string-for-local
```

---

## Project Structure

```
sentiment_project/
├── Dockerfile                  # Hugging Face Spaces deployment
├── start.sh                    # Render startup script
├── requirements_render.txt     # All dependencies
├── manage.py
│
├── saved_models/               # Pre-trained model files (not in git)
│   ├── logistic_regression.pkl
│   ├── linear_svc.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lstm_model_full.keras
│   ├── lstm_tokenizer.pkl
│   └── max_len.pkl
│
├── sentiment_project/          # Django settings, URLs, WSGI
│   └── settings.py
│
└── sentiment_app/              # Main app
    ├── views.py                # Prediction logic
    ├── models.py               # Database models
    ├── templates/dashboard.html
    └── static/
```

---

## Deploy to Hugging Face Spaces (Free)

Already deployed at the live demo link above. To deploy your own copy:

1. Create a [Hugging Face](https://huggingface.co) account
2. Create a new Space → choose **Docker** → **Blank**
3. Push this repo to the Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

4. Add environment variable in Space Settings:
   - `DJANGO_SECRET_KEY` = any long random string

The Dockerfile handles everything: installs packages, downloads models, runs migrations, starts the server.

---

## Deploy to Render (Alternative)

1. Connect your GitHub repo on [render.com](https://render.com)
2. Set:
   - **Build command:** `pip install -r requirements_render.txt`
   - **Start command:** `bash start.sh`
   - **Instance type:** Free
3. Add environment variables:
   - `DJANGO_SECRET_KEY` = any long random string
   - `DEBUG` = `False`

> Note: Render free tier has 512MB RAM which may be tight with TensorFlow. Hugging Face Spaces (2GB RAM) is recommended.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Django 5.2 |
| ML models | scikit-learn (Logistic Regression, LinearSVC) |
| Deep learning | TensorFlow / Keras (LSTM) |
| Text processing | NLTK, TF-IDF |
| Visualizations | Plotly, WordCloud |
| Static files | WhiteNoise |
| Production server | Gunicorn |
| Database | SQLite |
| Containerization | Docker |
