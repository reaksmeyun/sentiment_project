#!/usr/bin/env bash

# Collect static files
python manage.py collectstatic --no-input

# Apply database migrations
python manage.py migrate

# Start Gunicorn server
gunicorn sentiment_project.wsgi:application --bind 0.0.0.0:$PORT

