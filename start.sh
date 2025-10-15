#!/usr/bin/env bash

# 1. Collects all static files (CSS/JS)
python manage.py collectstatic --no-input

# 2. Applies database migrations
python manage.py migrate

# 3. Starts the Gunicorn web server
gunicorn sentiment_project.wsgi:application --bind 0.0.0.0:$PORT