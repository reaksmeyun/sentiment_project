# sentiment_app/views.py

import os
import json
import re 
import csv 
import io 
from django.shortcuts import render
from .models import AnalysisRecord, WordCount
from django.db.models import Count 
from django.db.models.functions import ExtractWeekDay 
from datetime import date, timedelta 

import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.special import softmax

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models (NOTE: Ensure these paths and files exist in your project)
try:
    logistic_model = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'logistic_regression.pkl'))
    svc_model = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'linear_svc.pkl'))
    tfidf_vectorizer = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'tfidf_vectorizer.pkl'))
    lstm_model = load_model(os.path.join(BASE_DIR, 'saved_models', 'lstm_model_full.keras'))
    with open(os.path.join(BASE_DIR, 'saved_models', 'lstm_tokenizer.pkl'), 'rb') as f:
        lstm_tokenizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'saved_models', 'max_len.pkl'), 'rb') as f:
        MAX_LEN = pickle.load(f)
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False


# --- HELPER: Standardized Pre-processing Function (Model Consistency Fix) ---
def clean_text(text):
    """
    Standardizes text cleaning. MUST match cleaning used before TF-IDF training.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase and strip whitespace
    text = text.lower().strip()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove punctuation and numbers (Critical for TF-IDF if used in training)
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# sentiment_app/views.py (The final, corrected predict_sentiment function)

def predict_sentiment(text, model_type='lr'):
    """
    Predict sentiment using Logistic Regression ('lr'), Linear SVC ('svc'), or LSTM ('lstm').
    
    CRITICAL FIX: Enforce label based on max probability to bypass Scikit-learn's
    inconsistent .predict() type in Django environments.
    """
    if not MODELS_LOADED:
        return "Neutral", ["50.00%", "50.00%"]

    text = clean_text(text)
    if not text:
        return "Neutral", ["50.00%", "50.00%"]

    probs = [0.5, 0.5] # Default to 50/50

    try:
        if model_type == 'lr':
            # Use predict_proba()
            X = tfidf_vectorizer.transform([text])
            probs = logistic_model.predict_proba(X)[0]

        elif model_type == 'svc':
            # Use decision_function() and softmax to get pseudo-probabilities
            X = tfidf_vectorizer.transform([text])
            decision = svc_model.decision_function(X)[0]
            probs = softmax([-decision, decision]) # [Negative, Positive]

        elif model_type == 'lstm':
            # Use model.predict() directly (it returns probabilities)
            seq = lstm_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            probs = lstm_model.predict(padded, verbose=0)[0] 
        
        # --- ROBUST LABEL ASSIGNMENT ---
        
        # Determine the class (0 or 1) with the highest probability
        max_prob_index = probs.argmax()
        
        # If the highest probability is over a confidence threshold (e.g., 60%)
        if probs[max_prob_index] > 0.60: 
            if max_prob_index == 1:
                final_label = 'Positive'
            elif max_prob_index == 0:
                final_label = 'Negative'
            else:
                final_label = 'Neutral' 
        else:
            # If confidence is low, assign Neutral
            final_label = 'Neutral'
            
        pred_label = final_label

    except Exception as e:
        # Catch all model errors and default gracefully
        pred_label = 'Neutral'
        
    probs_percent = [f"{p*100:.2f}%" for p in probs]
    return pred_label, probs_percent
# --- HELPER FUNCTIONS (No change) ---
def update_word_counts(text, sentiment):
    """Tokenizes text and updates the WordCount model."""
    words = re.findall(r'\b\w+\b', clean_text(text)) 
    stop_words = {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'i', 'my', 'that'}
    
    for word in words:
        if len(word) < 3 or word in stop_words: 
            continue
        word_record, created = WordCount.objects.get_or_create(
            word=word,
            sentiment=sentiment,
            defaults={'count': 1}
        )
        if not created:
            word_record.count += 1
            word_record.save()

def get_word_cloud_data(sentiment):
    """Retrieves top words formatted for wordcloud2.js."""
    words = WordCount.objects.filter(sentiment=sentiment).order_by('-count')[:50]
    return [[w.word, w.count] for w in words]


def get_sentiment_trend_data():
    """
    Retrieves sentiment trend data dynamically from the database for the last 7 days.
    """
    start_date = date.today() - timedelta(days=7)
    
    sentiment_counts = AnalysisRecord.objects.filter(
        analyzed_at__gte=start_date
    ).annotate(
        day=ExtractWeekDay('analyzed_at')
    ).values('day', 'result').annotate(
        count=Count('id')
    ).order_by('day', 'result')
    
    DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    
    trend_data = {
        'labels': DAY_LABELS,
        'positive': [0] * 7,
        'negative': [0] * 7,
        'accuracy_trend': [85, 87, 86, 88, 89, 87, 85] # Mocked
    }
    
    for item in sentiment_counts:
        index = item['day'] - 1 # Convert DB's 1-7 index to Python's 0-6 index
        
        if 0 <= index < 7:
            if item['result'] == 'Positive':
                trend_data['positive'][index] = item['count']
            elif item['result'] == 'Negative':
                trend_data['negative'][index] = item['count']
    
    return trend_data


def bulk_analyze_csv(uploaded_file, column_name, algorithm, request):
    """
    Reads a CSV file, analyzes the specified column, and returns a summary.
    """
    summary = {'total': 0, 'Positive': 0, 'Negative': 0, 'Neutral': 0}
    new_analysis_records = []
    
    try:
        file_data = uploaded_file.read().decode('utf-8')
        csv_data = csv.reader(io.StringIO(file_data))
        header = next(csv_data)
        
        try:
            text_column_index = header.index(column_name)
        except ValueError:
            text_column_index = 0
            
        
        for row in csv_data:
            if len(row) > text_column_index:
                text = row[text_column_index].strip()
                
                if not clean_text(text): 
                    continue
                
                pred_label, _ = predict_sentiment(text, model_type=algorithm)
                
                summary[pred_label] += 1
                summary['total'] += 1
                
                new_analysis_records.append(
                    AnalysisRecord(
                        input_text=text[:1000],
                        result=pred_label,
                        algorithm=algorithm
                    )
                )
                if pred_label in ('Positive', 'Negative'):
                    update_word_counts(text, pred_label)
                    
        if new_analysis_records:
            AnalysisRecord.objects.bulk_create(new_analysis_records)

        if summary['total'] > 0:
            summary['Positive_perc'] = f"{(summary['Positive'] / summary['total']) * 100:.2f}"
            summary['Negative_perc'] = f"{(summary['Negative'] / summary['total']) * 100:.2f}"
            summary['Neutral_perc'] = f"{(summary['Neutral'] / summary['total']) * 100:.2f}"
        
        return summary
    
    except Exception as e:
        return {'total': 0, 'Positive': 0, 'Negative': 0, 'Neutral': 0, 'error': str(e)}


# sentiment_app/views.py (Only the corrected dashboard function is shown)

def dashboard(request):
    
    sentiment_trend_data = get_sentiment_trend_data()
    performance_data = [75, 82, 90] 

    # 1. Initialize context with data that doesn't change based on POST, or needs defaults
    context = {
        'result': None,
        'probs_percent': None,
        'bulk_summary': None, 
        'performance_data': performance_data,
        'sentiment_trend_data': json.dumps(sentiment_trend_data),
        'analysis_history': AnalysisRecord.objects.all().order_by('-analyzed_at')[:20],
        # Temporarily set to None/Defaults, they will be fetched later in POST or before GET render
        'positive_words': None,
        'negative_words': None,
    }

    if request.method == 'POST':
        input_text = request.POST.get('input_text', '').strip()
        algorithm = request.POST.get('algorithm', 'lr')
        csv_file = request.FILES.get('csv_file')
        column_name = request.POST.get('column_name', 'text')

        model_key_map = {'lr': 'lr', 'svc': 'svc', 'lstm': 'lstm'}
        model_key = model_key_map.get(algorithm) 

        if not model_key:
            # Re-fetch the dynamic context data before rendering on error/no model
            context['positive_words'] = json.dumps(get_word_cloud_data('Positive'))
            context['negative_words'] = json.dumps(get_word_cloud_data('Negative'))
            return render(request, 'dashboard.html', context)
        
        
        if csv_file:
            # The bulk_analyze_csv function already calls update_word_counts internally
            summary = bulk_analyze_csv(csv_file, column_name, model_key, request)
            context['bulk_summary'] = summary
            
        elif input_text:
            pred_label, probs_percent = predict_sentiment(input_text, model_type=model_key)
            
            context['result'] = pred_label
            context['probs_percent'] = probs_percent
            
            # Save analysis and update word counts
            AnalysisRecord.objects.create(
                input_text=input_text,
                result=pred_label,
                algorithm=algorithm
            )
            
            # This is the call that updates the database
            if pred_label in ('Positive', 'Negative'):
                update_word_counts(input_text, pred_label)
        
    # 2. FINAL DATA FETCH (CRITICAL STEP)
    # This block executes after POST data is saved OR for an initial GET request.
    # It ensures the word cloud data reflects the most recent database changes.
    context['analysis_history'] = AnalysisRecord.objects.all().order_by('-analyzed_at')[:20]
    context['positive_words'] = json.dumps(get_word_cloud_data('Positive'))
    context['negative_words'] = json.dumps(get_word_cloud_data('Negative'))
    context['sentiment_trend_data'] = json.dumps(get_sentiment_trend_data())
        
    return render(request, 'dashboard.html', context)           
    