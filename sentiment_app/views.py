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

# 🚨 NEW: Import NLTK for smart word cloud filtering
import nltk 

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


# 🚨 NEW: Aggressive Stop Word List for Smart Filtering
# This list ensures common noise words are excluded even if their POS tag is accepted.
VERY_AGGRESSIVE_STOP_WORDS = {
    # Basic Stop Words
    'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'i', 'my', 'that', 
    
    # Words Cluttering Your Word Cloud
    'was', 'for', 'you', 'with', 'have', 'are', 'they', 'them', 'but', 
    'were', 'what', 'more', 'which', 'than', 'some', 'when', 'all', 'from',
    'get', 'dont', 'there', 'around', 'came', 'like', 'any', 'pretty', 
    'not', 'these', 'had', 'few', 'place', 'about', 'this', 'one', 'our', 
    'does', 'got', 'very', 'here', 'back', 'can', 'or', 'just', 'only', 
    'really', 'would', 'could', 'know', 'said', 'go', 'went', 'us', 'we', 
    'little', 'much', 'too', 'making', 'make', 'use', 'using', 'next', 'first',
    
    # Topic Words (Customize this for your specific reviews)
    'pizza', 'food', 'service', 'restaurant', 'order', 'menu', 'salsa', 
    'cream', 'fish', 'calzone', 'app', 'model', 'data'
}


# ----------------------------------------
# --- HELPER FUNCTIONS ---
# ----------------------------------------

def clean_text(text):
    """Standardizes text cleaning."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase and strip whitespace
    text = text.lower().strip()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def predict_sentiment(text, model_type='lr'):
    # ... (function body remains the same as it doesn't need POS tagging)
    if not MODELS_LOADED:
        return "Neutral", ["50.00%", "50.00%"]

    text = clean_text(text)
    if not text:
        return "Neutral", ["50.00%", "50.00%"]

    probs = [0.5, 0.5]
    CONFIDENCE_THRESHOLD = 0.60

    try:
        if model_type == 'lr':
            X = tfidf_vectorizer.transform([text])
            probs = logistic_model.predict_proba(X)[0]
        elif model_type == 'svc':
            X = tfidf_vectorizer.transform([text])
            decision = svc_model.decision_function(X)[0]
            probs = softmax([-decision, decision])
        elif model_type == 'lstm':
            seq = lstm_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            probs = lstm_model.predict(padded, verbose=0)[0] 
        
        # --- ROBUST LABEL ASSIGNMENT ---
        max_prob_index = probs.argmax()
        
        if probs[max_prob_index] > CONFIDENCE_THRESHOLD: 
            if max_prob_index == 1:
                final_label = 'Positive'
            elif max_prob_index == 0:
                final_label = 'Negative'
            else:
                final_label = 'Neutral' 
        else:
            final_label = 'Neutral'
            
        pred_label = final_label

    except Exception as e:
        pred_label = 'Neutral'
        
    probs_percent = [f"{p*100:.2f}%" for p in probs]
    return pred_label, probs_percent


def update_word_counts(text, sentiment):
    """
    Tokenizes text, filters by Part-of-Speech, and updates the WordCount model.
    (Used for single-text analysis only). 🚨 NOW USES POS TAGGING
    """
    cleaned_text = clean_text(text)
    
    # 1. Tokenize and POS tag the cleaned text
    words_and_tags = nltk.pos_tag(nltk.word_tokenize(cleaned_text))
    
    # 2. Define POS tags to keep (Adjectives, Adverbs, Verbs)
    SENTIMENT_TAGS = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN'}
    
    for word, tag in words_and_tags:
        word = word.lower()
        
        # 3. Apply smart filtering logic:
        is_sentiment_word = tag in SENTIMENT_TAGS
        is_stop_word = word in VERY_AGGRESSIVE_STOP_WORDS
        is_long_enough = len(word) >= 3

        if is_sentiment_word and not is_stop_word and is_long_enough:
            word_record, created = WordCount.objects.get_or_create(
                word=word,
                sentiment=sentiment,
                defaults={'count': 1}
            )
            if not created:
                word_record.count += 1
                word_record.save()


def collect_words(text, sentiment, words_to_update):
    """
    Aggregates word counts in a dictionary using POS filtering.
    (Used for bulk CSV analysis). 🚨 NOW USES POS TAGGING
    """
    cleaned_text = clean_text(text)
    
    # 1. Tokenize and POS tag the cleaned text
    words_and_tags = nltk.pos_tag(nltk.word_tokenize(cleaned_text))
    
    # 2. Define POS tags to keep
    SENTIMENT_TAGS = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN'}

    for word, tag in words_and_tags:
        word = word.lower()
        
        # 3. Apply smart filtering logic:
        is_sentiment_word = tag in SENTIMENT_TAGS
        is_stop_word = word in VERY_AGGRESSIVE_STOP_WORDS
        is_long_enough = len(word) >= 3

        if is_sentiment_word and not is_stop_word and is_long_enough:
            # Aggregate count in the dictionary (in memory)
            words_to_update[sentiment][word] = words_to_update[sentiment].get(word, 0) + 1


def bulk_update_word_counts(words_to_update):
    """Efficiently updates the WordCount table using the aggregated dictionary."""
    for sentiment, word_counts in words_to_update.items():
        for word, count_increment in word_counts.items():
            # Try to update existing record
            try:
                word_record = WordCount.objects.get(word=word, sentiment=sentiment)
                word_record.count += count_increment
                word_record.save()
            # If not found, create a new record
            except WordCount.DoesNotExist:
                WordCount.objects.create(word=word, sentiment=sentiment, count=count_increment)


def get_word_cloud_data(sentiment):
    """Retrieves top words formatted for wordcloud2.js."""
    words = WordCount.objects.filter(sentiment=sentiment).order_by('-count')[:50]
    return [[w.word, w.count] for w in words]


def get_sentiment_trend_data():
    """Retrieves sentiment trend data dynamically from the database for the last 7 days."""
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
        'accuracy_trend': [75, 78, 80, 82, 85, 87, 85] # Mocked/Placeholder
    }
    
    for item in sentiment_counts:
        index = item['day'] - 1 
        
        if 0 <= index < 7:
            if item['result'] == 'Positive':
                trend_data['positive'][index] = item['count']
            elif item['result'] == 'Negative':
                trend_data['negative'][index] = item['count']
    
    return trend_data


# ----------------------------------------
# --- MAIN VIEW FUNCTIONS ---
# ----------------------------------------

def bulk_analyze_csv(uploaded_file, column_name, algorithm, request):
    """
    Reads a CSV file, analyzes the specified column, and returns a summary.
    🚨 OPTIMIZED: Aggregates word counts in memory before bulk updating the DB.
    """
    summary = {'total': 0, 'Positive': 0, 'Negative': 0, 'Neutral': 0}
    new_analysis_records = []
    
    words_to_update = {'Positive': {}, 'Negative': {}} 
    
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
                
                # Calls the smart, POS-filtered collector
                if pred_label in ('Positive', 'Negative'):
                    collect_words(text, pred_label, words_to_update)
                        
        if new_analysis_records:
            AnalysisRecord.objects.bulk_create(new_analysis_records)

        # Efficiently update WordCount table once after the loop
        bulk_update_word_counts(words_to_update)

        if summary['total'] > 0:
            summary['Positive_perc'] = f"{(summary['Positive'] / summary['total']) * 100:.2f}"
            summary['Negative_perc'] = f"{(summary['Negative'] / summary['total']) * 100:.2f}"
            summary['Neutral_perc'] = f"{(summary['Neutral'] / summary['total']) * 100:.2f}"
        
        return summary
    
    except Exception as e:
        return {'total': 0, 'Positive': 0, 'Negative': 0, 'Neutral': 0, 'error': str(e)}


def dashboard(request):
    
    performance_data = [75, 82, 90] 

    # 1. Initialize context with defaults for form fields
    context = {
        'result': None,
        'probs_percent': None,
        'bulk_summary': None, 
        'performance_data': performance_data,
        'analysis_history': AnalysisRecord.objects.all().order_by('-analyzed_at')[:20],
        'positive_words': None,
        'negative_words': None,
        'sentiment_trend_data': None,
        
        # 💡 Set defaults for the form fields
        'selected_algorithm': 'lr', 
        'column_name': 'text',      
        'input_text': '',           
    }

    if request.method == 'POST':
        input_text = request.POST.get('input_text', '').strip()
        algorithm = request.POST.get('algorithm', 'lr')
        csv_file = request.FILES.get('csv_file')
        column_name = request.POST.get('column_name', 'text')

        # 🚀 FIX: Update context with submitted values to maintain form state
        context['selected_algorithm'] = algorithm
        context['column_name'] = column_name
        context['input_text'] = input_text
        
        model_key_map = {'lr': 'lr', 'svc': 'svc', 'lstm': 'lstm'}
        model_key = model_key_map.get(algorithm) 

        # We keep the old check for non-valid algorithm, but the state is already set
        if not model_key:
            pass # The rest of the page data loading below will still run
        
        elif csv_file:
            summary = bulk_analyze_csv(csv_file, column_name, model_key, request)
            context['bulk_summary'] = summary
            
        elif input_text:
            pred_label, probs_percent = predict_sentiment(input_text, model_type=model_key)
            
            context['result'] = pred_label
            context['probs_percent'] = probs_percent
            
            AnalysisRecord.objects.create(
                input_text=input_text,
                result=pred_label,
                algorithm=algorithm
            )
            
            # Calls the smart, POS-filtered single-text updater
            if pred_label in ('Positive', 'Negative'):
                update_word_counts(input_text, pred_label)
        
    # 2. FINAL DATA FETCH (Runs for both GET and POST)
    # This must run after the POST logic so it can fetch the updated history/word counts
    context['analysis_history'] = AnalysisRecord.objects.all().order_by('-analyzed_at')[:20]
    context['positive_words'] = json.dumps(get_word_cloud_data('Positive'))
    context['negative_words'] = json.dumps(get_word_cloud_data('Negative'))
    context['sentiment_trend_data'] = json.dumps(get_sentiment_trend_data())
        
    return render(request, 'dashboard.html', context)