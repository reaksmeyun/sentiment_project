# sentiment_app/views.py

import os
import json
import re
import csv
import io
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
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

# Load models
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
except Exception:
    MODELS_LOADED = False

# Load NLTK stopwords (already downloaded in Docker)
try:
    from nltk.corpus import stopwords as _nltk_sw
    _NLTK_STOPS = set(_nltk_sw.words('english'))
except Exception:
    _NLTK_STOPS = set()

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _TERM_SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
except Exception:
    # The Docker image installs the lexicon.  Keeping this optional makes the
    # application usable in a fresh local checkout before NLTK data is fetched.
    _TERM_SENTIMENT_ANALYZER = None

# Small fallback for local/test environments without NLTK's VADER lexicon.
# Production uses VADER's much larger curated lexicon.
_FALLBACK_TERM_SCORES = {
    'amazing': 2.8, 'awesome': 3.1, 'best': 3.2, 'better': 1.9,
    'excellent': 2.7, 'favorite': 2.0, 'good': 1.9, 'great': 3.1,
    'happy': 2.7, 'helpful': 1.9, 'love': 3.2, 'loved': 3.2,
    'perfect': 2.7, 'recommend': 2.0, 'satisfied': 1.8,
    'bad': -2.5, 'careless': -1.9, 'disappointing': -2.2,
    'disappointed': -2.1, 'frustrating': -2.0, 'hate': -2.7,
    'horrible': -2.5, 'poor': -2.1, 'sad': -2.1, 'terrible': -3.1,
    'unacceptable': -2.0, 'worst': -3.1,
}

NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'cannot', 'cant', "can't",
    'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't", 'isnt',
    "isn't", 'wasnt', "wasn't", 'werent', "weren't", 'wont', "won't",
    'wouldnt', "wouldn't", 'couldnt', "couldn't", 'shouldnt', "shouldn't",
}

# Comprehensive noise word list — pronouns, auxiliaries, filler words
# These add no insight to sentiment word clouds.
STOP_WORDS = _NLTK_STOPS | {
    # Built-in baseline so filtering still works when NLTK data is unavailable
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as',
    'this', 'that', 'these', 'those', 'is', 'am', 'are', 'was', 'were',
    'be', 'been', 'being', 'to', 'of', 'in', 'on', 'at', 'for', 'from',
    'with', 'by', 'not', 'no',
    # Pronouns
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    # Contractions / informal
    'im', 'ive', 'id', 'ill', 'youre', 'youve', 'youd', 'youll',
    'hes', 'shes', 'its', 'weve', 'theyre', 'theyve', 'dont', 'doesnt',
    'didnt', 'cant', 'couldnt', 'shouldnt', 'wouldnt', 'isnt', 'arent',
    'wasnt', 'werent', 'wont', 'havent', 'hadnt', 'hasnt',
    # Common filler / noise
    'get', 'got', 'go', 'went', 'come', 'came', 'make', 'made',
    'know', 'think', 'said', 'say', 'tell', 'told', 'going', 'getting',
    'one', 'two', 'also', 'even', 'back', 'still', 'much', 'many',
    'well', 'just', 'really', 'very', 'quite', 'pretty', 'actually',
    'always', 'never', 'ever', 'every', 'any', 'some', 'another',
    'there', 'here', 'when', 'where', 'why', 'how', 'then', 'than',
    'first', 'last', 'next', 'like', 'use', 'using', 'used',
    'would', 'could', 'should', 'might', 'must', 'shall', 'may',
    'place', 'time', 'thing', 'things', 'way', 'lot', 'bit',
    # People / generic nouns that add no sentiment insight
    'man', 'men', 'woman', 'women', 'girl', 'girls', 'boy', 'boys',
    'guy', 'guys', 'gal', 'gals', 'person', 'people', 'someone',
    'everyone', 'anyone', 'nobody', 'somebody', 'anybody',
    # Time words
    'ago', 'year', 'years', 'month', 'months', 'week', 'weeks',
    'day', 'days', 'hour', 'hours', 'minute', 'minutes', 'second',
    'today', 'yesterday', 'tomorrow', 'morning', 'evening', 'night',
    # Common verbs with no sentiment value
    'feel', 'felt', 'felt', 'seen', 'look', 'looks', 'looked', 'looking',
    'need', 'needs', 'needed', 'want', 'wants', 'wanted', 'try', 'tried',
    'take', 'took', 'taken', 'find', 'found', 'seem', 'seems', 'seemed',
    'left', 'right', 'watch', 'watched', 'watching', 'show', 'shows',
    'start', 'started', 'stop', 'stopped', 'keep', 'kept', 'put',
    'give', 'gave', 'given', 'let', 'set', 'ask', 'asked', 'asked',
    'work', 'works', 'worked', 'play', 'plays', 'played', 'happen',
    'happens', 'happened', 'remember', 'sure', 'actually', 'thank',
    # Entertainment / generic
    'video', 'videos', 'channel', 'content', 'game', 'games',
    'music', 'song', 'songs', 'movie', 'movies', 'show', 'shows',
    'episode', 'series', 'season', 'clip', 'clips', 'comment', 'post',
    'watch', 'watching', 'like', 'likes', 'share', 'subscribe',
    'click', 'link', 'page', 'site', 'app', 'data', 'model',
    # Misc
    'new', 'old', 'big', 'small', 'long', 'short', 'high', 'low',
    'many', 'much', 'more', 'most', 'less', 'least', 'few', 'little',
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


def extract_word_cloud_words(text):
    """Return useful words for the sentiment word clouds.

    Three-letter words are intentionally included because common sentiment words
    such as ``bad`` and ``sad`` would otherwise disappear from the cloud.
    """
    return [
        word
        for word in clean_text(text).split()
        if len(word) >= 3 and word not in STOP_WORDS
    ]


def extract_sentiment_terms(text):
    """Return ``(term, sentiment)`` pairs using term-level polarity.

    Only words that carry sentiment are returned; topic nouns such as
    ``restaurant`` and ``product`` are deliberately excluded. A negator in the
    previous three words reverses polarity and is retained in the display term,
    so ``not good`` is a negative phrase instead of a positive ``good``.
    """
    if not isinstance(text, str):
        return []

    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?,;:]", text.lower())
    recent_words = []
    result = []

    for token in tokens:
        if token in '.!?,;:':
            recent_words = []
            continue

        score = 0.0
        if _TERM_SENTIMENT_ANALYZER is not None:
            score = _TERM_SENTIMENT_ANALYZER.lexicon.get(token, 0.0)
        if not score:
            score = _FALLBACK_TERM_SCORES.get(token, 0.0)

        negator = next(
            (word for word in reversed(recent_words[-3:]) if word in NEGATION_WORDS),
            None,
        )
        if score and negator:
            score = -score

        if score:
            display_term = f'{negator} {token}' if negator else token
            result.append((display_term, 'Positive' if score > 0 else 'Negative'))

        recent_words.append(token)

    return result


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


def update_word_counts(text, sentiment=None):
    """
    Tokenizes text, filters by stopwords, and updates the WordCount model.
    (Used for single-text analysis only).
    """
    for word, term_sentiment in extract_sentiment_terms(text):
        word_record, created = WordCount.objects.get_or_create(
            word=word,
            sentiment=term_sentiment,
            defaults={'count': 1}
        )
        if not created:
            word_record.count += 1
            word_record.save(update_fields=['count'])


def collect_words(text, sentiment, words_to_update):
    """
    Aggregates word counts in a dictionary using stopword filtering.
    (Used for bulk CSV analysis).
    """
    for word, term_sentiment in extract_sentiment_terms(text):
        words_to_update[term_sentiment][word] = words_to_update[term_sentiment].get(word, 0) + 1


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
    """Returns top words with pre-calculated display sizes for server-side rendering."""
    qs = list(WordCount.objects.filter(sentiment=sentiment).order_by('-count')[:50])
    if not qs:
        return []
    max_c = qs[0].count
    min_c = qs[-1].count
    spread = max(max_c - min_c, 1)
    result = []
    for w in qs:
        ratio = (w.count - min_c) / spread          # 0.0 – 1.0
        size  = int(13 + ratio * 38)                 # 13 – 51 px
        alpha = round(0.5 + ratio * 0.5, 2)          # 0.5 – 1.0
        weight = 700 if w.count > 1 else 500
        result.append({'word': w.word, 'count': w.count,
                       'size': size, 'alpha': alpha, 'weight': weight})
    return result


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
    """Reads a CSV file, analyzes the specified column, returns a summary."""
    summary = {'total': 0, 'Positive': 0, 'Negative': 0, 'Neutral': 0}
    new_analysis_records = []
    words_to_update = {'Positive': {}, 'Negative': {}}

    # Try multiple encodings — many CSVs are not pure UTF-8
    raw_bytes = uploaded_file.read()
    file_data = None
    for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
        try:
            file_data = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if file_data is None:
        return {**summary, 'error': 'Cannot read file. Please save your CSV as UTF-8 and try again.'}

    try:
        csv_data = csv.reader(io.StringIO(file_data))
        header = next(csv_data, None)
        if header is None:
            return {**summary, 'error': 'CSV file is empty.'}

        # Case-insensitive, whitespace-stripped column matching
        col_lower = column_name.strip().lower()
        header_clean = [h.strip().lower() for h in header]
        if col_lower in header_clean:
            text_column_index = header_clean.index(col_lower)
        else:
            # fallback to first column; tell user which columns exist
            text_column_index = 0
            summary['column_warning'] = (
                f"Column '{column_name}' not found. "
                f"Available columns: {', '.join(header)}. "
                f"Using first column instead."
            )

        for row in csv_data:
            if len(row) <= text_column_index:
                continue
            text = row[text_column_index].strip()
            if not clean_text(text):
                continue

            pred_label, _ = predict_sentiment(text, model_type=algorithm)
            summary[pred_label] += 1
            summary['total'] += 1

            new_analysis_records.append(
                AnalysisRecord(input_text=text[:1000], result=pred_label, algorithm=algorithm)
            )

            if pred_label in ('Positive', 'Negative'):
                try:
                    collect_words(text, pred_label, words_to_update)
                except Exception:
                    pass

        if new_analysis_records:
            AnalysisRecord.objects.bulk_create(new_analysis_records)

        bulk_update_word_counts(words_to_update)

        if summary['total'] > 0:
            summary['Positive_perc'] = f"{summary['Positive'] / summary['total'] * 100:.1f}"
            summary['Negative_perc'] = f"{summary['Negative'] / summary['total'] * 100:.1f}"
            summary['Neutral_perc']  = f"{summary['Neutral']  / summary['total'] * 100:.1f}"
        else:
            summary['error'] = 'No valid rows were found in the CSV. Check the column name and file contents.'

        return summary

    except Exception as e:
        return {**summary, 'error': f'Error processing CSV: {str(e)}'}


@csrf_exempt
def dashboard(request):
    
    performance_data = [75, 82, 90] 

    # 1. Initialize context with defaults for form fields
    context = {
        'result': None,
        'probs_percent': None,
        'bulk_summary': None, 
        'performance_data': performance_data,
        'analysis_history': AnalysisRecord.objects.all().order_by('-analyzed_at')[:20],
        'positive_words': [],
        'negative_words': [],
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
                try:
                    update_word_counts(input_text, pred_label)
                except Exception:
                    pass
        
    # 2. FINAL DATA FETCH (Runs for both GET and POST)
    # This must run after the POST logic so it can fetch the updated history/word counts
    context['analysis_history'] = AnalysisRecord.objects.all().order_by('-analyzed_at')[:20]
    context['positive_words'] = get_word_cloud_data('Positive')
    context['negative_words'] = get_word_cloud_data('Negative')
    context['sentiment_trend_data'] = json.dumps(get_sentiment_trend_data())
        
    return render(request, 'dashboard.html', context)
