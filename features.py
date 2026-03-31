"""
model/features.py
Text cleaning, sentiment scoring, and feature matrix construction.
"""

import re
import string
import numpy as np
import pandas as pd
from typing import Tuple, List

# ── Optional heavy imports (graceful fallback) ─────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # Download required NLTK data silently
    for pkg in ["punkt", "stopwords", "wordnet", "vader_lexicon", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    VADER = SentimentIntensityAnalyzer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    STOP_WORDS = set()
    LEMMATIZER = None
    VADER = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Label mapping ─────────────────────────────────────────────────────────────

def label_from_rating(rating: float) -> str:
    """Convert numeric star rating to a three-class label."""
    if rating >= 4.0:
        return "Buy"
    elif rating >= 3.0:
        return "Caution"
    else:
        return "Don't Buy"


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove HTML/URLs/punctuation, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)     # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)           # keep letters only
    text = re.sub(r"\s+", " ", text).strip()

    if NLTK_AVAILABLE and LEMMATIZER:
        tokens = text.split()
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
        text = " ".join(tokens)
    return text


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def vader_score(text: str) -> float:
    """Return VADER compound score in [-1, 1]. Falls back to 0 if unavailable."""
    if not NLTK_AVAILABLE or VADER is None:
        return 0.0
    return VADER.polarity_scores(str(text))["compound"]


def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


# ── Feature matrix ────────────────────────────────────────────────────────────

# Global TF-IDF vectorizer (fitted once during training, reused at inference)
_tfidf: "TfidfVectorizer | None" = None
_feature_names: List[str] = []


def build_feature_matrix(
    texts: pd.Series,
    ratings: pd.Series,
    fit: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Combine TF-IDF features with numeric/sentiment features.

    Parameters
    ----------
    texts   : raw review texts
    ratings : star ratings (1–5)
    fit     : if True, fit the TF-IDF vectorizer (training). If False, transform only (inference).

    Returns
    -------
    X              : feature matrix (dense numpy array)
    feature_names  : list of feature column names
    """
    global _tfidf, _feature_names

    cleaned = texts.apply(clean_text)

    # TF-IDF
    if fit or _tfidf is None:
        _tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = _tfidf.fit_transform(cleaned)
    else:
        tfidf_matrix = _tfidf.transform(cleaned)

    tfidf_names = [f"tfidf_{t}" for t in _tfidf.get_feature_names_out()]

    # Sentiment + numeric features
    sentiments = texts.apply(vader_score).values
    rating_arr = pd.to_numeric(ratings, errors="coerce").fillna(3.0).values

    numeric = np.column_stack([
        sentiments,                       # VADER compound
        (sentiments >= 0.05).astype(int), # is_positive
        (sentiments <= -0.05).astype(int),# is_negative
        rating_arr,                        # star rating
        (rating_arr >= 4).astype(int),    # high_rating flag
        (rating_arr <= 2).astype(int),    # low_rating flag
    ])
    numeric_names = [
        "sentiment_compound", "is_positive", "is_negative",
        "star_rating", "high_rating_flag", "low_rating_flag",
    ]

    # Combine sparse TF-IDF with dense numeric
    numeric_sparse = csr_matrix(numeric)
    X = hstack([tfidf_matrix, numeric_sparse])
    _feature_names = tfidf_names + numeric_names

    # Convert to dense array for sklearn classifiers that need it
    return X.toarray(), _feature_names


def get_tfidf() -> "TfidfVectorizer | None":
    return _tfidf
