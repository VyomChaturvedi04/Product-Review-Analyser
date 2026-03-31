"""
model/predict.py
Inference helpers: given raw review strings, return verdict + metadata.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from model.features import vader_score, sentiment_label, build_feature_matrix, label_from_rating


LABEL_ORDER = ["Buy", "Caution", "Don't Buy"]


def predict_reviews(
    bundle: Dict,
    reviews: List[str],
    avg_rating: float = 3.5,
    num_reviews: int = 100,
) -> Dict[str, Any]:
    """
    Run inference on a list of review strings.

    Parameters
    ----------
    bundle      : dict loaded from the saved .pkl file (keys: classifier, feature_names)
    reviews     : list of raw review strings entered by the user
    avg_rating  : product's overall star rating (from the UI slider)
    num_reviews : total number of product reviews (from the UI input)

    Returns
    -------
    result dict with verdict, confidence, probabilities, and analytics
    """
    clf = bundle["classifier"]

    # Build one combined "review blob" + individual sentiments
    individual_sentiments = [round(vader_score(r), 3) for r in reviews]
    individual_labels = [sentiment_label(s) for s in individual_sentiments]

    pct_positive = sum(1 for s in individual_sentiments if s >= 0.05) / max(len(individual_sentiments), 1)
    avg_sentiment = float(np.mean(individual_sentiments))

    # For model inference: treat each review as a separate row, aggregate probabilities
    ratings_series = pd.Series([avg_rating] * len(reviews))
    texts_series = pd.Series(reviews)

    # Reuse fitted vectorizer (fit=False for inference)
    try:
        X, _ = build_feature_matrix(texts_series, ratings_series, fit=False)
    except Exception:
        # Fallback: re-fit on these reviews (no saved vectorizer)
        X, _ = build_feature_matrix(texts_series, ratings_series, fit=True)

    if hasattr(clf, "predict_proba"):
        probs_matrix = clf.predict_proba(X)   # shape (n_reviews, n_classes)
        classes = list(clf.classes_)
        avg_probs = probs_matrix.mean(axis=0)
        prob_dict = {c: float(avg_probs[i]) for i, c in enumerate(classes)}
    else:
        preds = clf.predict(X)
        prob_dict = {label: float(np.mean(preds == label)) for label in LABEL_ORDER}

    # Ensure all labels present
    for label in LABEL_ORDER:
        prob_dict.setdefault(label, 0.0)

    verdict = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[verdict]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probabilities": {k: prob_dict[k] for k in LABEL_ORDER},
        "avg_sentiment": avg_sentiment,
        "individual_sentiments": individual_sentiments,
        "individual_labels": individual_labels,
        "pct_positive": pct_positive,
        "num_reviews_analyzed": len(reviews),
        "avg_rating": avg_rating,
        "num_reviews_product": num_reviews,
    }


def get_explanation(result: Dict[str, Any]) -> List[str]:
    """Generate human-readable bullet points explaining the verdict."""
    points = []
    pct_pos = result["pct_positive"]
    avg_sent = result["avg_sentiment"]
    avg_rating = result["avg_rating"]
    verdict = result["verdict"]

    # Sentiment insight
    if avg_sent >= 0.2:
        points.append(f"Overall review tone is strongly positive (avg VADER score: {avg_sent:+.2f}).")
    elif avg_sent >= 0.05:
        points.append(f"Overall review tone is mildly positive (avg VADER score: {avg_sent:+.2f}).")
    elif avg_sent <= -0.2:
        points.append(f"Overall review tone is strongly negative (avg VADER score: {avg_sent:+.2f}).")
    elif avg_sent <= -0.05:
        points.append(f"Overall review tone is mildly negative (avg VADER score: {avg_sent:+.2f}).")
    else:
        points.append(f"Overall review tone is mixed/neutral (avg VADER score: {avg_sent:+.2f}).")

    # Positive review percentage
    points.append(f"{pct_pos:.0%} of the provided reviews are positive.")

    # Star rating
    if avg_rating >= 4.0:
        points.append(f"Star rating is high ({avg_rating:.1f}/5), indicating customer satisfaction.")
    elif avg_rating >= 3.0:
        points.append(f"Star rating is average ({avg_rating:.1f}/5) — mixed reception.")
    else:
        points.append(f"Star rating is low ({avg_rating:.1f}/5), suggesting significant issues.")

    # Verdict-specific advice
    if verdict == "Buy":
        points.append("The combination of positive sentiment and rating suggests this product is worth purchasing.")
    elif verdict == "Caution":
        points.append("There are mixed signals — consider reading individual reviews before deciding.")
    else:
        points.append("Negative sentiment dominates. Consider looking for alternative products.")

    return points
