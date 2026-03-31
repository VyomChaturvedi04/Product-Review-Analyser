"""
tests/test_features.py
Unit tests for the feature engineering and prediction pipeline.

Run with:
    pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.features import clean_text, vader_score, label_from_rating, sentiment_label


# ── label_from_rating ─────────────────────────────────────────────────────────

class TestLabelFromRating:
    def test_high_rating_is_buy(self):
        assert label_from_rating(5.0) == "Buy"
        assert label_from_rating(4.0) == "Buy"

    def test_mid_rating_is_caution(self):
        assert label_from_rating(3.5) == "Caution"
        assert label_from_rating(3.0) == "Caution"

    def test_low_rating_is_dont_buy(self):
        assert label_from_rating(2.9) == "Don't Buy"
        assert label_from_rating(1.0) == "Don't Buy"


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD").lower() == clean_text("hello world").lower()

    def test_removes_html(self):
        result = clean_text("<b>Great</b> product!")
        assert "<b>" not in result
        assert "great" in result

    def test_removes_urls(self):
        result = clean_text("Check https://amazon.com for details")
        assert "http" not in result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_returns_string(self):
        assert isinstance(clean_text("hello"), str)


# ── vader_score ───────────────────────────────────────────────────────────────

class TestVaderScore:
    def test_positive_text(self):
        score = vader_score("This product is absolutely amazing and wonderful!")
        assert score > 0.0, f"Expected positive score, got {score}"

    def test_negative_text(self):
        score = vader_score("This is terrible, broken garbage. Awful experience.")
        assert score < 0.0, f"Expected negative score, got {score}"

    def test_score_range(self):
        for text in ["great", "bad", "okay", ""]:
            s = vader_score(text)
            assert -1.0 <= s <= 1.0, f"Score {s} out of range for text: {text!r}"

    def test_returns_float(self):
        assert isinstance(vader_score("hello"), float)


# ── sentiment_label ───────────────────────────────────────────────────────────

class TestSentimentLabel:
    def test_positive(self):
        assert sentiment_label(0.5) == "positive"
        assert sentiment_label(0.05) == "positive"

    def test_negative(self):
        assert sentiment_label(-0.5) == "negative"
        assert sentiment_label(-0.05) == "negative"

    def test_neutral(self):
        assert sentiment_label(0.0) == "neutral"
        assert sentiment_label(0.04) == "neutral"
        assert sentiment_label(-0.04) == "neutral"


# ── Integration: build_feature_matrix ────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_basic_shape(self):
        from model.features import build_feature_matrix
        texts = pd.Series(["Great product!", "Terrible quality.", "It is okay."])
        ratings = pd.Series([5.0, 1.0, 3.0])
        X, names = build_feature_matrix(texts, ratings, fit=True)
        assert X.shape[0] == 3, "Should have one row per review"
        assert X.shape[1] == len(names), "Feature count mismatch"

    def test_no_nan_in_features(self):
        from model.features import build_feature_matrix
        texts = pd.Series(["Good item", "Bad item", "Average item", "Love it", "Hate it"])
        ratings = pd.Series([4.0, 2.0, 3.0, 5.0, 1.0])
        X, _ = build_feature_matrix(texts, ratings, fit=True)
        assert not np.any(np.isnan(X)), "Feature matrix contains NaN values"
