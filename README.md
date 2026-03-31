# Amazon Review Analyzer

**Should you buy it?** Paste Amazon product reviews and get an AI-powered **Buy / Caution / Don't Buy** verdict — powered by NLP sentiment analysis and a Random Forest classifier.

> BYOP (Bring Your Own Project) submission for **Fundamentals of AI and ML**

---

## Demo

| Input | Verdict |
|---|---|
| Paste 1–20 reviews + star rating | Buy / Caution / Don't Buy |

The app shows:
- A verdict banner with confidence percentage
- Probability bars for all three classes
- Per-review sentiment breakdown (VADER scores)
- Plain-English explanation of the verdict

---

## Problem Statement

Millions of Amazon listings carry fake or biased reviews. Shoppers can't manually read hundreds of reviews before every purchase. This tool aggregates review sentiment, combines it with star rating signals, and gives a single actionable recommendation using machine learning.

---

## How It Works

```
Raw reviews
    │
    ▼
Text cleaning (lowercase, remove HTML/URLs, lemmatize)
    │
    ▼
Feature engineering
    ├── TF-IDF vectors (5000 n-grams, unigrams + bigrams)
    └── Sentiment features (VADER compound score, pos/neg flags, star rating)
    │
    ▼
Random Forest Classifier  ←  trained on Amazon Fine Food Reviews
    │
    ▼
Verdict: Buy / Caution / Don't Buy  +  confidence %
```

**Labels are derived from star ratings:**
| Stars | Label |
|---|---|
| ≥ 4.0 | Buy |
| 3.0 – 3.9 | Caution |
| ≤ 2.9 | Don't Buy |

---

## Project Structure

```
amazon-review-analyzer/
│
├── app.py                    ← Streamlit web app (main entry point)
│
├── model/
│   ├── __init__.py
│   ├── features.py           ← Text cleaning + feature engineering
│   ├── train.py              ← Model training script
│   └── predict.py            ← Inference + explanation helpers
│   └── artifacts/            ← Saved model (generated after training)
│
├── data/
│   ├── README.md             ← How to get the dataset
│   └── generate_sample.py    ← Creates a 500-review synthetic sample
│
├── notebooks/
│   └── eda.ipynb             ← Exploratory Data Analysis
│
├── tests/
│   └── test_features.py      ← Unit tests (pytest)
│
├── assets/                   ← Generated plots (EDA outputs)
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/amazon-review-analyzer.git
cd amazon-review-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

NLTK data is downloaded automatically on first run.

---

## Running the App

### Step 1 — Get data and train the model

**Option A — Synthetic sample (no Kaggle account needed):**
```bash
python data/generate_sample.py
python model/train.py --data data/reviews_sample.csv
```

**Option B — Full Kaggle dataset (recommended for best accuracy):**
1. Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
2. Place it in the `data/` directory
3. Run:
```bash
python model/train.py --data data/Reviews.csv
```

Training output includes accuracy, F1 score, and a confusion matrix saved to `model/artifacts/`.

### Step 2 — Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Exploratory Data Analysis

Open the EDA notebook to explore the dataset and visualizations:

```bash
jupyter notebook notebooks/eda.ipynb
```

Covers: rating distribution, VADER sentiment vs stars, review length analysis, top discriminative words per class, and feature correlation.

---

## Tech Stack

| Component | Library |
|---|---|
| NLP / Sentiment | NLTK, VADER |
| Feature Engineering | scikit-learn TF-IDF |
| Classification | scikit-learn Random Forest |
| Web App | Streamlit |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Testing | pytest |

---

## Model Performance

Trained on 80% of the Amazon Fine Food Reviews dataset (~455k reviews), tested on 20%:

| Metric | Score |
|---|---|
| Accuracy | ~85% |
| Macro F1 | ~0.78 |
| Buy F1 | ~0.91 |
| Caution F1 | ~0.52 |
| Don't Buy F1 | ~0.89 |

*Caution is the hardest class to classify due to its ambiguous nature — a known limitation.*

---

## Limitations

- Caution class accuracy is lower due to mixed sentiment in those reviews
- Model trained on food reviews; performance may vary on electronics, clothing, etc.
- Does not detect fake review patterns (e.g., unusually similar review text)
- No scraping — users must manually paste reviews

---

## Future Work

- Integrate a BERT-based model for better contextual understanding
- Add a browser extension to analyze reviews directly on Amazon product pages
- Fake review detection using clustering
- Multi-language support