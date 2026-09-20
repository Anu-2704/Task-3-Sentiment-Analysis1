# Task 4 — Sentiment Analysis

## Overview
This task applies Natural Language Processing (NLP) sentiment analysis to 4,914 Amazon product reviews. Two sentiment models (VADER and TextBlob) are combined with a rule-based emotion detection system to classify reviews, detect emotions, and surface trends in public opinion.

---

## Dataset
| Property | Value |
|---|---|
| File | `amazon_reviews.csv` (one level up) |
| Records analyzed | 4,914 (1 dropped — missing reviewText) |
| Period | January 2012 – December 2014 |
| Product | ASIN B007WTAJTO |

---

## Models & Techniques

| Model | Library | Output |
|---|---|---|
| VADER | `vaderSentiment` | compound, pos, neg, neu scores → Positive/Neutral/Negative label |
| TextBlob | `textblob` | polarity (−1 to +1), subjectivity (0 to 1) |
| Emotion Detection | NLTK + custom lexicon | joy, anger, fear, disgust, sadness, surprise |
| Text Cleaning | `re` + NLTK stopwords | Normalized tokens for analysis |
| Word Cloud | `wordcloud` | Top-word visualizations per sentiment class |

---

## Key Results

| Metric | Value |
|---|---|
| VADER Positive | 3,950 (80.4%) |
| VADER Neutral | 353 (7.2%) |
| VADER Negative | 611 (12.4%) |
| Mean Compound Score | 0.4815 |
| Mean TextBlob Polarity | 0.2791 |
| Mean Subjectivity | 0.5398 |
| Model Agreement Rate (VADER vs Stars) | 80.4% |
| Dominant Emotion | Joy (2,647 reviews — 53.9%) |

---

## Visualizations Produced

| Chart | Type | Insight |
|---|---|---|
| 01 | Bar | VADER sentiment label distribution |
| 02 | Bar | TextBlob sentiment label distribution |
| 03 | Histogram | VADER compound score distribution (bimodal) |
| 04 | Bar + error bars | Mean VADER score by star rating |
| 05 | Scatter | TextBlob polarity vs subjectivity (colored by VADER) |
| 06 | Bar | Dominant emotion distribution |
| 07 | Stacked bar | Emotion composition by star rating |
| 08 | Line + area | Monthly avg VADER score trend (2012–2014) |
| 09 | Heatmap | VADER vs star rating agreement matrix |
| 10 | Histogram | Subjectivity distribution by sentiment class |
| 11 | Word clouds | Positive vs Negative top words side-by-side |
| 12 | 4-panel | Combined narrative story dashboard |

---

## Key Insights

1. **80.4% of reviews are Positive** — strong product satisfaction overall.
2. **Negative reviewers use stronger language** — 1-star reviews score −0.15 mean compound vs +0.55 for 5-star.
3. **Bimodal compound distribution** — most reviews are either strongly positive or strongly negative; few are neutral.
4. **80.4% agreement** between VADER labels and star-based labels. Disagreements often involve mixed reviews (positive stars + critical text).
5. **Joy dominates emotions (53.9%)**; Fear at 6% reflects durability/reliability concerns.
6. **Negative reviews are more factual** (lower subjectivity), while positive reviews are more subjective (personal opinions).
7. **Temporal stability** — monthly avg score stays above 0.3 throughout the dataset with no major sentiment crisis.

---

## Outputs

| File | Description |
|---|---|
| `outputs/sentiment_results.csv` | Full per-review sentiment scores and labels |
| `outputs/summary.json` | Aggregated stats for the frontend |

---

## Project Structure
```
Task_4_Sentiment/
├── sentiment_analysis.py    <- Main analysis script
├── requirements.txt         <- Python dependencies
├── README.md                <- This file
├── dashboard.html           <- Single-page HTML frontend dashboard
├── plots/                   <- 12 generated chart PNGs
└── outputs/
    ├── sentiment_results.csv
    └── summary.json
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the analysis
```bash
# From inside Task_4_Sentiment/
python sentiment_analysis.py
```

### 3. View the dashboard
Open `dashboard.html` in any browser — all charts are inline, no server required.

---

## Dependencies

| Package | Purpose |
|---|---|
| `vaderSentiment` | Primary sentiment scoring engine |
| `textblob` | Polarity & subjectivity analysis |
| `nltk` | Tokenization, stopwords, VADER lexicon |
| `wordcloud` | Word cloud generation |
| `matplotlib` | Chart rendering |
| `seaborn` | Heatmaps and styled charts |
| `pandas` | Data wrangling |
| `numpy` | Numeric operations |
| `scipy` | Statistical operations |
