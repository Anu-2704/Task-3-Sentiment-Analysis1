"""
=============================================================
TASK 4 - Sentiment Analysis
Dataset : Amazon Product Reviews (amazon_reviews.csv)
Tools   : VADER, TextBlob, NLTK, Matplotlib, Seaborn, WordCloud
=============================================================
Pipeline:
  1. Text cleaning & preprocessing
  2. VADER sentiment scoring (compound, pos, neg, neu)
  3. TextBlob polarity & subjectivity
  4. Rule-based emotion detection (joy, anger, fear, surprise, disgust)
  5. Sentiment vs. star-rating agreement analysis
  6. Temporal sentiment trends
  7. Word clouds per sentiment class
  8. Statistical analysis & insights
  9. Export results to outputs/sentiment_results.csv
 10. Generate 12 charts saved to plots/

Run: python sentiment_analysis.py
=============================================================
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud
from scipy import stats

warnings.filterwarnings("ignore")

# ── NLTK data ─────────────────────────────────────────────────────────────────
for pkg in ["vader_lexicon", "stopwords", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "amazon_reviews.csv")
OUT_PLOTS  = os.path.join(SCRIPT_DIR, "plots")
OUT_DATA   = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_DATA,  exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
MAIN   = "#3b82d4"
GREEN  = "#2da44e"
RED    = "#e5534b"
YELLOW = "#f59e0b"
PURPLE = "#7c5cd8"
MUTED  = "#57606a"
BG     = "#ffffff"
SURF   = "#f7f8fa"
BORDER = "#e5e7eb"

SENT_COLORS = {"Positive": GREEN, "Neutral": YELLOW, "Negative": RED}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURF,
    "axes.edgecolor": BORDER, "axes.labelcolor": "#1f2328",
    "axes.labelsize": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": BORDER, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "legend.fontsize": 9, "font.family": "sans-serif",
})

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOTS, name), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {name}")

# =============================================================================
# 1. LOAD & CLEAN DATA
# =============================================================================
print("=" * 60)
print("1. LOADING & CLEANING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df["year"]       = df["reviewTime"].dt.year
df["month"]      = df["reviewTime"].dt.month
df["year_month"] = df["reviewTime"].dt.to_period("M")

# Drop the single missing reviewText row
df = df.dropna(subset=["reviewText"]).copy()
print(f"Records after dropping null reviewText: {len(df)}")

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # URLs
    text = re.sub(r"[^a-z\s']", " ", text)           # non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["reviewText"].apply(clean_text)

# Tokenize & remove stopwords
def tokenize(text):
    tokens = word_tokenize(text)
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]

df["tokens"] = df["clean_text"].apply(tokenize)
df["review_length"] = df["reviewText"].apply(len)
df["word_count"]    = df["tokens"].apply(len)

print(f"Avg tokens per review: {df['word_count'].mean():.1f}")

# =============================================================================
# 2. VADER SENTIMENT SCORING
# =============================================================================
print("\n" + "=" * 60)
print("2. VADER SENTIMENT SCORING")
print("=" * 60)

sia = SentimentIntensityAnalyzer()

vader_scores = df["reviewText"].apply(lambda t: sia.polarity_scores(str(t)))
df["vader_compound"] = vader_scores.apply(lambda x: x["compound"])
df["vader_pos"]      = vader_scores.apply(lambda x: x["pos"])
df["vader_neg"]      = vader_scores.apply(lambda x: x["neg"])
df["vader_neu"]      = vader_scores.apply(lambda x: x["neu"])

def vader_label(compound):
    if compound >= 0.05:  return "Positive"
    if compound <= -0.05: return "Negative"
    return "Neutral"

df["vader_label"] = df["vader_compound"].apply(vader_label)

vc = df["vader_label"].value_counts()
print("VADER label distribution:")
print(vc.to_string())
print(f"\nMean compound score: {df['vader_compound'].mean():.4f}")
print(f"Std  compound score: {df['vader_compound'].std():.4f}")

# =============================================================================
# 3. TEXTBLOB SENTIMENT
# =============================================================================
print("\n" + "=" * 60)
print("3. TEXTBLOB POLARITY & SUBJECTIVITY")
print("=" * 60)

tb_results = df["reviewText"].apply(lambda t: TextBlob(str(t)).sentiment)
df["tb_polarity"]     = tb_results.apply(lambda x: x.polarity)
df["tb_subjectivity"] = tb_results.apply(lambda x: x.subjectivity)

def tb_label(polarity):
    if polarity > 0.05:  return "Positive"
    if polarity < -0.05: return "Negative"
    return "Neutral"

df["tb_label"] = df["tb_polarity"].apply(tb_label)

print("TextBlob label distribution:")
print(df["tb_label"].value_counts().to_string())
print(f"\nMean polarity:     {df['tb_polarity'].mean():.4f}")
print(f"Mean subjectivity: {df['tb_subjectivity'].mean():.4f}")

# =============================================================================
# 4. EMOTION DETECTION (rule-based lexicon)
# =============================================================================
print("\n" + "=" * 60)
print("4. EMOTION DETECTION (Rule-based)")
print("=" * 60)

EMOTION_LEXICON = {
    "joy":      ["love","great","excellent","amazing","wonderful","fantastic","perfect","happy","best",
                 "awesome","beautiful","good","nice","brilliant","outstanding","delighted","pleased",
                 "superb","enjoy","glad","terrific","incredible","fabulous","magnificent","thrilled"],
    "anger":    ["terrible","awful","horrible","hate","worst","useless","garbage","broken","rubbish",
                 "defective","disgusting","angry","frustrated","outraged","furious","disappoint",
                 "pathetic","junk","ridiculous","waste","scam","lied","cheated","fault","damage"],
    "fear":     ["worried","afraid","scared","concern","risk","danger","problem","issue","warning",
                 "careful","caution","uncertain","doubt","unsafe","avoid","failure","nervous"],
    "surprise": ["unexpected","surprised","wow","incredible","unbelievable","suddenly","shock",
                 "amazing","astonishing","remarkable","extraordinary","impressive","astounded"],
    "disgust":  ["disgusting","gross","nasty","revolting","repulsive","offensive","vile","awful",
                 "horrible","unacceptable","cheap","fake","counterfeit","poor","inferior","shoddy"],
    "sadness":  ["sad","disappointed","sorry","unfortunate","regret","unhappy","miss","lost",
                 "broken","fail","worse","bad","poor","disappointment","letdown","upset"],
}

def detect_emotions(tokens):
    counts = {e: 0 for e in EMOTION_LEXICON}
    for tok in tokens:
        for emotion, lexicon in EMOTION_LEXICON.items():
            if tok in lexicon:
                counts[emotion] += 1
    return counts

emotion_data = df["tokens"].apply(detect_emotions)
for emotion in EMOTION_LEXICON:
    df[f"emo_{emotion}"] = emotion_data.apply(lambda x: x[emotion])

df["dominant_emotion"] = df[[f"emo_{e}" for e in EMOTION_LEXICON]].apply(
    lambda row: row.idxmax().replace("emo_", "") if row.max() > 0 else "none", axis=1)

print("Dominant emotion distribution:")
print(df["dominant_emotion"].value_counts().to_string())

# =============================================================================
# 5. AGREEMENT: VADER vs STAR RATING
# =============================================================================
print("\n" + "=" * 60)
print("5. VADER vs. STAR RATING AGREEMENT")
print("=" * 60)

def star_to_sentiment(r):
    if r >= 4: return "Positive"
    if r == 3: return "Neutral"
    return "Negative"

df["star_label"] = df["overall"].apply(star_to_sentiment)

agree = (df["vader_label"] == df["star_label"]).sum()
total = len(df)
print(f"Agreement rate (VADER vs Stars): {agree}/{total} = {agree/total*100:.1f}%")

# Confusion-like pivot
agreement_pivot = pd.crosstab(df["star_label"], df["vader_label"],
                              rownames=["Star Rating"], colnames=["VADER Label"])
print("\nCross-tabulation:")
print(agreement_pivot.to_string())

# =============================================================================
# 6. TEMPORAL SENTIMENT TREND
# =============================================================================
print("\n" + "=" * 60)
print("6. TEMPORAL SENTIMENT TRENDS")
print("=" * 60)

monthly_sent = (df.groupby(["year_month", "vader_label"])
                  .size().reset_index(name="count"))
monthly_sent["period"] = monthly_sent["year_month"].astype(str)

monthly_compound = df.groupby("year_month")["vader_compound"].mean().reset_index()
monthly_compound["period"] = monthly_compound["year_month"].astype(str)
print("Monthly avg compound score (first 5):")
print(monthly_compound.head().to_string(index=False))

# =============================================================================
# 7. EXPORT RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("7. EXPORTING RESULTS")
print("=" * 60)

export_cols = ["reviewerID", "overall", "star_label",
               "vader_compound", "vader_pos", "vader_neg", "vader_neu",
               "vader_label", "tb_polarity", "tb_subjectivity", "tb_label",
               "dominant_emotion", "review_length", "word_count",
               "year", "month"]
df[export_cols].to_csv(os.path.join(OUT_DATA, "sentiment_results.csv"), index=False)
print("Exported: outputs/sentiment_results.csv")

# Also export summary stats as JSON for the frontend
summary = {
    "total_reviews": int(len(df)),
    "vader_distribution": df["vader_label"].value_counts().to_dict(),
    "tb_distribution": df["tb_label"].value_counts().to_dict(),
    "mean_compound": round(float(df["vader_compound"].mean()), 4),
    "mean_polarity": round(float(df["tb_polarity"].mean()), 4),
    "mean_subjectivity": round(float(df["tb_subjectivity"].mean()), 4),
    "agreement_rate": round(float(agree / total * 100), 2),
    "dominant_emotion_dist": df["dominant_emotion"].value_counts().to_dict(),
    "avg_compound_by_rating": df.groupby("overall")["vader_compound"].mean().round(4).to_dict(),
    "avg_polarity_by_rating": df.groupby("overall")["tb_polarity"].mean().round(4).to_dict(),
}
with open(os.path.join(OUT_DATA, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Exported: outputs/summary.json")

# =============================================================================
# CHARTS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING CHARTS")
print("=" * 60)

# ── Chart 01: VADER label distribution ───────────────────────────────────────
labels   = ["Positive", "Neutral", "Negative"]
counts01 = [vc.get(l, 0) for l in labels]
colors01 = [GREEN, YELLOW, RED]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, counts01, color=colors01, edgecolor=BG,
              linewidth=1.2, width=0.55)
for bar, val in zip(bars, counts01):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_xlabel("Sentiment Class"); ax.set_ylabel("Number of Reviews")
ax.set_title("Chart 1 - VADER Sentiment Distribution")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("01_vader_distribution.png")

# ── Chart 02: TextBlob label distribution ────────────────────────────────────
tb_vc    = df["tb_label"].value_counts()
counts02 = [tb_vc.get(l, 0) for l in labels]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, counts02, color=colors01, edgecolor=BG,
              linewidth=1.2, width=0.55)
for bar, val in zip(bars, counts02):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_xlabel("Sentiment Class"); ax.set_ylabel("Number of Reviews")
ax.set_title("Chart 2 - TextBlob Sentiment Distribution")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("02_textblob_distribution.png")

# ── Chart 03: VADER compound score distribution ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(df["vader_compound"], bins=50, color=MAIN, edgecolor=BG,
        linewidth=0.6, alpha=0.85)
ax.axvline(0.05,  color=GREEN, linestyle="--", linewidth=1.5,
           label="Positive threshold (+0.05)")
ax.axvline(-0.05, color=RED,   linestyle="--", linewidth=1.5,
           label="Negative threshold (-0.05)")
ax.axvline(df["vader_compound"].mean(), color=PURPLE, linestyle="-",
           linewidth=2, label=f"Mean ({df['vader_compound'].mean():.3f})")
ax.set_xlabel("VADER Compound Score (-1 = most negative, +1 = most positive)")
ax.set_ylabel("Number of Reviews")
ax.set_title("Chart 3 - Distribution of VADER Compound Scores")
ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
save("03_vader_compound_dist.png")

# ── Chart 04: VADER compound score by star rating ────────────────────────────
avg_compound = df.groupby("overall")["vader_compound"].mean()
std_compound = df.groupby("overall")["vader_compound"].std()
star_ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
rating_colors = [RED, "#f59e0b", MUTED, MAIN, GREEN]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar([str(int(r)) for r in star_ratings],
              [avg_compound[r] for r in star_ratings],
              color=rating_colors, edgecolor=BG, linewidth=1, width=0.6,
              yerr=[std_compound[r] for r in star_ratings],
              capsize=5, error_kw={"elinewidth": 1.2, "ecolor": "#1f2328"})
ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
for bar, r in zip(bars, star_ratings):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_compound[r] + 0.01,
            f"{avg_compound[r]:.3f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")
ax.set_xlabel("Star Rating"); ax.set_ylabel("Mean VADER Compound Score")
ax.set_title("Chart 4 - Mean VADER Score by Star Rating")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("04_compound_by_rating.png")

# ── Chart 05: TextBlob polarity vs subjectivity scatter ─────────────────────
sample = df.sample(min(800, len(df)), random_state=42)
sc_colors = sample["vader_label"].map(SENT_COLORS)

fig, ax = plt.subplots(figsize=(9, 5.5))
sc = ax.scatter(sample["tb_polarity"], sample["tb_subjectivity"],
                c=sample["vader_compound"], cmap="RdYlGn",
                alpha=0.45, s=16, edgecolors="none",
                vmin=-1, vmax=1)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("VADER Compound", fontsize=9)
ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
ax.set_xlabel("TextBlob Polarity")
ax.set_ylabel("TextBlob Subjectivity")
ax.set_title("Chart 5 - TextBlob Polarity vs Subjectivity (colored by VADER)")
ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
save("05_polarity_vs_subjectivity.png")

# ── Chart 06: Emotion distribution (bar) ─────────────────────────────────────
emo_counts = df["dominant_emotion"].value_counts()
emo_colors_map = {
    "joy": GREEN, "anger": RED, "sadness": MAIN,
    "disgust": PURPLE, "fear": YELLOW, "surprise": "#f59e0b", "none": MUTED
}
emo_c = [emo_colors_map.get(e, MUTED) for e in emo_counts.index]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(emo_counts.index, emo_counts.values,
              color=emo_c, edgecolor=BG, linewidth=1, width=0.6)
for bar, val in zip(bars, emo_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10, str(val),
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Dominant Emotion"); ax.set_ylabel("Number of Reviews")
ax.set_title("Chart 6 - Dominant Emotion Distribution")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("06_emotion_distribution.png")

# ── Chart 07: Emotion by star rating (stacked bar) ───────────────────────────
emotions = ["joy", "anger", "sadness", "disgust", "fear", "surprise"]
emo_by_rating = (df.groupby("overall")[[f"emo_{e}" for e in emotions]]
                   .mean())
emo_by_rating.columns = emotions

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(5)
emo_colors_list = [GREEN, RED, MAIN, PURPLE, YELLOW, "#f59e0b"]
for emotion, color in zip(emotions, emo_colors_list):
    ax.bar([str(int(r)) for r in emo_by_rating.index],
           emo_by_rating[emotion], bottom=bottom,
           label=emotion.capitalize(), color=color,
           edgecolor=BG, linewidth=0.6)
    bottom += emo_by_rating[emotion].values
ax.set_xlabel("Star Rating"); ax.set_ylabel("Avg Emotion Word Count")
ax.set_title("Chart 7 - Emotion Composition by Star Rating (Stacked)")
ax.legend(title="Emotion", ncol=6, loc="upper left")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("07_emotion_by_rating.png")

# ── Chart 08: Temporal sentiment trend (monthly avg compound) ────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
x_range = range(len(monthly_compound))
ax.fill_between(x_range, monthly_compound["vader_compound"],
                alpha=0.15, color=GREEN,
                where=monthly_compound["vader_compound"] >= 0)
ax.fill_between(x_range, monthly_compound["vader_compound"],
                alpha=0.15, color=RED,
                where=monthly_compound["vader_compound"] < 0)
ax.plot(x_range, monthly_compound["vader_compound"],
        color=MAIN, linewidth=2, marker="o", markersize=4,
        markerfacecolor=BG, markeredgecolor=MAIN)
ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
step = max(1, len(monthly_compound) // 12)
ax.set_xticks(list(range(0, len(monthly_compound), step)))
ax.set_xticklabels(monthly_compound["period"].iloc[::step],
                   rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Month"); ax.set_ylabel("Avg VADER Compound Score")
ax.set_title("Chart 8 - Monthly Avg Sentiment Score Trend")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("08_temporal_sentiment.png")

# ── Chart 09: Sentiment agreement heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(agreement_pivot, annot=True, fmt="d",
            cmap="YlOrRd", linewidths=0.5,
            linecolor=BG, ax=ax,
            annot_kws={"size": 10})
ax.set_title("Chart 9 - VADER vs Star Rating Agreement Heatmap")
ax.set_xlabel("VADER Label"); ax.set_ylabel("Star-based Label")
save("09_agreement_heatmap.png")

# ── Chart 10: Subjectivity by sentiment ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, color in SENT_COLORS.items():
    subset = df[df["vader_label"] == label]["tb_subjectivity"]
    ax.hist(subset, bins=30, alpha=0.55, color=color,
            label=f"{label} (n={len(subset):,})", edgecolor=BG, linewidth=0.4)
ax.set_xlabel("TextBlob Subjectivity (0=objective, 1=subjective)")
ax.set_ylabel("Count")
ax.set_title("Chart 10 - Subjectivity Distribution by Sentiment Class")
ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
save("10_subjectivity_by_sentiment.png")

# ── Chart 11: Word clouds (Positive / Negative) ───────────────────────────────
def make_wc(tokens_series, color, max_words=120):
    all_words = " ".join([" ".join(t) for t in tokens_series])
    wc = WordCloud(width=700, height=350,
                   background_color="white",
                   colormap=color,
                   max_words=max_words,
                   collocations=False,
                   prefer_horizontal=0.85).generate(all_words)
    return wc

pos_tokens = df[df["vader_label"] == "Positive"]["tokens"]
neg_tokens = df[df["vader_label"] == "Negative"]["tokens"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
wc_pos = make_wc(pos_tokens, "Greens")
wc_neg = make_wc(neg_tokens, "Reds")

axes[0].imshow(wc_pos, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("Positive Reviews — Top Words", fontsize=12, fontweight="bold")
axes[0].patch.set_facecolor(BG)

axes[1].imshow(wc_neg, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Negative Reviews — Top Words", fontsize=12, fontweight="bold")
axes[1].patch.set_facecolor(BG)

fig.suptitle("Chart 11 - Word Clouds: Positive vs Negative Reviews",
             fontsize=13, fontweight="bold")
save("11_word_clouds.png")

# ── Chart 12: 4-panel story dashboard ────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Chart 12 - Sentiment Analysis Story Dashboard",
             fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# Panel A: donut — VADER distribution
ax_a = fig.add_subplot(gs[0, 0])
vc_vals = [vc.get(l, 0) for l in labels]
wedges_a, _, auto_a = ax_a.pie(
    vc_vals, colors=[GREEN, YELLOW, RED],
    startangle=90, autopct="%1.1f%%",
    pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=1.5))
for at in auto_a: at.set_fontsize(8)
centre = plt.Circle((0, 0), 0.45, color=BG)
ax_a.add_patch(centre)
ax_a.text(0, 0, "VADER\nSentiment", ha="center", va="center",
          fontsize=8, fontweight="bold")
ax_a.set_title("A) Sentiment Share")
ax_a.legend([f"{l}" for l in labels], loc="lower center",
            fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.08))

# Panel B: avg compound by rating
ax_b = fig.add_subplot(gs[0, 1])
ax_b.bar([str(int(r)) for r in star_ratings],
         [avg_compound[r] for r in star_ratings],
         color=rating_colors, edgecolor=BG, linewidth=1, width=0.6)
ax_b.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
ax_b.set_xlabel("Star Rating"); ax_b.set_ylabel("Mean Compound")
ax_b.set_title("B) Avg Sentiment by Rating")
ax_b.yaxis.grid(True); ax_b.set_axisbelow(True)

# Panel C: temporal trend
ax_c = fig.add_subplot(gs[1, 0])
ax_c.fill_between(range(len(monthly_compound)),
                  monthly_compound["vader_compound"],
                  alpha=0.15, color=GREEN,
                  where=monthly_compound["vader_compound"] >= 0)
ax_c.plot(range(len(monthly_compound)),
          monthly_compound["vader_compound"],
          color=MAIN, linewidth=1.8)
step4 = max(1, len(monthly_compound) // 8)
ax_c.set_xticks(range(0, len(monthly_compound), step4))
ax_c.set_xticklabels(monthly_compound["period"].iloc[::step4],
                     rotation=40, ha="right", fontsize=7)
ax_c.set_ylabel("Avg Compound"); ax_c.set_title("C) Monthly Sentiment Trend")
ax_c.yaxis.grid(True); ax_c.set_axisbelow(True)

# Panel D: emotion distribution
ax_d = fig.add_subplot(gs[1, 1])
emo_plot = emo_counts[emo_counts.index != "none"].head(6)
ax_d.barh(emo_plot.index, emo_plot.values,
          color=[emo_colors_map.get(e, MUTED) for e in emo_plot.index],
          edgecolor=BG, linewidth=1)
ax_d.set_xlabel("Reviews"); ax_d.set_title("D) Dominant Emotions")
ax_d.xaxis.grid(True); ax_d.set_axisbelow(True)

save("12_story_dashboard.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total reviews analyzed : {len(df):,}")
print(f"VADER - Positive       : {vc.get('Positive',0):,} ({vc.get('Positive',0)/len(df)*100:.1f}%)")
print(f"VADER - Neutral        : {vc.get('Neutral',0):,}  ({vc.get('Neutral',0)/len(df)*100:.1f}%)")
print(f"VADER - Negative       : {vc.get('Negative',0):,}  ({vc.get('Negative',0)/len(df)*100:.1f}%)")
print(f"Mean compound score    : {df['vader_compound'].mean():.4f}")
print(f"Mean TB polarity       : {df['tb_polarity'].mean():.4f}")
print(f"Mean TB subjectivity   : {df['tb_subjectivity'].mean():.4f}")
print(f"VADER/Star agreement   : {agree/total*100:.1f}%")
print(f"\nPlots saved to  : {OUT_PLOTS}")
print(f"Results saved to: {OUT_DATA}")
print("Sentiment analysis complete.")
