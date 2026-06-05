import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

URL = "https://raw.githubusercontent.com/dpanagop/ML_and_AI_examples/master/data/amazon_reviews_sample.csv"

print("Loading Amazon Reviews dataset...")

try:
    df = pd.read_csv(URL)
    print(f"Loaded {len(df)} rows.")
    text_col = [c for c in df.columns if 'review' in c.lower() or 'text' in c.lower()][0]
    score_col = [c for c in df.columns if 'score' in c.lower() or 'rating' in c.lower()]
    score_col = score_col[0] if score_col else None
    print(f"   Text column  : '{text_col}'")
    print(f"   Score column : '{score_col}'")
except Exception as e:
    print(f"Could not load URL ({e}).")
    print("   Generating a small synthetic dataset for demonstration...")
    sample_reviews = [
        ("This product is absolutely amazing! I love it.", 5),
        ("Great quality, fast shipping. Very happy.", 5),
        ("Decent product, nothing special.", 3),
        ("Terrible. Broke after one use. Waste of money.", 1),
        ("Not what I expected. Disappointed.", 2),
        ("Fantastic! Will buy again for sure.", 5),
        ("Average product. Does the job.", 3),
        ("Horrible smell. Returned immediately.", 1),
        ("Pretty good for the price point.", 4),
        ("Exceeded my expectations. Highly recommend!", 5),
        ("Okay product but packaging was damaged.", 2),
        ("Absolutely love this. Best purchase ever!", 5),
        ("Meh. Could be better.", 3),
        ("Worst product I have ever bought.", 1),
        ("Very useful and well made.", 4),
    ] * 20
    df = pd.DataFrame(sample_reviews, columns=['review_text', 'score'])
    text_col  = 'review_text'
    score_col = 'score'

df = df.dropna(subset=[text_col]).reset_index(drop=True)
df = df.head(500)

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)

print("\nCleaning text...")
df['clean_text'] = df[text_col].apply(clean_text)
df['clean_no_stop'] = df['clean_text'].apply(remove_stopwords)

print("Running sentiment analysis...")

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    return pd.Series([polarity, subjectivity, label])

df[['polarity', 'subjectivity', 'sentiment']] = df['clean_text'].apply(get_sentiment)

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

print("\nAverage Polarity by Sentiment:")
print(df.groupby('sentiment')['polarity'].mean().round(3))

if score_col:
    print("\nAverage Star Rating by Sentiment:")
    print(df.groupby('sentiment')[score_col].mean().round(2))

pos_words = ' '.join(df[df['sentiment']=='Positive']['clean_no_stop']).split()
neg_words = ' '.join(df[df['sentiment']=='Negative']['clean_no_stop']).split()
top_pos = Counter(pos_words).most_common(15)
top_neg = Counter(neg_words).most_common(15)

sns.set_theme(style='whitegrid')
COLORS = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle('Amazon Reviews — Sentiment Analysis', fontsize=18,
             fontweight='bold', y=1.01)

sent_counts = df['sentiment'].value_counts()
bars = axes[0, 0].bar(sent_counts.index,
                      sent_counts.values,
                      color=[COLORS[s] for s in sent_counts.index],
                      edgecolor='white', linewidth=1.2)
axes[0, 0].set_title('Sentiment Distribution')
axes[0, 0].set_ylabel('Number of Reviews')
for bar in bars:
    axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    str(int(bar.get_height())),
                    ha='center', fontweight='bold')

axes[0, 1].hist(df[df['sentiment']=='Positive']['polarity'],
                bins=30, alpha=0.7, color='#2ecc71', label='Positive')
axes[0, 1].hist(df[df['sentiment']=='Negative']['polarity'],
                bins=30, alpha=0.7, color='#e74c3c', label='Negative')
axes[0, 1].hist(df[df['sentiment']=='Neutral']['polarity'],
                bins=30, alpha=0.7, color='#f39c12', label='Neutral')
axes[0, 1].set_title('Polarity Score Distribution')
axes[0, 1].set_xlabel('Polarity (-1 = Negative, +1 = Positive)')
axes[0, 1].legend()

for sentiment, grp in df.groupby('sentiment'):
    axes[0, 2].scatter(grp['polarity'], grp['subjectivity'],
                       c=COLORS[sentiment], label=sentiment, alpha=0.5, s=30)
axes[0, 2].set_title('Polarity vs Subjectivity')
axes[0, 2].set_xlabel('Polarity')
axes[0, 2].set_ylabel('Subjectivity')
axes[0, 2].axvline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0, 2].legend()

if score_col and df[score_col].notna().sum() > 10:
    rating_polarity = df.groupby(score_col)['polarity'].mean()
    axes[1, 0].bar(rating_polarity.index.astype(str),
                   rating_polarity.values,
                   color='#3498db', edgecolor='white')
    axes[1, 0].set_title('Avg Polarity by Star Rating')
    axes[1, 0].set_xlabel('Star Rating')
    axes[1, 0].set_ylabel('Mean Polarity')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=0.8)
else:
    axes[1, 0].text(0.5, 0.5, 'No star rating data', ha='center',
                    va='center', transform=axes[1, 0].transAxes, fontsize=12)
    axes[1, 0].set_title('Avg Polarity by Star Rating')

pos_labels, pos_vals = zip(*top_pos) if top_pos else ([''], [0])
axes[1, 1].barh(pos_labels[::-1], pos_vals[::-1], color='#2ecc71')
axes[1, 1].set_title('Top Words in Positive Reviews')
axes[1, 1].set_xlabel('Frequency')

neg_labels, neg_vals = zip(*top_neg) if top_neg else ([''], [0])
axes[1, 2].barh(neg_labels[::-1], neg_vals[::-1], color='#e74c3c')
axes[1, 2].set_title('Top Words in Negative Reviews')
axes[1, 2].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('sentiment_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as 'sentiment_analysis.png'")

print("\n" + "=" * 55)
print("LIVE SENTIMENT PREDICTIONS")
print("=" * 55)
test_reviews = [
    "This is the best product I have ever bought!",
    "Terrible quality. I want a refund.",
    "It is okay, nothing great about it.",
    "Fast delivery and the item was well packaged.",
    "Completely useless. Total waste of money.",
]
for review in test_reviews:
    blob = TextBlob(review)
    pol  = blob.sentiment.polarity
    label = 'Positive' if pol > 0.1 else ('Negative' if pol < -0.1 else 'Neutral')
    print(f"  [{label:8s}] ({pol:+.2f})  \"{review}\"")

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
for s, cnt in df['sentiment'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"  {s:10s}: {cnt:4d} reviews ({pct:.1f}%)")
print(f"\n  Mean Polarity   : {df['polarity'].mean():.3f}")
print(f"  Mean Subjectivity: {df['subjectivity'].mean():.3f}")
print("=" * 55)
