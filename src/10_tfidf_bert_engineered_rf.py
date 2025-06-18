import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel

# Load features
df = pd.read_csv("tweets_bundestag_features.csv", encoding="utf-8-sig")
min_tweet_count = 1000
df = df[df["partei"].map(df["partei"].value_counts()) >= min_tweet_count]
df = df.dropna(subset=["text"])

# For a quick, balanced test, sample up to 1000 tweets per party
sample_per_party = 1000
df_sample = (
    df.groupby('partei', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), sample_per_party), random_state=42))
    .reset_index(drop=True)
)

print(f"Number of tweets used: {len(df_sample)}")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = vectorizer.fit_transform(df_sample["text"])

# BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
model = AutoModel.from_pretrained("bert-base-german-cased")
model.eval()
def embed_texts(texts, max_len=64):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            output = model(**encoded)
            cls_emb = output.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_emb)
    return np.array(embeddings)

print("Calculating BERT embeddings (this may take several minutes)...")
X_bert = embed_texts(df_sample["text"].tolist())

# Engineered features
feature_cols = [
    "tweet_length_chars", "tweet_length_words", "avg_word_length", "uppercase_ratio",
    "exclamations", "questions", "multi_punct_count", "political_term_count",
    "num_emojis", "num_hashtags", "num_mentions", "num_urls", "dots", "is_retweet"
]
scaler = StandardScaler()
X_eng = scaler.fit_transform(df_sample[feature_cols])

# Combine all features
X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng])
y = df_sample["partei"]

X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
