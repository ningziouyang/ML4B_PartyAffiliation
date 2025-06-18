import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load CSV
df = pd.read_csv("tweets_bundestag.csv", encoding="utf-8-sig")

# Filter out unknown or too-small parties
min_tweet_count = 1000
valid_parties = df["partei"].value_counts()[df["partei"].value_counts() >= min_tweet_count].index
df = df[df["partei"].isin(valid_parties) & (df["partei"] != "Unbekannt")]

# Optional: Downsample if dataset is huge
max_samples = 50000
if len(df) > max_samples:
    df = df.sample(n=max_samples, random_state=42)

print(f"Verbleibende Parteien: {df['partei'].unique()}")
print("Tweets pro Partei im Sample:\n", df["partei"].value_counts())

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["partei"],
    test_size=0.2,
    random_state=42,
    stratify=df["partei"]
)

# TF-IDF vectorization (keep hashtags, mentions, emojis, URLs for now)
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression baseline
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train_vec, y_train)

# Evaluation
y_pred = clf.predict(X_test_vec)
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
