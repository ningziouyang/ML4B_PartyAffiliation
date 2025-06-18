import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Choose variant here (e.g., 'raw', 'no_urls', etc.)
VARIANT = "no_urls"

df = pd.read_csv(f"tweets_bundestag_{VARIANT}.csv", encoding="utf-8-sig")
min_tweet_count = 1000
df = df[df["partei"].map(df["partei"].value_counts()) >= min_tweet_count]
df = df.dropna(subset=["text"])

sample_size = min(50000, len(df))
df = df.sample(n=sample_size, random_state=42)

print("Class distribution in data:\n", df["partei"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["partei"],
    test_size=0.2,
    random_state=42,
    stratify=df["partei"]
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression with balanced class weights
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)

print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer for reproducibility/Streamlit
import joblib
joblib.dump(clf, f"lr_model_{VARIANT}.joblib")
joblib.dump(vectorizer, f"tfidf_{VARIANT}.joblib")
