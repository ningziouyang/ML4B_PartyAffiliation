import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib

def count_emojis(text):
    return sum(char in emoji.EMOJI_DATA for char in str(text))

def count_hashtags(text):
    return len(re.findall(r"#\w+", str(text)))

def count_mentions(text):
    return len(re.findall(r"@\w+", str(text)))

def count_urls(text):
    return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))

# Choose best variant based on previous results
VARIANT = "no_urls"
df = pd.read_csv(f"tweets_bundestag_{VARIANT}.csv", encoding="utf-8-sig")
min_tweet_count = 1000
df = df[df["partei"].map(df["partei"].value_counts()) >= min_tweet_count]
df = df.dropna(subset=["text"])

# Add extra features
df["num_emojis"] = df["text"].apply(count_emojis)
df["num_hashtags"] = df["text"].apply(count_hashtags)
df["num_mentions"] = df["text"].apply(count_mentions)
df["num_urls"] = df["text"].apply(count_urls)

sample_size = min(50000, len(df))
df = df.sample(n=sample_size, random_state=42)

X_text = df["text"]
X_extra = df[["num_emojis", "num_hashtags", "num_mentions", "num_urls"]].values
y = df["partei"]

X_train_text, X_test_text, X_train_extra, X_test_extra, y_train, y_test = train_test_split(
    X_text, X_extra, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

scaler = StandardScaler(with_mean=False)
X_train_extra_scaled = scaler.fit_transform(X_train_extra)
X_test_extra_scaled = scaler.transform(X_test_extra)

X_train_combined = hstack([X_train_vec, X_train_extra_scaled])
X_test_combined = hstack([X_test_vec, X_test_extra_scaled])

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train_combined, y_train)

y_pred = clf.predict(X_test_combined)
print("Classification report (with extra features):")
print(classification_report(y_test, y_pred, zero_division=0, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, f"lr_model_extra_{VARIANT}.joblib")
joblib.dump(vectorizer, f"tfidf_extra_{VARIANT}.joblib")
joblib.dump(scaler, f"scaler_extra_{VARIANT}.joblib")

# Feature importances for numeric features
coef = clf.coef_
print("Numeric feature importances (per class):")
for i, col in enumerate(["num_emojis", "num_hashtags", "num_mentions", "num_urls"]):
    print(f"{col}: {coef[:, -(i+1)].mean():.4f}")
