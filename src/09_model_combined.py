import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import warnings

warnings.filterwarnings("ignore")

# 1. Load features extracted in 07
df = pd.read_csv("tweets_bundestag_features.csv", encoding="utf-8-sig")
min_tweet_count = 1000
df = df[df["partei"].map(df["partei"].value_counts()) >= min_tweet_count]
df = df.dropna(subset=["text"])
df = df.reset_index(drop=True)
sample_size = min(50000, len(df))
df = df.sample(n=sample_size, random_state=42)

# 2. BERT tokenizer and model (German BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
model = AutoModel.from_pretrained("bert-base-german-cased")
model.eval()  # Eval mode

def embed_texts(texts, max_len=128, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            output = model(**encoded)
            # CLS token embedding (batch_size, hidden_size)
            cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

X_text = df["text"].tolist()
y = df["partei"]

# 3. Numeric features to use
feature_cols = [
    "tweet_length_chars", "tweet_length_words", "avg_word_length", "uppercase_ratio",
    "exclamations", "questions", "multi_punct_count", "political_term_count",
    "num_emojis", "num_hashtags", "num_mentions", "num_urls", "dots", "is_retweet"
]
X_numeric = df[feature_cols].values

# (Optional) Balanced sampling: Max 1000 per party
sample_per_party = 1000
df_balanced = (
    df.groupby("partei", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), sample_per_party), random_state=42))
      .reset_index(drop=True)
)
X_text_bal = df_balanced["text"].tolist()
X_numeric_bal = df_balanced[feature_cols].values
y_bal = df_balanced["parted"]

# Train/test split on balanced data
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text_bal, X_numeric_bal, y_bal,
    test_size=0.2,
    random_state=42,
    stratify=y_bal
)

# 4. Compute BERT embeddings (can be cached if slow)
X_text_train_emb = embed_texts(X_text_train)
X_text_test_emb = embed_texts(X_text_test)

# 5. Scale numeric features
scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

# 6. Combine BERT and numeric features
X_train_combined = np.hstack([X_text_train_emb, X_num_train_scaled])
X_test_combined = np.hstack([X_text_test_emb, X_num_test_scaled])

# 7. Train Logistic Regression
clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
clf.fit(X_train_combined, y_train)

y_pred = clf.predict(X_test_combined)
print("\nClassification report (combined features):")
print(classification_report(y_test, y_pred, zero_division=0, digits=3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler for reproducibility/Streamlit
joblib.dump(clf, "lr_model_combined.joblib")
joblib.dump(scaler, "scaler_combined.joblib")
