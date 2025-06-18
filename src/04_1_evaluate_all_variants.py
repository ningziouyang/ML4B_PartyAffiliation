import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

variants = [
    "raw",
    "lowercase",
    "no_urls",
    "no_mentions",
    "no_urls_mentions",
    "demojize",
    "no_hashsymbol",
    "no_urls_mentions_demojize",
    "specialtok"
]

results = []

for variant in variants:
    print(f"\n=== Testing variant: {variant} ===")
    df = pd.read_csv(f"tweets_bundestag_{variant}.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["text" if variant != "specialtok" else "specialtok_text"])
    text_col = "text" if variant != "specialtok" else "specialtok_text"
    min_tweet_count = 1000
    df = df[df["partei"].map(df["partei"].value_counts()) >= min_tweet_count]
    sample_size = min(50000, len(df))
    df = df.sample(n=sample_size, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df["partei"],
        test_size=0.2,
        random_state=42,
        stratify=df["partei"]
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}, Weighted F1: {weighted_f1:.3f}")
    results.append({
        "variant": variant,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

print("\n=== Summary Table ===")
print(pd.DataFrame(results))
