import pandas as pd
import re
import emoji

# Variant 1 functions
def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", str(text))

def remove_mentions(text):
    return re.sub(r"@\w+", "", str(text))

def clean_hashtags(text, keep_hash=True):
    if keep_hash:
        return str(text)
    return re.sub(r"#(\w+)", r"\1", str(text))

def emoji_to_text(text, demojize=True):
    if demojize:
        return emoji.demojize(str(text), delimiters=(" ", " "))
    return str(text)

def preprocess_variant(
    text,
    lower=True,
    remove_url=False,
    remove_mention=False,
    remove_hash_symbol=False,
    demojize=False
):
    t = str(text)
    if lower:
        t = t.lower()
    if remove_url:
        t = remove_urls(t)
    if remove_mention:
        t = remove_mentions(t)
    t = clean_hashtags(t, keep_hash=not remove_hash_symbol)
    t = emoji_to_text(t, demojize=demojize)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Variant 2 (special token) functions
def replace_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", " URL ", str(text))

def replace_mentions(text):
    return re.sub(r"@\w+", " USER ", str(text))

def special_hashtags(text):
    # Replace #hashtag with HASHTAG_hashtag
    return re.sub(r"#(\w+)", r"HASHTAG_\1", str(text))

def preprocess_specialtok(text):
    t = str(text).lower()
    t = replace_urls(t)
    t = replace_mentions(t)
    t = special_hashtags(t)
    t = emoji_to_text(t, demojize=True)
    t = re.sub(r"\s+", " ", t).strip()
    return t

if __name__ == "__main__":
    df = pd.read_csv("tweets_bundestag.csv", encoding="utf-8-sig")
    min_tweet_count = 1000
    df = df[df["partei"].map(df["partei"].value_counts()) >= min_tweet_count]
    df = df[df["partei"] != "Unbekannt"]
    df = df.sample(n=50000, random_state=42).reset_index(drop=True)

    # Save different preprocessing variants for ablation studies (Variant 1)
    variants = {
        "raw": df["text"],
        "lowercase": df["text"].apply(lambda x: preprocess_variant(x, lower=True, remove_url=False, remove_mention=False, demojize=False, remove_hash_symbol=False)),
        "no_urls": df["text"].apply(lambda x: preprocess_variant(x, lower=True, remove_url=True, remove_mention=False, demojize=False, remove_hash_symbol=False)),
        "no_mentions": df["text"].apply(lambda x: preprocess_variant(x, lower=True, remove_url=False, remove_mention=True, demojize=False, remove_hash_symbol=False)),
        "no_urls_mentions": df["text"].apply(lambda x: preprocess_variant(x, lower=True, remove_url=True, remove_mention=True, demojize=False, remove_hash_symbol=False)),
        "demojize": df["text"].apply(lambda x: preprocess_variant(x, lower=True, remove_url=False, remove_mention=False, demojize=True, remove_hash_symbol=False)),
        "no_hashsymbol": df["text"].apply(lambda x: preprocess_variant(x, lower=True, remove_url=False, remove_mention=False, demojize=False, remove_hash_symbol=True)),
        "no_urls_mentions_demojize": df["text"].apply(lambda x: preprocess_variant(x, lower=True, remove_url=True, remove_mention=True, demojize=True, remove_hash_symbol=False)),
    }

    for variant, series in variants.items():
        out = df[["partei"]].copy()
        out["text"] = series
        out.to_csv(f"tweets_bundestag_{variant}.csv", index=False, encoding="utf-8-sig")
        print(f"Saved variant: {variant} to tweets_bundestag_{variant}.csv")

    print("Example cleaned texts (Variant 1):")
    print(df.head(10)[["text"]])

    # Special token variant (Variant 2)
    df["specialtok_text"] = df["text"].apply(preprocess_specialtok)
    df[["specialtok_text", "partei"]].to_csv("tweets_bundestag_specialtok.csv", index=False, encoding="utf-8-sig")
    print("Example special token cleaned texts:")
    print(df["specialtok_text"].head(10))
