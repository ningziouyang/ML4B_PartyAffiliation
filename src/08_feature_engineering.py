import pandas as pd
import re
import emoji

# List of important political terms (example, expand as needed)
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def count_political_terms(text):
    text = str(text).lower()
    return sum(1 for word in POLITICAL_TERMS if word in text)

def uppercase_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 0
    return sum(1 for c in text if c.isupper()) / len(text)

def avg_word_length(text):
    words = re.findall(r"\w+", str(text))
    if not words:
        return 0
    return sum(len(w) for w in words) / len(words)

def multi_punct_count(text):
    return len(re.findall(r"[!?]{2,}", str(text)))

def count_emojis(text):
    return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)

def count_hashtags(text):
    return len(re.findall(r"#\w+", str(text)))

def count_mentions(text):
    return len(re.findall(r"@\w+", str(text)))

def count_urls(text):
    return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))

def count_dots(text):
    return len(re.findall(r"\.\.+", str(text)))

def is_retweet(text):
    return int(str(text).strip().lower().startswith("rt @"))

if __name__ == "__main__":
    # Use the best preprocessing variant as base
    VARIANT = "no_urls"
    df = pd.read_csv(f"tweets_bundestag_{VARIANT}.csv", encoding="utf-8-sig")
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df = df.reset_index(drop=True)

    # Feature extraction
    df["tweet_length_chars"] = df["text"].apply(len)
    df["tweet_length_words"] = df["text"].apply(lambda x: len(str(x).split()))
    df["avg_word_length"] = df["text"].apply(avg_word_length)
    df["uppercase_ratio"] = df["text"].apply(uppercase_ratio)
    df["exclamations"] = df["text"].apply(lambda x: str(x).count("!"))
    df["questions"] = df["text"].apply(lambda x: str(x).count("?"))
    df["multi_punct_count"] = df["text"].apply(multi_punct_count)
    df["political_term_count"] = df["text"].apply(count_political_terms)
    df["num_emojis"] = df["text"].apply(count_emojis)
    df["num_hashtags"] = df["text"].apply(count_hashtags)
    df["num_mentions"] = df["text"].apply(count_mentions)
    df["num_urls"] = df["text"].apply(count_urls)
    df["dots"] = df["text"].apply(count_dots)
    df["is_retweet"] = df["text"].apply(is_retweet)

    # Save engineered features for downstream use
    df.to_csv("tweets_bundestag_features.csv", index=False, encoding="utf-8-sig")
    print("Feature file saved as tweets_bundestag_features.csv")
