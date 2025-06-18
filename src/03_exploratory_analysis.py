import pandas as pd
import emoji
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("tweets_bundestag.csv", encoding="utf-8-sig")

# Filter out small/unknown parties for clearer analysis
min_tweet_count = 1000
df = df[df["partei"].map(df["partei"].value_counts()) >= min_tweet_count]
df = df[df["partei"] != "Unbekannt"]

# Feature extraction functions
def count_emojis(text):
    return sum(char in emoji.EMOJI_DATA for char in str(text))

def count_hashtags(text):
    return len(re.findall(r"#\w+", str(text)))

def count_mentions(text):
    return len(re.findall(r"@\w+", str(text)))

def count_urls(text):
    return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))

# Apply features
df["num_emojis"] = df["text"].apply(count_emojis)
df["num_hashtags"] = df["text"].apply(count_hashtags)
df["num_mentions"] = df["text"].apply(count_mentions)
df["num_urls"] = df["text"].apply(count_urls)

# Show descriptive statistics
summary = df.groupby("partei")[["num_emojis", "num_hashtags", "num_mentions", "num_urls"]].mean().sort_values(by="num_emojis", ascending=False)
print("\nDurchschnittliche Feature-Anzahl pro Partei:\n", summary)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(summary, annot=True, fmt=".2f", cmap="Blues")
plt.title("Durchschnittliche Anzahl von Emojis, Hashtags, Mentions und URLs pro Tweet (nach Partei)")
plt.tight_layout()
plt.show()

# Example for hashtags:
def get_top_n(pattern, n=10):
    all_matches = df["text"].str.findall(pattern).explode()
    return all_matches.value_counts().head(n)

print("\nTop 10 Hashtags insgesamt:\n", get_top_n(r"#\w+"))
print("\nTop 10 Mentions insgesamt:\n", get_top_n(r"@\w+"))
