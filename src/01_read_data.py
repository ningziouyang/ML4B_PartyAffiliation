import json
import glob
import pandas as pd

# List to collect all tweets
all_data = []

# Iterate through all .jl files in the data folder
for filepath in glob.glob("../twitter-bundestag-2022/data/*.jl"):
    with open(filepath, "r", encoding="utf-8") as file:
        partei = None

        for line in file:
            entry = json.loads(line)

            # Extract party from account_data, usually in the first line
            if not partei and "account_data" in entry:
                partei = entry["account_data"].get("Partei", "Unbekannt")

            # Extract tweets from lines with "response"
            if "response" in entry:
                tweets = entry["response"].get("data", [])
                for tweet in tweets:
                    text = tweet.get("text", "")
                    if text:
                        all_data.append({
                            "text": text,
                            "partei": partei
                        })

# Create DataFrame
df = pd.DataFrame(all_data)

# Show basic info
print(f"Tweets insgesamt: {len(df)}")
print("Spalten:", df.columns.tolist())
if not df.empty:
    print(df.sample(5, random_state=1))

# Show party distribution
print("\nTweets pro Partei:")
print(df["partei"].value_counts())

# Save to CSV
df.to_csv("tweets_bundestag.csv", index=False, encoding="utf-8-sig")
print("CSV gespeichert unter: tweets_bundestag.csv")
