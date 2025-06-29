import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel

# Load artifacts
MODEL_PATH = "models/lr_tfidf_bert_engineered.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_bert_engineered.joblib"
SCALER_PATH = "models/feature_scaler_bert_engineered.joblib"
BERT_PATH = "bert-base-german-cased"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)

tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model = AutoModel.from_pretrained(BERT_PATH)
bert_model.eval()

# Feature engineering as in training
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
    try:
        import emoji
        return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)
    except ImportError:
        # fallback: just count colons
        return str(text).count(":")

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

def extract_features(text):
    feats = [
        len(str(text)),                              # tweet_length_chars
        len(str(text).split()),                      # tweet_length_words
        avg_word_length(text),                       # avg_word_length
        uppercase_ratio(text),                       # uppercase_ratio
        str(text).count("!"),                        # exclamations
        str(text).count("?"),                        # questions
        multi_punct_count(text),                     # multi_punct_count
        count_political_terms(text),                 # political_term_count
        count_emojis(text),                          # num_emojis
        count_hashtags(text),                        # num_hashtags
        count_mentions(text),                        # num_mentions
        count_urls(text),                            # num_urls
        count_dots(text),                            # dots
        is_retweet(text),                            # is_retweet
    ]
    return np.array(feats).reshape(1, -1)

# BERT CLS embedding
def embed_single_text(text, tokenizer, model, max_len=64):
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        output = model(**encoded)
        # CLS token
        cls_emb = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_emb.reshape(1, -1)

# Streamlit UI 
st.title("Parteivorhersage für Bundestags-Tweets (ML4B-Projekt)")
tweet = st.text_area("Gib einen Bundestags-Tweet ein:")

if tweet and st.button("Vorhersagen"):
    # 1. TF-IDF
    X_tfidf = vectorizer.transform([tweet])  # (1, 2000)
    # 2. BERT
    X_bert = embed_single_text(tweet, tokenizer, bert_model)  # (1, 768)
    # 3. Engineered features
    X_eng = extract_features(tweet)  # (1, 14)
    X_eng_scaled = scaler.transform(X_eng)
    # 4. Combine
    X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
    # 5. Predict
    pred = model.predict(X_all)[0]
    st.success(f"**Vorhergesagte Partei:** {pred}")

    # show class probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_all)[0]
        parties = model.classes_
        st.subheader("Wahrscheinlichkeit je Partei")
        st.bar_chart({p: float(prob) for p, prob in zip(parties, probs)})

st.write("---")
st.markdown("**Hinweis:** Die Vorhersage basiert auf einer Kombination von TF-IDF, BERT-Embeddings und engineered Features.")
