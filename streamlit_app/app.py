import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel

# Konfiguration
MODEL_PATH = "models/lr_tfidf_bert_engineered.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_bert_engineered.joblib"
SCALER_PATH = "models/feature_scaler_bert_engineered.joblib"
BERT_PATH = "bert-base-german-cased"

# Geräteerkennung
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelle cachen
@st.cache_resource
def load_artifacts():
    """Lädt alle benötigten Modelle und Tools"""
    try:
        # Sklearn-Modelle
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # BERT-Komponenten (korrigierte Ladeprozedur)
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        bert_model = AutoModel.from_pretrained(BERT_PATH)
        
        # Explizite Gerätezuweisung
        device = get_device()
        if str(device) == "cuda":
            bert_model = bert_model.to(device)
        
        bert_model.eval()
        return model, vectorizer, scaler, tokenizer, bert_model
        
    except Exception as e:
        st.error(f"Initialisierungsfehler: {str(e)}")
        raise

# Initialisierung
try:
    model, vectorizer, scaler, tokenizer, bert_model = load_artifacts()
except Exception as e:
    st.error(f"Kritischer Fehler beim Start: {str(e)}")
    st.stop()

# Feature Engineering
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung"
]

def count_political_terms(text):
    text = str(text).lower()
    return sum(1 for word in POLITICAL_TERMS if word in text)

def uppercase_ratio(text):
    text = str(text)
    return sum(1 for c in text if c.isupper()) / len(text) if text else 0

def extract_features(text):
    """Extrahiert 13 manuelle Features"""
    words = re.findall(r"\w+", str(text))
    feats = [
        len(str(text)),
        len(words),
        sum(len(w) for w in words)/len(words) if words else 0,
        uppercase_ratio(text),
        str(text).count("!"),
        str(text).count("?"),
        len(re.findall(r"[!?]{2,}", str(text))),
        count_political_terms(text),
        len(re.findall(r"#\w+", str(text))),
        len(re.findall(r"@\w+", str(text))),
        len(re.findall(r"http\S+|www\S+|https\S+", str(text))),
        len(re.findall(r"\.\.+", str(text))),
        int(str(text).strip().lower().startswith("rt @"))
    ]
    return np.array(feats).reshape(1, -1)

# BERT Embedding
def embed_single_text(text, tokenizer, model, max_len=64):
    device = next(model.parameters()).device
    with torch.no_grad():
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        ).to(device)
        output = model(**encoded)
        return output.last_hidden_state[:, 0, :].cpu().numpy().reshape(1, -1)

# Streamlit UI
st.title("🇩🇪 Bundestags-Tweet Analyse")
st.markdown("Parteivorhersage mit ML-Modell")

tweet = st.text_area(
    "Tweet eingeben (Deutsch):",
    placeholder="Beispiel: 'Wir fordern mehr Klimaschutz...'",
    max_chars=280
)

if st.button("Analyse starten"):
    if not tweet.strip():
        st.warning("Bitte Text eingeben!")
    else:
        with st.spinner("Analyse läuft..."):
            try:
                # Feature-Extraktion
                X_tfidf = vectorizer.transform([tweet])
                X_bert = embed_single_text(tweet, tokenizer, bert_model)
                X_eng = scaler.transform(extract_features(tweet))
                
                # Kombination und Vorhersage
                X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng])
                pred = model.predict(X_all)[0]
                
                # Ergebnisanzeige
                st.success(f"Vorhersage: {pred}")
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_all)[0]
                    st.subheader("Wahrscheinlichkeiten")
                    st.bar_chart(dict(zip(model.classes_, probs)))
                    
            except Exception as e:
                st.error(f"Analysefehler: {str(e)}")

# Footer
st.markdown("---")
st.caption("Modell trainiert mit Bundestags-Tweets | Max. 280 Zeichen")
