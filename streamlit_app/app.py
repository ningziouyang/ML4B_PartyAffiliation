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

# Gerätehandling korrigiert
def get_device():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    return torch.device("cpu")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Lädt Modelle mit korrigierter Gerätezuweisung"""
    try:
        # Sklearn-Modelle
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # BERT-Komponenten
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        
        # Kritische Änderung: Vermeidung von Meta-Tensoren
        bert_model = AutoModel.from_pretrained(
            BERT_PATH,
            torch_dtype=torch.float32  # Expliziter Datentyp
        ).to(get_device())
        
        bert_model.eval()
        return model, vectorizer, scaler, tokenizer, bert_model
        
    except Exception as e:
        st.error(f"Modellinitialisierungsfehler: {str(e)}")
        raise

# Initialisierung mit Fehlerhandling
try:
    device = get_device()
    model, vectorizer, scaler, tokenizer, bert_model = load_artifacts()
    if str(device) == "cuda":
        torch.cuda.empty_cache()
except Exception as e:
    st.error(f"Kritischer Systemfehler: {str(e)}")
    st.stop()

# Feature Engineering
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung"
]

def count_political_terms(text):
    text = str(text).lower()
    return sum(1 for word in POLITICAL_TERMS if word in text)

def extract_features(text):
    """Extrahiert 13 numerische Features ohne Emojis"""
    text_str = str(text)
    words = re.findall(r"\w+", text_str)
    return np.array([
        len(text_str),
        len(words),
        sum(len(w) for w in words)/len(words) if words else 0,
        sum(1 for c in text_str if c.isupper()) / len(text_str) if text_str else 0,
        text_str.count("!"),
        text_str.count("?"),
        len(re.findall(r"[!?]{2,}", text_str)),
        count_political_terms(text_str),
        len(re.findall(r"#\w+", text_str)),
        len(re.findall(r"@\w+", text_str)),
        len(re.findall(r"http\S+|www\S+|https\S+", text_str)),
        len(re.findall(r"\.\.+", text_str)),
        int(text_str.strip().lower().startswith("rt @"))
    ]).reshape(1, -1)

# BERT Embedding mit Device-Sicherheit
def embed_single_text(text, _tokenizer, _model, max_len=64):
    """Generiert Embeddings mit sicherer Gerätezuweisung"""
    try:
        with torch.no_grad():
            inputs = _tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_len
            ).to(_model.device)
            
            outputs = _model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().reshape(1, -1)
            
    except RuntimeError as e:
        st.error(f"BERT-Verarbeitungsfehler: {str(e)}")
        return np.zeros((1, 768))  # Fallback

# Streamlit UI
st.set_page_config(page_title="Bundestags-Tweet Analyse", layout="wide")
st.title("🇩🇪 Parteivorhersage für Tweets")

with st.container():
    tweet = st.text_area(
        "Tweet eingeben:",
        placeholder="Beispiel: 'Wir müssen die Klimakrise entschlossen bekämpfen...'",
        max_chars=280,
        height=150
    )
    
    if st.button("Analyse starten", type="primary"):
        if not tweet.strip():
            st.warning("Bitte geben Sie einen Tweet-Text ein")
        else:
            with st.spinner("Textanalyse wird durchgeführt..."):
                try:
                    # 1. TF-IDF Features
                    X_tfidf = vectorizer.transform([tweet])
                    
                    # 2. BERT Embeddings
                    X_bert = embed_single_text(tweet, tokenizer, bert_model)
                    
                    # 3. Numerische Features
                    X_eng = scaler.transform(extract_features(tweet))
                    
                    # Kombination und Vorhersage
                    X_combined = np.hstack([
                        X_tfidf.toarray(), 
                        X_bert, 
                        X_eng
                    ])
                    
                    prediction = model.predict(X_combined)[0]
                    st.success(f"**Vorhergesagte Partei:** {prediction}")
                    
                    if hasattr(model, "predict_proba"):
                        st.subheader("Wahrscheinlichkeitsverteilung")
                        probs = model.predict_proba(X_combined)[0]
                        st.bar_chart({
                            party: prob 
                            for party, prob in zip(model.classes_, probs)
                        })
                        
                except Exception as e:
                    st.error(f"Analysefehler: {str(e)}")

# Footer
st.divider()
st.caption("""
⚠️ **Hinweis:** 
- Modell wurde mit Bundestags-Tweets (2017-2023) trainiert
- Maximale Eingabelänge: 280 Zeichen
- Unterstützt Deutsch und politische Terminologie
""")
