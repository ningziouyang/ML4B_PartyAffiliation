import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel

# Konfiguration und Konstanten
MODEL_PATH = "models/lr_tfidf_bert_engineered.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_bert_engineered.joblib"
SCALER_PATH = "models/feature_scaler_bert_engineered.joblib"
BERT_PATH = "bert-base-german-cased"

# Automatische Geräteerkennung (priorisiert GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ressourcen cachen um wiederholtes Laden zu vermeiden
@st.cache_resource
def load_artifacts():
    """Lädt alle vortrainierten Modelle und Tools"""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        bert_model = AutoModel.from_pretrained(BERT_PATH).to(device)
        bert_model.eval()
        return model, vectorizer, scaler, tokenizer, bert_model
    except Exception as e:
        st.error(f"Modellladefehler: {str(e)}")
        raise

# Modelle und Tools laden
model, vectorizer, scaler, tokenizer, bert_model = load_artifacts()

# Liste politischer Begriffe (für Feature Engineering)
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung"
]

# --- Feature Engineering Funktionen ---
def count_political_terms(text):
    """Zählt Vorkommen politischer Begriffe"""
    text = str(text).lower()
    return sum(1 for word in POLITICAL_TERMS if word in text)

def uppercase_ratio(text):
    """Berechnet Großbuchstabenanteil"""
    text = str(text)
    return sum(1 for c in text if c.isupper()) / len(text) if text else 0

def extract_features(text):
    """Extrahiert 13 manuelle Features (ohne Emoji)"""
    words = re.findall(r"\w+", str(text))
    feats = [
        len(str(text)),                              # Zeichenanzahl
        len(words),                                  # Wortanzahl
        sum(len(w) for w in words)/len(words) if words else 0,  # Durchschn. Wortlänge
        uppercase_ratio(text),                       # Großbuchstabenanteil
        str(text).count("!"),                        # Ausrufezeichen
        str(text).count("?"),                        # Fragezeichen
        len(re.findall(r"[!?]{2,}", str(text))),     # Mehrfachzeichen
        count_political_terms(text),                 # Politische Begriffe
        len(re.findall(r"#\w+", str(text))),         # Hashtags
        len(re.findall(r"@\w+", str(text))),         # Erwähnungen
        len(re.findall(r"http\S+|www\S+|https\S+", str(text))),  # URLs
        len(re.findall(r"\.\.+", str(text))),        # Mehrfachpunkte
        int(str(text).strip().lower().startswith("rt @"))  # Retweet-Kennung
    ]
    return np.array(feats).reshape(1, -1)

# --- BERT Embedding Funktion ---
def embed_single_text(text, tokenizer, model, max_len=64):
    """Erzeugt BERT-Embedding für einzelnen Text"""
    device = next(model.parameters()).device  # Aktuelles Gerät des Modells
    with torch.no_grad():
        # Eingabedaten auf Modellgerät verschieben
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        ).to(device)
        output = model(**encoded)
        # [CLS]-Token Embedding als numpy Array
        cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_emb.reshape(1, -1)

# --- Streamlit UI ---
st.title("🇩🇪 Deutscher Bundestag Tweet-Analyse")
st.markdown("ML-Modell mit TF-IDF, BERT-Embeddings und manuellen Features")

tweet = st.text_area(
    "Tweet eingeben (Deutsch):",
    placeholder="z.B.: 'Klimaschutz ist unsere Priorität...'",
    max_chars=280
)

if st.button("Partei vorhersagen", type="primary"):
    if not tweet.strip():
        st.warning("Bitte gültigen Tweet eingeben!")
    else:
        with st.spinner("Analyse läuft..."):
            try:
                # 1. TF-IDF Features
                X_tfidf = vectorizer.transform([tweet])
                # 2. BERT Embeddings
                X_bert = embed_single_text(tweet, tokenizer, bert_model)
                # 3. Manuelle Features
                X_eng = extract_features(tweet)
                X_eng_scaled = scaler.transform(X_eng)
                # 4. Feature-Kombination
                X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
                # 5. Vorhersage
                pred = model.predict(X_all)[0]
                
                # Ergebnis anzeigen
                st.success(f"**Vorhergesagte Partei**: {pred}")
                
                # Wahrscheinlichkeitsverteilung (falls unterstützt)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_all)[0]
                    st.subheader("Parteien-Wahrscheinlichkeiten")
                    st.bar_chart({
                        partei: prob 
                        for partei, prob in zip(model.classes_, probs)
                    })

            except Exception as e:
                st.error(f"Vorhersagefehler: {str(e)}")

# Fußzeile
st.markdown("---")
st.caption("""
⚠️ Hinweis:  
- Modell trainiert mit Tweets von Bundestagsmitgliedern  
- Beste Ergebnisse mit deutschen politischen Inhalten  
- Maximallänge: 280 Zeichen (Twitter-Limit)
""")
