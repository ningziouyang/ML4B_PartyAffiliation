import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gerät festlegen (CPU)
device = torch.device('cpu')
logger.info(f"Verwende Gerät: {device}")

@st.cache_resource
def load_models():
    """Modelle laden und zwischenspeichern"""
    try:
        MODEL_PATH = "models/lr_tfidf_bert_engineered.joblib"
        VECTORIZER_PATH = "models/tfidf_vectorizer_bert_engineered.joblib"
        SCALER_PATH = "models/feature_scaler_bert_engineered.joblib"
        BERT_PATH = "models/bert"

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)

        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        bert_model = AutoModel.from_pretrained(BERT_PATH, local_files_only=True)
        bert_model.eval()
        bert_model.to(device)

        for param in bert_model.parameters():
            param.requires_grad = False

        logger.info("Modelle erfolgreich geladen")
        return model, vectorizer, scaler, tokenizer, bert_model

    except Exception as e:
        logger.error(f"Fehler beim Laden der Modelle: {e}")
        st.error(f"Modellladefehler: {e}")
        return None, None, None, None, None

POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def extract_features(text):
    """Extrahiere manuelle Merkmale aus dem Text"""
    if not text:
        text = ""
    try:
        def avg_word_length(txt):
            words = re.findall(r"\w+", txt)
            return np.mean([len(w) for w in words]) if words else 0

        features = [
            len(text),
            len(text.split()),
            avg_word_length(text),
            sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            text.count("!"),
            text.count("?"),
            len(re.findall(r"[!?]{2,}", text)),
            sum(1 for w in POLITICAL_TERMS if w in text.lower()),
            len(re.findall(r"#\w+", text)),
            len(re.findall(r"@\w+", text)),
            len(re.findall(r"http\S+|www\S+", text)),
            len(re.findall(r"\.\.+", text)),
            int(text.lower().strip().startswith("rt @"))
        ]
        return np.array(features, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren der Merkmale: {e}")
        return np.zeros((1, 14), dtype=np.float32)

def embed_single_text(text, tokenizer, model, max_len=64):
    """BERT-Embedding aus Text erzeugen"""
    if not text:
        return np.zeros((1, 768), dtype=np.float32)
    try:
        text = str(text).strip()
        if not text:
            return np.zeros((1, 768), dtype=np.float32)

        with torch.no_grad():
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            cls_emb = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            if cls_emb.ndim == 1:
                cls_emb = cls_emb.reshape(1, -1)
            return cls_emb.astype(np.float32)
    except Exception as e:
        logger.error(f"BERT-Fehler: {e}")
        return np.zeros((1, 768), dtype=np.float32)

def predict_party(tweet, model, vectorizer, scaler, tokenizer, bert_model):
    """Partei aus Tweet vorhersagen"""
    try:
        if not tweet or not tweet.strip():
            return None, None, "Eingabetext ist leer."

        tweet = tweet.strip()
        X_tfidf = vectorizer.transform([tweet])
        X_bert = embed_single_text(tweet, tokenizer, bert_model)
        X_eng = extract_features(tweet)
        X_eng_scaled = scaler.transform(X_eng)
        X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
        pred = model.predict(X_all)[0]
        probs = model.predict_proba(X_all)[0] if hasattr(model, "predict_proba") else None
        parties = model.classes_ if probs is not None else None
        return pred, (parties, probs), None
    except Exception as e:
        error_msg = f"Vorhersagefehler: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg

def main():
    st.title("🗳️ Parteivorhersage für Bundestags-Tweets")
    st.markdown("*ML4B-Projekt: Automatische Parteizuordnung basierend auf Tweet-Inhalten*")

    model, vectorizer, scaler, tokenizer, bert_model = load_models()
    if model is None:
        st.error("❌ Modell konnte nicht geladen werden. Bitte Dateien überprüfen.")
        return

    st.markdown("### 📝 Tweet eingeben")
    tweet = st.text_area(
        "Gib einen Bundestags-Tweet ein:",
        height=100,
        placeholder="Beispiel: Wir brauchen mehr Klimaschutz und eine faire Energiewende für alle Bürger..."
    )

    st.markdown("#### 💡 Beispiele:")
    examples = [
        "Wir müssen die Klimakrise ernst nehmen und jetzt handeln! #Klimaschutz",
        "Die Wirtschaftspolitik muss Arbeitsplätze schaffen und Innovation fördern.",
        "Mehr Geld für Bildung und faire Löhne für alle! #Gerechtigkeit"
    ]
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Beispiel 1"):
            tweet = examples[0]
            st.experimental_rerun()
    with col2:
        if st.button("Beispiel 2"):
            tweet = examples[1]
            st.experimental_rerun()
    with col3:
        if st.button("Beispiel 3"):
            tweet = examples[2]
            st.experimental_rerun()

    if st.button("🔍 Partei vorhersagen"):
        if not tweet.strip():
            st.warning("⚠️ Bitte gib einen Tweet ein.")
            return
        with st.spinner("Analysiere..."):
            pred, prob_data, error = predict_party(tweet, model, vectorizer, scaler, tokenizer, bert_model)
        if error:
            st.error(f"❌ {error}")
        elif pred:
            st.success(f"🎯 **Vorhergesagte Partei:** {pred}")
            if prob_data[0] is not None and prob_data[1] is not None:
                parties, probs = prob_data
                st.markdown("### 📊 Wahrscheinlichkeiten")
                prob_dict = {p: float(prob) for p, prob in zip(parties, probs)}
                sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(sorted_probs)
                st.markdown("#### Details:")
                for party, prob in sorted_probs.items():
                    st.write(f"**{party}**: {prob*100:.1f}%")

    st.markdown("---")
    with st.expander("ℹ️ Über das Modell"):
        st.markdown("""
        - 🔤 TF-IDF: Termfrequenz-Vektoren
        - 🧠 BERT: Kontextuelle Sprachrepräsentationen
        - 🔧 Feature Engineering: Länge, Hashtags, Erwähnungen, politische Begriffe etc.
        """)

if __name__ == "__main__":
    main()
