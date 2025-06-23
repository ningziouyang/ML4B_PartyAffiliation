import streamlit as st
import joblib
import numpy as np
import re
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/lr_tfidf_bert_engineered.joblib")
        vectorizer = joblib.load("models/tfidf_vectorizer_bert_engineered.joblib")
        scaler = joblib.load("models/feature_scaler_bert_engineered.joblib")
        return model, vectorizer, scaler
    except Exception as e:
        logger.error(f"Fehler beim Laden der Modelle: {e}")
        st.error(f"Modellladefehler: {e}")
        return None, None, None

POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def extract_features(text):
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
            int(text.lower().strip().startswith("rt @")),
            0.0
        ]
        return np.array(features, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        logger.error(f"Fehler bei Merkmalsextraktion: {e}")
        return np.zeros((1, 14), dtype=np.float32)

def predict_party(tweet, model, vectorizer, scaler):
    try:
        if not tweet or not tweet.strip():
            return None, None, "Tweet ist leer."

        tweet = tweet.strip()
        X_tfidf = vectorizer.transform([tweet])
        X_eng = extract_features(tweet)
        X_eng_scaled = scaler.transform(X_eng)
        bert_placeholder = np.zeros((1, 768), dtype=np.float32)
        X_all = np.hstack([X_tfidf.toarray(), bert_placeholder, X_eng_scaled])

        pred = model.predict(X_all)[0]
        probs = model.predict_proba(X_all)[0] if hasattr(model, "predict_proba") else None
        classes = model.classes_ if probs is not None else None
        return pred, (classes, probs), None
    except Exception as e:
        logger.error(f"Fehler bei Vorhersage: {e}")
        return None, None, f"Vorhersagefehler: {str(e)}"

def main():
    st.title("🗳️ Parteivorhersage für Bundestags-Tweets")
    st.markdown("*Verwendet nur TF-IDF & manuelle Features, BERT eingebettet als Platzhalter*")

    model, vectorizer, scaler = load_models()
    if model is None:
        st.error("❌ Modell konnte nicht geladen werden.")
        return

    st.markdown("### 📝 Tweet eingeben")
    tweet = st.text_area("Gib einen Tweet ein:")

    if st.button("🔍 Partei vorhersagen"):
        if not tweet.strip():
            st.warning("⚠️ Bitte gib einen Tweet ein.")
            return
        with st.spinner("Analysiere..."):
            pred, prob_data, error = predict_party(tweet, model, vectorizer, scaler)
        if error:
            st.error(f"❌ {error}")
        elif pred:
            st.success(f"🎯 **Vorhergesagte Partei:** {pred}")
            if prob_data[0] is not None and prob_data[1] is not None:
                st.markdown("### 📊 Wahrscheinlichkeiten")
                prob_dict = {p: float(prob) for p, prob in zip(prob_data[0], prob_data[1])}
                sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(sorted_probs)
                for party, prob in sorted_probs.items():
                    st.write(f"**{party}**: {prob*100:.1f}%")

if __name__ == "__main__":
    main()
