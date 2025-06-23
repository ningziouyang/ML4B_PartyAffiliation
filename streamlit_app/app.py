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

# Geräteinitialisierung mit Fehlerbehandlung
def initialize_device():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Speicherbereinigung für CUDA
            torch.cuda.empty_cache()
            return device
        return torch.device("cpu")
    except RuntimeError as e:
        st.error(f"Geräteinitialisierungsfehler: {str(e)}")
        return torch.device("cpu")

# Modellladung mit Meta-Tensor-Handling
@st.cache_resource(show_spinner=False)
def load_models():
    """Lädt Modelle mit speziellem Meta-Tensor-Handling"""
    try:
        # 1. Lade sklearn-Modelle
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # 2. BERT-Komponenten (kritischer Abschnitt)
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        
        # Lösung für Meta-Tensor Problem:
        # Phase 1: Zuerst auf CPU laden mit explizitem Speicher
        bert_model = AutoModel.from_pretrained(
            BERT_PATH,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True  # Wichtig für Meta-Tensor Vermeidung
        )
        
        # Phase 2: Dann auf Zielgerät verschieben
        device = initialize_device()
        if str(device) == "cuda":
            bert_model = bert_model.to(device)
        
        bert_model.eval()
        return model, vectorizer, scaler, tokenizer, bert_model
        
    except Exception as e:
        st.error(f"Modellladefehler: {str(e)}")
        raise

# Systeminitialisierung
try:
    device = initialize_device()
    model, vectorizer, scaler, tokenizer, bert_model = load_models()
except Exception as e:
    st.error(f"Systemstart fehlgeschlagen: {str(e)}")
    st.stop()

# Feature Engineering (wie zuvor)
POLITICAL_TERMS = [...]  # Ihre politische Begriffsliste

def extract_features(text):
    [...]  # Ihre Feature-Extraktion

# Sicherer BERT-Embedding Generator
def get_bert_embedding(text, _tokenizer, _model, max_len=64):
    """Generiert Embeddings mit Meta-Tensor-Sicherheit"""
    try:
        # 1. Input vorbereiten
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        
        # 2. Explizite Gerätezuweisung
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        
        # 3. Berechnung mit GPU/CPU-Synchronisation
        with torch.no_grad():
            outputs = _model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Sicherer Transfer zu CPU
            if embeddings.is_cuda:
                torch.cuda.synchronize()
            
            return embeddings.cpu().numpy().reshape(1, -1)
            
    except Exception as e:
        st.error(f"Embedding-Fehler: {str(e)}")
        return np.zeros((1, 768))  # Notfall-Embedding

# Streamlit UI (wie zuvor)
[...]
