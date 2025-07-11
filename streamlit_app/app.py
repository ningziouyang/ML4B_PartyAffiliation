import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import os # 导入 os 模块

# ==== 模型配置（相对路径） ====
# 使用 os.path.join 来构建路径，这样可以更好地处理不同操作系统的路径分隔符
# os.path.dirname(__file__) 获取当前脚本所在的目录
# "models" 是模型文件所在的子文件夹名
# 这个方法会构建一个绝对路径，更健壮
BASE_DIR = os.path.dirname(__file__)

MODEL_OPTIONS = {
    "TF-IDF baseline (no_urls)": {
        "model": os.path.join(BASE_DIR, "models", "lr_model_no_urls.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "models", "tfidf_no_urls.joblib"),
        "scaler": None
    },
    "TF-IDF + Extra Features": {
        "model": os.path.join(BASE_DIR, "models", "lr_model_extra_no_urls.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "models", "tfidf_extra_no_urls.joblib"),
        "scaler": os.path.join(BASE_DIR, "models", "scaler_extra_no_urls.joblib")
    },
    "TF-IDF + BERT + Engineered": {
        "model": os.path.join(BASE_DIR, "models", "lr_tfidf_bert_engineered.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "models", "tfidf_vectorizer_bert_engineered.joblib"),
        "scaler": os.path.join(BASE_DIR, "models", "feature_scaler_bert_engineered.joblib")
    }
}

st.title("Parteivorhersage für Bundestags-Tweets 🇩🇪")

choice = st.selectbox("📦 Wähle ein Modell:", list(MODEL_OPTIONS.keys()))
info = MODEL_OPTIONS[choice]

# **重要：在加载文件前添加文件存在性检查**
# 这有助于在开发或部署时快速发现文件路径问题
try:
    model = joblib.load(info["model"])
    vectorizer = joblib.load(info["vectorizer"])
    scaler = joblib.load(info["scaler"]) if info["scaler"] else None
except FileNotFoundError as e:
    st.error(f"❌ 错误：无法找到模型文件。请检查文件路径和模型文件是否已上传。")
    st.error(f"详细错误: {e}")
    st.stop() # 停止应用执行，避免后续错误

use_bert = "BERT" in choice

# ==== BERT ====
if use_bert:
    # 可以在这里添加try-except块，以防模型下载或加载失败
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        bert_model = AutoModel.from_pretrained("bert-base-german-cased")
        bert_model.eval()
    except Exception as e:
        st.error(f"❌ BERT 模型加载失败。请检查网络连接或模型名称。")
        st.error(f"详细错误: {e}")
        st.stop()


# ==== Feature Engineering ====
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def count_emojis(text):
    try:
        import emoji
        return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)
    except ImportError:
        # 如果 emoji 库没有安装，则使用一个简单的替代方案
        return str(text).count(":")

def extract_features(text):
    feats = [
        len(str(text)),
        len(str(text).split()),
        avg_word_length(text),
        uppercase_ratio(text),
        str(text).count("!"),
        str(text).count("?"),
        multi_punct_count(text),
        count_political_terms(text),
        count_emojis(text),
        count_hashtags(text),
        count_mentions(text),
        count_urls(text),
        count_dots(text),
        is_retweet(text),
    ]
    return np.array(feats).reshape(1, -1)

def avg_word_length(text):
    words = re.findall(r"\w+", str(text))
    return sum(len(w) for w in words) / len(words) if words else 0

def uppercase_ratio(text):
    text = str(text)
    return sum(1 for c in text if c.isupper()) / len(text) if text else 0

def multi_punct_count(text): return len(re.findall(r"[!?]{2,}", str(text)))
def count_political_terms(text): return sum(1 for w in POLITICAL_TERMS if w in str(text).lower())
def count_hashtags(text): return len(re.findall(r"#\w+", str(text)))
def count_mentions(text): return len(re.findall(r"@\w+", str(text)))
def count_urls(text): return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))
def count_dots(text): return len(re.findall(r"\.\.+", str(text)))
def is_retweet(text): return int(str(text).strip().lower().startswith("rt @"))

def embed_single_text(text):
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        output = bert_model(**encoded)
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)

# ==== UI Eingabe ====
st.markdown("✏️ **Gib einen Bundestags-Tweet ein:**")
tweet = st.text_area("", placeholder="Wir fordern mehr Klimaschutz und soziale Gerechtigkeit für alle...")

if tweet and st.button("🔮 Vorhersagen"):
    # 确保 scaler 和 bert_model 在使用前已被正确初始化
    X_tfidf = vectorizer.transform([tweet])
    
    X_eng_scaled = None
    if scaler: # 只有当 scaler 存在时才进行特征提取和缩放
        X_eng = extract_features(tweet)
        X_eng_scaled = scaler.transform(X_eng)
    
    X_bert = None
    if use_bert and 'bert_model' in locals(): # 确保 bert_model 变量已定义
        X_bert = embed_single_text(tweet)

    # === Merkmals-Kombination ===
    if use_bert and X_bert is not None:
        if X_eng_scaled is not None:
            X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
        else:
            X_all = np.hstack([X_tfidf.toarray(), X_bert]) # 如果没有 scaler，只有 tfidf 和 bert
    elif X_eng_scaled is not None:
        X_all = np.hstack([X_tfidf.toarray(), X_eng_scaled])
    else:
        X_all = X_tfidf.toarray() # 确保这里也是 numpy array

    try:
        pred = model.predict(X_all)[0]
        st.success(f"🗳️ **Vorhergesagte Partei:** {pred}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_all)[0]
            st.subheader("📊 Wahrscheinlichkeit je Partei")
            st.bar_chart({p: float(prob) for p, prob in zip(model.classes_, probs)})
    except Exception as e:
        st.error(f"❌ 预测时发生错误：{e}")
        st.warning("请确保所有模型和缩放器文件都已正确加载且非空。")


st.markdown("---")
st.markdown("🔍 Dieses Tool kombiniert klassische Textmerkmale (TF-IDF), BERT-Embeddings und engineered Features zur Klassifikation von Bundestags-Tweets nach Partei.")
