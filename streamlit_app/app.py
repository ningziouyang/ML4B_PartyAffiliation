import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import os # 确保导入了 os 模块

# ==== 模型配置（相对路径） ====
# app.py 和模型文件都在 streamlit_app/ 目录下
# 所以 BASE_DIR 就是 app.py 所在的目录，模型直接在这个目录下
BASE_DIR = os.path.dirname(__file__)

MODEL_OPTIONS = {
    "TF-IDF baseline (no_urls)": {
        "model": os.path.join(BASE_DIR, "lr_model_no_urls.joblib"), # 直接在 app.py 所在目录查找
        "vectorizer": os.path.join(BASE_DIR, "tfidf_no_urls.joblib"), # 直接在 app.py 所在目录查找
        "scaler": None
    },
    "TF-IDF + Extra Features": {
        "model": os.path.join(BASE_DIR, "lr_model_extra_no_urls.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "tfidf_extra_no_urls.joblib"),
        "scaler": os.path.join(BASE_DIR, "scaler_extra_no_urls.joblib")
    },
    "TF-IDF + BERT + Engineered": {
        "model": os.path.join(BASE_DIR, "lr_tfidf_bert_engineered.joblib"),
        "vectorizer": os.path.join(BASE_DIR, "tfidf_vectorizer_bert_engineered.joblib"),
        "scaler": os.path.join(BASE_DIR, "feature_scaler_bert_engineered.joblib")
    }
    # 确保所有模型路径都改成了 os.path.join(BASE_DIR, "文件名")
    # 例如：
    # 如果你的 lr_model_combined.joblib 需要被某个模型选项使用，也要这样写：
    # "某个选项": {
    #     "model": os.path.join(BASE_DIR, "lr_model_combined.joblib"),
    #     ...
    # }
}

st.title("Parteivorhersage für Bundestags-Tweets 🇩🇪")

choice = st.selectbox("📦 Wähle ein Modell:", list(MODEL_OPTIONS.keys()))
info = MODEL_OPTIONS[choice]

# **重要：在加载文件前添加文件存在性检查**
# 这有助于在开发或部署时快速发现文件路径问题
try:
    # 确认文件路径是否存在
    if not os.path.exists(info["model"]):
        st.error(f"❌ 错误：模型文件 '{info['model']}' 不存在。请检查文件是否已上传。")
        st.stop()
    model = joblib.load(info["model"])

    if not os.path.exists(info["vectorizer"]):
        st.error(f"❌ 错误：向量化器文件 '{info['vectorizer']}' 不存在。请检查文件是否已上传。")
        st.stop()
    vectorizer = joblib.load(info["vectorizer"])

    if info["scaler"]:
        if not os.path.exists(info["scaler"]):
            st.error(f"❌ 错误：缩放器文件 '{info['scaler']}' 不存在。请检查文件是否已上传。")
            st.stop()
        scaler = joblib.load(info["scaler"])
    else:
        scaler = None

except FileNotFoundError as e:
    st.error(f"❌ 错误：无法找到模型文件。请检查文件路径和模型文件是否已上传。")
    st.error(f"详细错误: {e}")
    st.stop() # 停止应用执行，避免后续错误
except Exception as e: # 捕获其他可能的加载错误，例如文件损坏导致 joblib 无法解析
    st.error(f"❌ 错误：加载模型文件时发生异常。文件可能损坏或内容不完整。")
    st.error(f"详细错误: {e}")
    st.stop()


use_bert = "BERT" in choice

# ... (其余代码保持不变，与您之前提供的代码一致) ...
