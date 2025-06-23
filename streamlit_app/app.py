import streamlit as st
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查设备并设置
device = torch.device('cpu')  # 强制使用CPU以避免设备不匹配问题
logger.info(f"Using device: {device}")

@st.cache_resource
def load_models():
    """缓存模型加载以提高性能"""
    try:
        # Load artifacts
        MODEL_PATH = "models/lr_tfidf_bert_engineered.joblib"
        VECTORIZER_PATH = "models/tfidf_vectorizer_bert_engineered.joblib"
        SCALER_PATH = "models/feature_scaler_bert_engineered.joblib"
        BERT_PATH = "bert-base-german-cased"

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)

        # 加载BERT模型并设置为评估模式和指定设备
        tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        bert_model = AutoModel.from_pretrained(BERT_PATH)
        bert_model.to(device)  # 确保模型在正确的设备上
        bert_model.eval()
        
        # 禁用梯度计算以节省内存
        for param in bert_model.parameters():
            param.requires_grad = False
            
        logger.info("Models loaded successfully")
        return model, vectorizer, scaler, tokenizer, bert_model
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"模型加载失败: {e}")
        return None, None, None, None, None

# Feature engineering as in training
POLITICAL_TERMS = [
    "klimaschutz", "freiheit", "bürgergeld", "migration", "rente", "gerechtigkeit",
    "steuern", "digitalisierung", "gesundheit", "bildung", "europa", "verteidigung",
    "arbeitsmarkt", "soziales", "integration", "umweltschutz", "innenpolitik"
]

def count_political_terms(text):
    """计算政治术语数量"""
    if not text:
        return 0
    text = str(text).lower()
    return sum(1 for word in POLITICAL_TERMS if word in text)

def uppercase_ratio(text):
    """计算大写字母比例"""
    if not text:
        return 0
    text = str(text)
    if len(text) == 0:
        return 0
    return sum(1 for c in text if c.isupper()) / len(text)

def avg_word_length(text):
    """计算平均单词长度"""
    if not text:
        return 0
    words = re.findall(r"\w+", str(text))
    if not words:
        return 0
    return sum(len(w) for w in words) / len(words)

def multi_punct_count(text):
    """计算多重标点符号数量"""
    if not text:
        return 0
    return len(re.findall(r"[!?]{2,}", str(text)))

def count_emojis(text):
    """计算表情符号数量"""
    if not text:
        return 0
    try:
        import emoji
        return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)
    except ImportError:
        # fallback: just count colons
        return str(text).count(":")

def count_hashtags(text):
    """计算hashtag数量"""
    if not text:
        return 0
    return len(re.findall(r"#\w+", str(text)))

def count_mentions(text):
    """计算@提及数量"""
    if not text:
        return 0
    return len(re.findall(r"@\w+", str(text)))

def count_urls(text):
    """计算URL数量"""
    if not text:
        return 0
    return len(re.findall(r"http\S+|www\S+|https\S+", str(text)))

def count_dots(text):
    """计算连续点数量"""
    if not text:
        return 0
    return len(re.findall(r"\.\.+", str(text)))

def is_retweet(text):
    """检查是否为转推"""
    if not text:
        return 0
    return int(str(text).strip().lower().startswith("rt @"))

def extract_features(text):
    """提取工程特征"""
    if not text:
        text = ""
    
    try:
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
        return np.array(feats, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        # 返回默认特征向量
        return np.zeros((1, 14), dtype=np.float32)

def embed_single_text(text, tokenizer, model, max_len=64):
    """使用BERT生成文本嵌入"""
    if not text:
        return np.zeros((1, 768), dtype=np.float32)
    
    try:
        # 确保输入文本是字符串
        text = str(text).strip()
        if not text:
            return np.zeros((1, 768), dtype=np.float32)
        
        with torch.no_grad():
            # 编码文本
            encoded = tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=max_len, 
                return_tensors="pt"
            )
            
            # 确保所有张量都在同一设备上
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # 前向传播
            output = model(**encoded)
            
            # 获取CLS token的嵌入
            cls_emb = output.last_hidden_state[:, 0, :].squeeze()
            
            # 转换为numpy数组
            cls_emb = cls_emb.cpu().numpy()
            
            # 确保形状正确
            if cls_emb.ndim == 1:
                cls_emb = cls_emb.reshape(1, -1)
                
            return cls_emb.astype(np.float32)
            
    except Exception as e:
        logger.error(f"Error in BERT embedding: {e}")
        # 返回零向量作为fallback
        return np.zeros((1, 768), dtype=np.float32)

def predict_party(tweet, model, vectorizer, scaler, tokenizer, bert_model):
    """预测推文的政党归属"""
    try:
        if not tweet or not tweet.strip():
            return None, None, "输入的推文为空"
        
        tweet = tweet.strip()
        
        # 1. TF-IDF特征
        X_tfidf = vectorizer.transform([tweet])  # (1, 2000)
        
        # 2. BERT嵌入
        X_bert = embed_single_text(tweet, tokenizer, bert_model)  # (1, 768)
        
        # 3. 工程特征
        X_eng = extract_features(tweet)  # (1, 14)
        X_eng_scaled = scaler.transform(X_eng)
        
        # 4. 合并所有特征
        X_all = np.hstack([X_tfidf.toarray(), X_bert, X_eng_scaled])
        
        # 5. 预测
        pred = model.predict(X_all)[0]
        
        # 6. 获取概率（如果支持）
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_all)[0]
            parties = model.classes_
        else:
            parties = None
            
        return pred, (parties, probs), None
        
    except Exception as e:
        error_msg = f"预测过程中发生错误: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg

# 主应用程序
def main():
    st.title("🗳️ Parteivorhersage für Bundestags-Tweets")
    st.markdown("*ML4B-Projekt: Automatische Parteizuordnung basierend auf Tweet-Inhalten*")
    
    # 加载模型
    model, vectorizer, scaler, tokenizer, bert_model = load_models()
    
    if model is None:
        st.error("模型加载失败，请检查模型文件是否存在。")
        return
    
    # 用户界面
    st.markdown("### 📝 Tweet eingeben")
    tweet = st.text_area(
        "Gib einen Bundestags-Tweet ein:",
        height=100,
        placeholder="Beispiel: Wir brauchen mehr Klimaschutz und eine faire Energiewende für alle Bürger..."
    )
    
    # 添加一些示例
    st.markdown("#### 💡 Beispiele zum Testen:")
    examples = [
        "Wir müssen die Klimakrise ernst nehmen und jetzt handeln! #Klimaschutz",
        "Die Wirtschaftspolitik muss Arbeitsplätze schaffen und Innovation fördern.",
        "Mehr Geld für Bildung und faire Löhne für alle! #Gerechtigkeit"
    ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Beispiel 1", key="ex1"):
            tweet = examples[0]
            st.experimental_rerun()
    with col2:
        if st.button("Beispiel 2", key="ex2"):
            tweet = examples[1]
            st.experimental_rerun()
    with col3:
        if st.button("Beispiel 3", key="ex3"):
            tweet = examples[2]
            st.experimental_rerun()
    
    # 预测按钮
    if st.button("🔍 Partei vorhersagen", type="primary"):
        if not tweet or not tweet.strip():
            st.warning("⚠️ Bitte gib einen Tweet ein!")
            return
        
        # 显示加载状态
        with st.spinner("Analysiere Tweet..."):
            pred, prob_data, error = predict_party(
                tweet, model, vectorizer, scaler, tokenizer, bert_model
            )
        
        if error:
            st.error(f"❌ {error}")
            return
        
        if pred:
            # 显示预测结果
            st.success(f"🎯 **Vorhergesagte Partei:** {pred}")
            
            # 显示概率分布
            if prob_data[0] is not None and prob_data[1] is not None:
                parties, probs = prob_data
                st.markdown("### 📊 Wahrscheinlichkeitsverteilung")
                
                # 创建概率字典并排序
                prob_dict = {p: float(prob) for p, prob in zip(parties, probs)}
                sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                
                # 显示条形图
                st.bar_chart(sorted_probs)
                
                # 显示详细概率
                st.markdown("#### 详细概率:")
                for party, prob in sorted_probs.items():
                    percentage = prob * 100
                    st.write(f"**{party}**: {percentage:.1f}%")
    
    # 信息部分
    st.markdown("---")
    with st.expander("ℹ️ Über dieses Modell"):
        st.markdown("""
        **Modell-Features:**
        - 🔤 **TF-IDF Vektorisierung**: Textuelle Inhaltsanalyse
        - 🧠 **BERT Embeddings**: Kontextuelle Wortrepräsentationen (German BERT)
        - 🔧 **Engineered Features**: Tweet-Länge, Hashtags, Mentions, politische Begriffe, etc.
        
        **Hinweise:**
        - Das Modell wurde auf deutschen Bundestags-Tweets trainiert
        - Die Vorhersage basiert auf einer Kombination verschiedener ML-Techniken
        - Ergebnisse sind Wahrscheinlichkeitsschätzungen, keine Garantien
        """)
    
    st.markdown("---")
    st.markdown("*Entwickelt für das ML4B-Projekt | Verwendete Technologien: Streamlit, scikit-learn, Transformers, PyTorch*")

if __name__ == "__main__":
    main()

