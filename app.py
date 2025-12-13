import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="IT Helpdesk AI", 
    page_icon="🤖",
    layout="wide"
)

# --- ROBUST NLTK DOWNLOADER (Fixes the LookupError) ---
@st.cache_resource
def setup_nltk():
    """
    Explicitly checks for and downloads required NLTK resources.
    Uses specific paths to avoid false positives.
    """
    # Dictionary of 'nltk_data_path': 'download_name'
    required_resources = {
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }

    for path, download_name in required_resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(download_name, quiet=True)

setup_nltk()

# ==========================================
# 2. UTILITY FUNCTIONS (Preprocessing)
# ==========================================
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def clean_text(text):
    if not isinstance(text, str): return ""
    
    # 1. Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Tokenize
    words = text.split()
    
    # 3. Remove Stopwords
    # We load stopwords inside the function to ensure resource exists
    stops = set(stopwords.words('english'))
    words = [word for word in words if word not in stops]
    
    # 4. POS Tagging (This caused your error before, now fixed)
    pos_tags = pos_tag(words)
    
    # 5. Lemmatization
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    
    return ' '.join(lemmatized)

# --- CUSTOM ATTENTION LAYER ---
# Must exactly match the class used in the Jupyter Notebook training
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, inputs):
        # u_t = tanh(W.h + b)
        u_t = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # attention scores
        att = tf.nn.softmax(tf.tensordot(u_t, self.u, axes=1), axis=1)
        # weighted sum of inputs
        output = tf.reduce_sum(inputs * att, axis=1)
        return output

# ==========================================
# 3. LOAD MODELS
# ==========================================
@st.cache_resource
def load_all_models():
    # Helper to check if file exists
    required_files = [
        "processed_tickets.csv", "metadata.pkl", "tokenizer.pkl", 
        "tfidf_vectorizer.pkl", "knn_cbr_model.pkl", "gru_solution_model.h5"
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing file: {f}. Please run the export script in your notebook.")
            st.stop()

    # Load Data
    df = pd.read_csv("processed_tickets.csv")
    
    # Load Metadata
    with open("metadata.pkl", "rb") as f:
        meta = pickle.load(f)
        max_len = meta['max_len']

    # Load Tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load TF-IDF & KNN
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    knn = joblib.load("knn_cbr_model.pkl")

    # Load GRU Model
    model = load_model("gru_solution_model.h5", custom_objects={'Attention': Attention})
    
    return df, max_len, tokenizer, tfidf, knn, model

# Load models once
df, max_len, tokenizer, tfidf, knn, model = load_all_models()

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
def predict_solution_gru(text):
    """Deep Learning prediction using GRU + Attention"""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    
    pred = model.predict(pad)
    cluster_id = pred.argmax()
    confidence = np.max(pred) * 100
    
    # Get most frequent solution for this cluster
    suggested_fix = df[df['solution_cluster'] == cluster_id]['close_notes'].mode()[0]
    return suggested_fix, confidence

def get_similar_cases_knn(text, k=3):
    """Case-Based Reasoning using KNN"""
    cleaned = clean_text(text)
    query_vec = tfidf.transform([cleaned])
    distances, indices = knn.kneighbors(query_vec, n_neighbors=k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = (1 - dist) * 100
        original_issue = df.iloc[idx]['short_description']
        solution = df.iloc[idx]['close_notes']
        results.append((similarity, original_issue, solution))
    return results

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("🤖 AI IT Helpdesk Assistant")
st.markdown("Enter a ticket description below to get an AI-suggested solution or find similar historical cases.")

# Input Area
user_input = st.text_area("Describe the issue:", placeholder="e.g., My Outlook email is not syncing and asking for password", height=100)

col1, col2 = st.columns([1, 1])

if st.button("Analyze Ticket", type="primary"):
    if not user_input:
        st.warning("Please enter a description first.")
    else:
        with st.spinner("Analyzing..."):
            
            # --- Method 1: Deep Learning Prediction ---
            st.subheader("🧠 Recommended Solution (AI Model)")
            prediction, confidence = predict_solution_gru(user_input)
            
            # Visual indicator of confidence
            if confidence > 80:
                st.success(f"High Confidence: {confidence:.1f}%")
            elif confidence > 50:
                st.warning(f"Medium Confidence: {confidence:.1f}%")
            else:
                st.error(f"Low Confidence: {confidence:.1f}%")
            
            st.info(f"**Resolution Step:**\n\n{prediction}")
            
            st.markdown("---")
            
            # --- Method 2: Similar Cases (KNN) ---
            st.subheader("📚 Similar Historical Cases")
            similar_cases = get_similar_cases_knn(user_input)
            
            for i, (sim, issue, sol) in enumerate(similar_cases):
                with st.expander(f"Case #{i+1} (Similarity: {sim:.1f}%)"):
                    st.markdown(f"**Original Issue:** {issue}")
                    st.markdown(f"**Resolution:** {sol}")

# Sidebar
st.sidebar.header("System Stats")
st.sidebar.metric("Total Historical Tickets", len(df))
st.sidebar.markdown("### How it works")
st.sidebar.info(
    """
    **1. Neural Network:** A GRU model understands the sequence of your words to predict the category of the problem.
    
    **2. Vector Search:** A KNN algorithm compares your text mathematically to thousands of past tickets to find the closest match.
    """
)
