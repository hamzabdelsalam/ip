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
# 1. SETUP & CONFIGURATION (CRITICAL FIX)
# ==========================================
st.set_page_config(
    page_title="IT Helpdesk AI", 
    page_icon="🤖",
    layout="wide"
)

# --- CRITICAL NLTK FIX: Setting Data Path ---
@st.cache_resource
def setup_nltk_and_download_resources():
    """
    Forces the download of necessary NLTK components into a 
    specific, user-writable directory and sets the NLTK data path.
    This resolves persistent LookupErrors in cloud environments.
    """
    # Define a custom directory for NLTK data (e.g., inside the app root)
    NLTK_DATA_DIR = os.path.join(os.getcwd(), ".nltk_data")
    
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(NLTK_DATA_DIR):
        os.makedirs(NLTK_DATA_DIR)
        
    # 2. Tell NLTK to look in this directory
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_DIR)
        
    # 3. Download the packages, specifying the new directory
    packages = ['stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for package in packages:
        try:
            # Check if package is already downloaded in the new location
            # We use a known path to check for presence
            nltk.data.find(f'taggers/{package}') 
        except LookupError:
            # Download it to the specified directory
            nltk.download(package, download_dir=NLTK_DATA_DIR, quiet=True)

    # Final check for the specific resource that keeps failing
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
        st.success("NLTK resources (including POS Tagger) loaded successfully.")
    except LookupError:
        st.error("FATAL ERROR: 'averaged_perceptron_tagger' is still missing. Please ensure internet access and permissions.")
        st.stop()


# Call the setup function immediately
setup_nltk_and_download_resources()

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
    stops = set(stopwords.words('english'))
    words = [word for word in words if word not in stops]
    
    # 4. POS Tagging (Source of the LookupError, now should be fixed)
    pos_tags = pos_tag(words)
    
    # 5. Lemmatization
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    
    return ' '.join(lemmatized)

# --- CUSTOM ATTENTION LAYER ---
# Must match the class definition used during model training
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
        u_t = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        att = tf.nn.softmax(tf.tensordot(u_t, self.u, axes=1), axis=1)
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
            st.error(f"Missing file: {f}. Please run the export script in your notebook and ensure all files are in the same directory.")
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

    # Load GRU Model (with custom object)
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
    
    # Use model.predict, suppress output
    with st.spinner('Predicting cluster...'):
        pred = model.predict(pad, verbose=0)
    
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
        st.markdown("---")
        with st.spinner("Running AI Analysis..."):
            
            # --- Method 1: Deep Learning Prediction ---
            st.subheader("🧠 Recommended Solution (AI Model)")
            prediction, confidence = predict_solution_gru(user_input)
            
            # Visual indicator of confidence
            col_conf, col_gap = st.columns([1, 3])
            with col_conf:
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
                    st.markdown(f"**Original Issue (Pre-processed):** {issue}")
                    st.markdown(f"**Resolution (Pre-processed):** {sol}")

# Sidebar
st.sidebar.header("System Stats")
st.sidebar.metric("Total Historical Tickets", len(df))
st.sidebar.markdown("### 🛠️ Model Architecture")
st.sidebar.info(
    """
    **1. Neural Network:** A **GRU** (Gated Recurrent Unit) model with an **Attention Layer**  classifies the input issue into 30 solution clusters.
    
    **2. Vector Search:** A **k-Nearest Neighbors (KNN)** model 

[Image of k-Nearest Neighbors clustering]
 uses **TF-IDF** (Term Frequency-Inverse Document Frequency) to find the most similar historical tickets.
    """
)
