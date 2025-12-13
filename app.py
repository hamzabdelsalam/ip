import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="IT Helpdesk AI", layout="wide")

# Download NLTK data (cached to avoid redownloading)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

download_nltk_data()

# ==========================================
# 2. UTILITY FUNCTIONS (Preprocessing)
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    pos_tags = pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized)

# Custom Attention Layer (Must match the one in Notebook)
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

    # Load GRU Model with Custom Layer
    model = load_model("gru_solution_model.h5", custom_objects={'Attention': Attention})
    
    return df, max_len, tokenizer, tfidf, knn, model

try:
    df, max_len, tokenizer, tfidf, knn, model = load_all_models()
    st.success("System ready. Models loaded successfully.")
except Exception as e:
    st.error(f"Error loading models. Make sure you ran the export script in the notebook! Error: {e}")
    st.stop()

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
def predict_solution_gru(text):
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
user_input = st.text_area("Describe the issue:", placeholder="e.g., My Outlook email is not syncing and asking for password")

col1, col2 = st.columns(2)

if st.button("Analyze Ticket"):
    if not user_input:
        st.warning("Please enter a description first.")
    else:
        with st.spinner("Analyzing..."):
            
            # --- Method 1: Deep Learning Prediction ---
            st.markdown("### 🧠 AI Recommended Solution (GRU Model)")
            prediction, confidence = predict_solution_gru(user_input)
            
            st.info(f"**Predicted Solution:**\n\n{prediction}")
            st.caption(f"Model Confidence: {confidence:.2f}%")
            
            st.markdown("---")
            
            # --- Method 2: Similar Cases (KNN) ---
            st.markdown("### 📚 Similar Historical Cases (Case-Based Reasoning)")
            similar_cases = get_similar_cases_knn(user_input)
            
            for i, (sim, issue, sol) in enumerate(similar_cases):
                with st.expander(f"Case #{i+1} (Similarity: {sim:.1f}%)"):
                    st.markdown(f"**Original Issue:** {issue}")
                    st.markdown(f"**Resolution:** {sol}")

# Sidebar info
st.sidebar.header("System Stats")
st.sidebar.metric("Total Historical Tickets", len(df))
st.sidebar.markdown("### How it works")
st.sidebar.info(
    """
    **1. Classification:** A GRU Neural Network with Attention predicts the category of the issue and suggests the most common fix.
    
    **2. Search:** A KNN algorithm finds the closest matching tickets from the past using text similarity.
    """
)
