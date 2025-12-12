# app.py
import streamlit as st
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

# ----------------------
# App Title
st.title("IT Ticket Solution Recommender")

# ----------------------
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")  # your uploaded CSV
    df = df[['short_description','close_notes']].dropna().drop_duplicates()
    return df

df = load_data()
st.write("Sample data:", df.sample(5))

# ----------------------
# Load models
@st.cache_resource
def load_models():
    # Custom Attention layer
    class Attention(tf.keras.layers.Layer):
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

    # Load GRU model with Attention
    gru_model = load_model("gru_solution_model.h5", compile=False, custom_objects={"Attention": Attention})
    
    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Load TF-IDF + kNN CBR
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    knn = joblib.load("knn_cbr_model.pkl")
    
    return gru_model, tokenizer, tfidf, knn

gru_model, tokenizer, tfidf, knn = load_models()

# ----------------------
# Helper functions
def predict_gru(issue_text, top_n=1):
    """
    Predict solution using GRU.
    Returns the top N most common close_notes in the predicted cluster.
    """
    seq = tokenizer.texts_to_sequences([issue_text])
    max_len = 100  # same as used during training
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = gru_model.predict(pad)
    cluster_id = pred.argmax()
    
    # Find the most frequent close_notes in the dataset as "solution"
    # If top_n > 1, return multiple possible solutions
    solutions = df['close_notes'].iloc[np.argsort(-pred[0])][:top_n].tolist()
    return solutions[0]  # return first/top solution

def predict_cbr(issue_text, top_k=5):
    query_vec = tfidf.transform([issue_text])
    distances, indices = knn.kneighbors(query_vec, n_neighbors=top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = 1 - dist
        issue = df.iloc[idx]['short_description']
        solution = df.iloc[idx]['close_notes']
        results.append({
            "similarity": similarity,
            "issue": issue,
            "solution": solution
        })
    return results

# ----------------------
# User input
user_input = st.text_input("Describe your IT issue:")

if user_input:
    st.subheader("GRU Model Prediction (AI Suggested Solution)")
    solution_text = predict_gru(user_input)
    st.write(solution_text)
    
    st.subheader("kNN CBR Top Matches")
    results = predict_cbr(user_input)
    for i, r in enumerate(results, 1):
        st.write(f"Match {i}:")
        st.write(f"- Similarity: {r['similarity']:.2f}")
        st.write(f"- Issue: {r['issue']}")
        st.write(f"- Solution: {r['solution']}")
        st.write("---")
