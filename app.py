# app.py
import streamlit as st
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------
# App Title
st.title("IT Ticket Solution Tester")

# ----------------------
# Load models and tokenizer
@st.cache_resource
def load_models():
    # Load GRU model
    gru_model = load_model("gru_solution_model.h5", compile=False)
    
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
def predict_gru(issue_text):
    seq = tokenizer.texts_to_sequences([issue_text])
    pad = pad_sequences(seq, maxlen=tokenizer.num_words if hasattr(tokenizer, 'num_words') else 100, padding='post')
    pred = gru_model.predict(pad)
    cluster_id = pred.argmax()
    return f"Predicted cluster ID: {cluster_id}"

def predict_cbr(issue_text, top_k=5):
    query_vec = tfidf.transform([issue_text])
    distances, indices = knn.kneighbors(query_vec, n_neighbors=top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = 1 - dist
        results.append({
            "similarity": similarity,
            "example_index": idx
        })
    return results

# ----------------------
# User input
user_input = st.text_input("Enter your IT issue:")

if user_input:
    st.subheader("GRU Model Prediction")
    st.write(predict_gru(user_input))
    
    st.subheader("kNN CBR Top Matches")
    cbr_results = predict_cbr(user_input)
    for i, r in enumerate(cbr_results, 1):
        st.write(f"Match {i}:")
        st.write(f"- Similarity: {r['similarity']:.2f}")
        st.write(f"- Example index in training data: {r['example_index']}")
        st.write("---")
