# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import pickle

# ----------------------
# App Title
st.title("IT Ticket Solution Recommender")

# ----------------------
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic-it-call-center-tickets.csv")
    df = df[['short_description','close_notes']].dropna().drop_duplicates()
    return df

df = load_data()
st.write("Sample data:", df.sample(5))

# ----------------------
# Load models
@st.cache_resource
def load_models():
    # GRU model
    from tensorflow.keras.models import load_model
    gru_model = load_model("gru_solution_model.h5")
    # Tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    # TF-IDF + kNN
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    knn = joblib.load("knn_cbr_model.pkl")
    # KMeans clusters
    kmeans = joblib.load("kmeans_cluster_model.pkl")
    return gru_model, tokenizer, tfidf, knn, kmeans

gru_model, tokenizer, tfidf, knn, kmeans = load_models()

# ----------------------
# Helper Functions
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_gru(issue_text):
    seq = tokenizer.texts_to_sequences([issue_text])
    pad = pad_sequences(seq, maxlen=tokenizer.num_words if hasattr(tokenizer, 'num_words') else 100, padding='post')
    pred = gru_model.predict(pad)
    cluster_id = pred.argmax()
    solution = df[df['solution_cluster'] == cluster_id]['close_notes'].mode()[0]
    return solution

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
# User Input
user_input = st.text_input("Describe your issue:")

if user_input:
    st.subheader("GRU Model Prediction")
    st.write(predict_gru(user_input))
    
    st.subheader("kNN CBR Top Matches")
    results = predict_cbr(user_input)
    for r in results:
        st.write(f"**Similarity:** {r['similarity']:.2f}")
        st.write(f"**Issue:** {r['issue']}")
        st.write(f"**Solution:** {r['solution']}")
        st.write("---")
