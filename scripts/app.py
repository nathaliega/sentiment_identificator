import streamlit as st
import torch
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import preprocess_single_review
from src.model import RNN
from src.config import EMBED_DIM, RNN_HIDDEN_SIZE, FC_HIDDEN_SIZE

with open("outputs/vocab/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = RNN(len(vocab), EMBED_DIM, RNN_HIDDEN_SIZE, FC_HIDDEN_SIZE)
model.load_state_dict(torch.load("outputs/models/model.pt", map_location=torch.device('cpu')))
model.eval()

def predict_sentiment(text, threshold=0.5):
    if not text.strip():
        return "", 0.0
    with torch.no_grad():
        text_tensor, lengths = preprocess_single_review(text, vocab)
        output = model(text_tensor, lengths)[:, 0]
        prob = output.item()
        sentiment = "Positive" if prob >= threshold else "Negative"
        return sentiment, prob

# Streamlit app layout
st.set_page_config(page_title="Live Sentiment Analysis", layout="centered")

st.title("🔍 Live Sentiment Analysis")
text_input = st.text_area("Type your review below:", height=200)

if text_input:
    sentiment, prob = predict_sentiment(text_input)
    st.markdown(f"### Sentiment: **{sentiment}**")
    st.markdown(f"Confidence: `{prob:.4f}`")

