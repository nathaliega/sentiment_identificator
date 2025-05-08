from src.data_preprocessing import preprocess_single_review
from src.model import RNN
from src.config import EMBED_DIM, RNN_HIDDEN_SIZE, FC_HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, DATA_PATH
import torch
import torch
import pickle
from src.model import RNN
from src.config import EMBED_DIM, RNN_HIDDEN_SIZE, FC_HIDDEN_SIZE

with open("outputs/vocab/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = RNN(len(vocab), EMBED_DIM, RNN_HIDDEN_SIZE, FC_HIDDEN_SIZE)
model.load_state_dict(torch.load("outputs/models/model.pt"))
model.eval()

def predict_sentiment(text, threshold=0.5):
    model.eval()
    with torch.no_grad():
        text_tensor, lengths = preprocess_single_review(text, vocab)
        output = model(text_tensor, lengths)[:, 0]
        prob = output.item()
        sentiment = "Positive" if prob >= threshold else "Negative"
        return sentiment, prob

if __name__ == "__main__":
    while True:
        review = input("Enter a review (or 'q' to quit): ")
        if review.lower() == 'q':
            break
        sentiment, prob = predict_sentiment(review)
        print(f"Sentiment: {sentiment} (Confidence: {prob:.4f})")
