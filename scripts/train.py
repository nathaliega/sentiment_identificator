from src.data_preprocessing import clean_df, collate_batch, build_vocab
from src.dataset import ReviewDataset
from torch.utils.data import DataLoader
from src.model import RNN
from src.config import BATCH_SIZE, EMBED_DIM, RNN_HIDDEN_SIZE, FC_HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, DATA_PATH
import torch
from src.train_eval import train, evaluate
import logging
from datetime import datetime
import pandas as pd
import os


os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/training_log_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger()


review_df = pd.read_csv(DATA_PATH)


train_dataset, val_dataset, test_dataset = clean_df(df=review_df, rating_col='rating', review_col='review')

vocab = build_vocab(train_dataset, review_df, save_path='outputs/vocab/vocab.pkl')
logger.info(f"Vocabulary size: {len(vocab)}")
logger.info(f"Vocab saved to outputs/vocab/vocab.pkl")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_batch(batch, vocab))
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_batch(batch, vocab))
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_batch(batch, vocab))

model = RNN(len(vocab), EMBED_DIM, RNN_HIDDEN_SIZE, FC_HIDDEN_SIZE)

loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


for epoch in range(NUM_EPOCHS):
    acc_train, loss_train = train(train_dataloader, model, optimizer, loss_func)
    acc_valid, loss_valid = evaluate(val_dataloader, model, loss_func)
    logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
            f"train_loss={loss_train:.4f}, train_acc={acc_train:.4f}, "
            f"val_loss={loss_valid:.4f}, val_acc={acc_valid:.4f}")



acc_test, _ = evaluate(test_dataloader, model, loss_func)
logger.info(f"Test Accuracy: {acc_test:.4f}")

torch.save(model.state_dict(), f"outputs/models/model.pt")
logger.info(f"Model saved to outputs/models/model_{timestamp}.pt")