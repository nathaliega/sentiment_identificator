import re
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
import pandas as pd
from src.dataset import ReviewDataset
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import pickle
import os

def load_data(df_path):
    """Loads a CSV file into a pandas DataFrame.

    Args:
        df_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(df_path)


def convert_rating_to_sentiment(rating):
    """Converts a numerical rating into binary sentiment.

    Args:
        rating (int or float): Original rating value.

    Returns:
        int: 1 if rating >= 3, else 0.
    """
    return 1 if rating >= 3 else 0


def clean_text(text):
    """Cleans input text by lowercasing, removing punctuation, extra whitespace, and emojis.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF\u2B50\u2934\u2B06\u2194\u25AA\u25AB\u2B06\u25FE\u274C\u274E\u2753\u2754\u2764\u2728\u263A\u263B]+")
    text = re.sub(emoji_pattern, '', text)
    return text.strip()



def clean_df(df, rating_col, review_col):
    """Cleans and prepares a DataFrame for model training by converting ratings and reviews.

    Args:
        df (pandas.DataFrame): Input DataFrame with raw data.
        rating_col (str): Name of the column containing rating values.
        review_col (str): Name of the column containing review text.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.
    """
    df[rating_col] = df[rating_col].apply(convert_rating_to_sentiment)
    df[review_col] = df[review_col].apply(clean_text)

    torch.manual_seed(1)
    reviews_dataset = ReviewDataset(df)
    train_len = int(len(df) * 0.7)
    val_len = int(len(df) * 0.15)
    test_len = len(df) - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(reviews_dataset, lengths=[train_len, val_len, test_len])
    return train_dataset, val_dataset, test_dataset


def tokenizer(text):
    """Tokenizes text and removes English stopwords.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    return [token for token in tokens if token not in stop_words]


def build_vocab(train_dataset, df, save_path=None):
    """Builds a vocabulary dictionary from a training dataset.

    Args:
        train_dataset (Subset): Subset of ReviewDataset for training.
        df (pandas.DataFrame): Original DataFrame.
        save_path (str, optional): Path to save the vocabulary pickle file. Defaults to None.

    Returns:
        Dict[str, int]: Vocabulary mapping from token to index.
    """
    token_counts = Counter()
    for idx in train_dataset.indices: 
        line = df.iloc[idx]  
        tokens = tokenizer(line['review'])
        token_counts.update(tokens)

    sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {token: idx+2 for idx, (token, _) in enumerate(sorted_by_freq_tuples)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)

    return vocab


def text_to_int(text, vocab):
    """Converts text to a list of token indices using a vocabulary.

    Args:
        text (str): Input text.
        vocab (Dict[str, int]): Token-to-index mapping.

    Returns:
        List[int]: List of token indices.
    """
    return [vocab.get(token, vocab["<unk>"]) for token in tokenizer(text)]


def pad_sequences(sequences, padding_value=0):
    """Pads a list of variable-length sequences to the same length.

    Args:
        sequences (List[torch.Tensor]): List of 1D tensors.
        padding_value (int, optional): Value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: Padded 2D tensor of shape (batch_size, max_len).
    """
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def collate_batch(batch, vocab):
    """Custom collate function to be used with DataLoader.

    Args:
        batch (List[Tuple[int, str]]): Batch of label-text pairs.
        vocab (Dict[str, int]): Token-to-index mapping.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded text tensor, label tensor, and lengths tensor.
    """
    label_list, text_list, lengths = [], [], []

    for _label, _text in batch:
        label_list.append(_label)
        int_text = torch.tensor(text_to_int(_text, vocab))
        text_list.append(int_text)
        lengths.append(len(int_text))
    
    padded_text_list = pad_sequences(text_list)
    
    return padded_text_list, torch.tensor(label_list, dtype=torch.float32), torch.tensor(lengths)


def preprocess_single_review(text, vocab):
    """Preprocesses a single review for model inference.

    Args:
        text (str): Raw review text.
        vocab (Dict[str, int]): Token-to-index mapping.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded token indices tensor and length tensor.
    """
    cleaned = clean_text(text)
    int_tokens = text_to_int(cleaned, vocab)
    tensor = torch.tensor(int_tokens, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([len(int_tokens)])
    padded = pad_sequences([tensor.squeeze(0)])  
    return padded, lengths