import re
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

def convert_rating_to_sentiment(rating):
    """
    Converts numerical rating into 'positive' or 'negative' label.
    """
    return 'negative' if rating < 3 else 'positive'


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF\u2B50\u2934\u2B06\u2194\u25AA\u25AB\u2B06\u25FE\u274C\u274E\u2753\u2754\u2764\u2728\u263A\u263B]+")
    re.sub(emoji_pattern, '', text)
    return text.strip()


def tokenizer(text):
    return text.split()


def build_vocab(train_dataset, df):
    token_counts = Counter()
    for idx in train_dataset.indices: 
        line = df.iloc[idx]  
        tokens = tokenizer(line['review'])
        token_counts.update(tokens)

    sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {token: idx+2 for idx, (token, _) in enumerate(sorted_by_freq_tuples)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    return vocab


def text_to_int(text, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenizer(text)]


def label_to_int(label):
    return 1 if label == "positive" else 0


def pad_sequences(sequences, padding_value=0):
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []

    for _label, _text in batch:
        label_list.append(label_to_int(_label))
        int_text = torch.tensor(text_to_int(_text))
        text_list.append(int_text)
        lengths.append(len(int_text))
    
    padded_text_list = pad_sequences(text_list)
    
    return padded_text_list, torch.tensor(label_list, dtype=torch.float32), torch.tensor(lengths)

