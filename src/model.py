import torch.nn as nn

# class RNN(nn.Module):
#     def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
#         super().__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings=vocab_size,
#             embedding_dim=embed_dim,
#             padding_idx=0
#         )
#         self.rnn = nn.LSTM(
#             input_size=embed_dim,
#             hidden_size=rnn_hidden_size,
#             batch_first=True
#         )
#         self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(fc_hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, text, lengths):
#         embedded = self.embedding(text)
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(
#             embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
#         )
#         packed_output, (hidden, _) = self.rnn(packed_embedded)
#         out = hidden[-1]
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         return out

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embedded = self.embedding_dropout(embedded)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        out = self.dropout(hidden[-1])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)
