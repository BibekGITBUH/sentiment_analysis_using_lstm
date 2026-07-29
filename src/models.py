"""
Models:
  * LSTMClassifier - embedding layer (initialized from Word2Vec/GloVe) +
    (bi-)LSTM + dropout + fully connected classifier head.
  * build_baseline_pipeline - TF-IDF + Logistic Regression, used as the
    classic ML comparison point against the deep sequence model.
"""
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 embedding_matrix=None, n_layers=1, bidirectional=True,
                 dropout=0.4, pad_idx=0, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, output_dim)
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        # x: (batch, seq_len), lengths: (batch,) true (unpadded) sequence lengths
        embedded = self.embedding(x)                     # (batch, seq_len, emb_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)                # hidden: (n_layers*dirs, batch, hidden)

        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]

        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)                         # (batch, output_dim)
        return logits


def build_baseline_pipeline(max_features=10000, ngram_range=(1, 2), C=1.0):
    """TF-IDF + Logistic Regression baseline for comparison against the LSTM."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("clf", LogisticRegression(max_iter=1000, C=C, class_weight="balanced")),
    ])
