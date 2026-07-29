"""
Vocabulary construction, sequence encoding/padding, and the PyTorch
Dataset used to feed the LSTM.
"""
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(tokenized_texts, min_freq=1, max_vocab_size=20000):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    most_common = [w for w, c in counter.most_common(max_vocab_size) if c >= min_freq]

    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word in most_common:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def encode_tokens(tokens, word2idx, max_len):
    length = min(len(tokens), max_len)
    ids = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [word2idx[PAD_TOKEN]] * (max_len - len(ids))
    return ids, max(length, 1)  # length >= 1 so pack_padded_sequence never sees a 0-length sequence


class SentimentDataset(Dataset):
    def __init__(self, tokenized_texts, labels, word2idx, max_len):
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        ids, length = encode_tokens(self.tokenized_texts[idx], self.word2idx, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


LABEL2IDX = {"negative": 0, "neutral": 1, "positive": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


def encode_labels(labels):
    return [LABEL2IDX[l] for l in labels]
