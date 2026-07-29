"""
Word embedding utilities.

* `train_word2vec` trains embeddings from scratch on the training corpus
  using gensim (works fully offline).
* `load_glove_embeddings` loads a pretrained GloVe `.txt` vector file
  (e.g. glove.6B.100d.txt) if the user has one available locally.
* `build_embedding_matrix` turns either source into a numpy matrix aligned
  with the vocabulary's word->index mapping, for use as the LSTM's
  embedding layer initializer.
"""
import numpy as np
from gensim.models import Word2Vec


def train_word2vec(tokenized_sentences, vector_size=100, window=5,
                    min_count=1, epochs=20, sg=1, seed=42):
    """Train a Word2Vec model on the (already tokenized) corpus.
    sg=1 -> skip-gram, sg=0 -> CBOW."""
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,
        epochs=epochs,
        seed=seed,
    )
    return model


def build_embedding_matrix_from_w2v(w2v_model, word2idx, embedding_dim):
    vocab_size = len(word2idx)
    matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim)).astype(np.float32)
    matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim, dtype=np.float32)

    found = 0
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
            found += 1
    coverage = found / max(1, vocab_size)
    print(f"Word2Vec coverage: {found}/{vocab_size} tokens ({coverage:.1%})")
    return matrix


def load_glove_embeddings(glove_path, embedding_dim):
    """Load a GloVe txt file into a dict: word -> vector."""
    print(f"Loading GloVe vectors from {glove_path} ...")
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            try:
                vec = np.asarray(values[1:], dtype="float32")
            except ValueError:
                continue
            if len(vec) != embedding_dim:
                continue
            embeddings_index[word] = vec
    print(f"Loaded {len(embeddings_index)} GloVe vectors.")
    return embeddings_index


def build_embedding_matrix_from_glove(embeddings_index, word2idx, embedding_dim):
    vocab_size = len(word2idx)
    matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim)).astype(np.float32)
    matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim, dtype=np.float32)

    found = 0
    for word, idx in word2idx.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            matrix[idx] = vec
            found += 1
    coverage = found / max(1, vocab_size)
    print(f"GloVe coverage: {found}/{vocab_size} tokens ({coverage:.1%})")
    return matrix
