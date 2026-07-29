"""
Builds a REAL, diverse labeled sentiment dataset (as opposed to the tiny
templated one in generate_sample_data.py) using corpora bundled with NLTK,
so no external dataset download/account is required:

  * positive / negative  -> nltk 'sentence_polarity' corpus: 5331 positive
    and 5331 negative one-sentence movie-review snippets (Pang & Lee, Rotten
    Tomatoes critic excerpts). This is short, declarative, review-style
    English ("a well-made, effective thriller") - the same register as
    product/service reviews - so it covers everyday sentiment vocabulary
    ("excellent", "disappointing", "dirty", "slow", "delicious", ...) far
    better than social-media slang does.
  * neutral               -> nltk 'reuters' news corpus (factual,
    non-opinionated reporting sentences), which gives a real, large,
    non-polar vocabulary for the neutral class

This fixes the generalization problem the synthetic dataset had: because
it only contained ~600 sentences built from 15 templates, most everyday
words ("love", "amazing", "delicious", "perfectly"...) were never seen in
training and fell back to <UNK>, so the LSTM effectively ignored the input.
An earlier version of this script used Twitter data for positive/negative,
which fixed the OOV collapse but still generalized poorly to clean,
grammatical review sentences because tweets are a very different register
(hashtags, @mentions, emoticons, abbreviations) - sentence_polarity is a
much closer domain match.

Usage:
    python data/build_real_dataset.py
    # writes data/real_reviews.csv, then:
    python main.py --data data/real_reviews.csv --embedding_type glove --glove_path <path>
    # (or --embedding_type word2vec if you don't have a GloVe file)
"""
import csv
import os
import random
import re

import nltk

random.seed(42)


def ensure_corpora():
    for pkg, path in [
        ("sentence_polarity", "corpora/sentence_polarity"),
        ("reuters", "corpora/reuters"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def build_pos_neg(n_per_class=2500):
    from nltk.corpus import sentence_polarity

    pos_sents = list(sentence_polarity.sents(categories="pos"))
    neg_sents = list(sentence_polarity.sents(categories="neg"))
    random.shuffle(pos_sents)
    random.shuffle(neg_sents)

    def tokens_to_text(tokens):
        text = " ".join(tokens)
        # tidy spacing: "movie 's" -> "movie's", "word ." -> "word."
        text = re.sub(r"\s+'", "'", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    pos = [tokens_to_text(t) for t in pos_sents]
    neg = [tokens_to_text(t) for t in neg_sents]

    n = min(n_per_class, len(pos), len(neg))
    rows = [(t, "positive") for t in pos[:n]] + [(t, "negative") for t in neg[:n]]
    return rows


def build_neutral(n_neutral, min_words=6, max_words=28):
    from nltk.corpus import reuters

    sent_lists = list(reuters.sents())
    random.shuffle(sent_lists)

    rows = []
    for tokens in sent_lists:
        if len(rows) >= n_neutral:
            break
        if not (min_words <= len(tokens) <= max_words):
            continue
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # tidy spacing before punctuation
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 15:
            continue
        rows.append((text, "neutral"))
    return rows


def main():
    ensure_corpora()

    n_per_class = 2500
    pos_neg_rows = build_pos_neg(n_per_class=n_per_class)
    neutral_rows = build_neutral(n_neutral=n_per_class)

    # Hand-written plain-English sentences (see handwritten_sentences.py) -
    # these cover the short, everyday register that neither the movie-critic
    # corpus nor the news corpus contains. Duplicated 6x so they make up a
    # meaningful fraction of the final training set (~1080 of ~8580 rows)
    # instead of being drowned out by the ~7500 rows from the other sources.
    from handwritten_sentences import all_sentences
    handwritten_rows = all_sentences() * 6

    rows = pos_neg_rows + neutral_rows + handwritten_rows
    random.shuffle(rows)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "real_reviews.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    from collections import Counter
    counts = Counter(label for _, label in rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print("Class balance:", dict(counts))


if __name__ == "__main__":
    main()
