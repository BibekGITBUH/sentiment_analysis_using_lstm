"""
Text preprocessing pipeline:
  1. Lowercasing + noise removal (URLs, mentions, hashtags, punctuation, digits)
  2. Tokenization (NLTK)
  3. Stopword removal (NLTK, with a few negation words kept since they
     carry sentiment signal)
  4. Lemmatization (default) or stemming (--stem flag)
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


def ensure_nltk_data():
    """Download required NLTK resources if not already present."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:  # pragma: no cover
                print(f"Warning: could not download NLTK resource '{pkg}': {e}")


ensure_nltk_data()

_STOPWORDS = set(stopwords.words("english"))
# Keep negations - they matter a lot for sentiment ("not good" != "good")
_NEGATIONS = {"no", "not", "nor", "never", "none", "n't", "cannot", "without"}
_STOPWORDS = _STOPWORDS - _NEGATIONS

_LEMMATIZER = WordNetLemmatizer()
_STEMMER = PorterStemmer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s']")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, mentions, hashtag symbols, digits, and
    punctuation (keeping apostrophes so contractions survive tokenization)."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def lemmatize(tokens):
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def stem(tokens):
    return [_STEMMER.stem(t) for t in tokens]


def preprocess_text(text: str, use_stemming: bool = False):
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize/stem.
    Returns a list of processed tokens."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens) if use_stemming else lemmatize(tokens)
    return tokens


def preprocess_corpus(texts, use_stemming: bool = False, show_progress: bool = True):
    """Preprocess a list/Series of raw texts. Returns list of token lists."""
    processed = []
    iterator = texts
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Preprocessing")
        except ImportError:
            pass
    for text in iterator:
        processed.append(preprocess_text(text, use_stemming=use_stemming))
    return processed


if __name__ == "__main__":
    sample = "I LOVED this product!! It's not bad at all, check http://example.com #great @brand"
    print("Original:", sample)
    print("Lemmatized:", preprocess_text(sample, use_stemming=False))
    print("Stemmed:  ", preprocess_text(sample, use_stemming=True))
