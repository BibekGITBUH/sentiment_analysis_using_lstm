"""
End-to-end sentiment analysis pipeline.

Usage:
    python main.py --data data/sample_reviews.csv --epochs 15 --embedding_type word2vec
    python main.py --data data/sample_reviews.csv --embedding_type glove --glove_path glove.6B.100d.txt

Steps:
    1. Load & clean dataset                         (deliverable: cleaned dataset)
    2. Preprocess text (tokenize/stopwords/lemma)    (deliverable: preprocessing pipeline)
    3. Build vocab + train/load embeddings           (Word2Vec or GloVe)
    4. Train baseline (TF-IDF + LogisticRegression)  (deliverable: trained model #1)
    5. Train LSTM classifier on embeddings           (deliverable: trained model #2)
    6. Evaluate both -> accuracy/precision/recall/F1 (deliverable: evaluation + comparison)
    7. Error analysis on the stronger/LSTM model     (deliverable: error analysis section)
"""
import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch

from src.preprocessing import preprocess_corpus, clean_text
from src.embeddings import (
    train_word2vec,
    build_embedding_matrix_from_w2v,
    load_glove_embeddings,
    build_embedding_matrix_from_glove,
)
from src.dataset import build_vocab, SentimentDataset, encode_labels, LABEL2IDX, IDX2LABEL
from src.models import LSTMClassifier, build_baseline_pipeline
from src.train import train_baseline, train_lstm
from src.evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    save_classification_report,
    compare_models,
)
from src.error_analysis import build_error_dataframe, write_error_report

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Sentiment analysis pipeline")
    p.add_argument("--data", type=str, default="data/sample_reviews.csv",
                    help="CSV file with 'text' and 'label' columns")
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--stem", action="store_true", help="Use stemming instead of lemmatization")
    p.add_argument("--embedding_type", choices=["word2vec", "glove"], default="word2vec")
    p.add_argument("--glove_path", type=str, default=None,
                    help="Path to a pretrained GloVe .txt file (required if embedding_type=glove)")
    p.add_argument("--embedding_dim", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--max_len", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--model_dir", type=str, default="models")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ---- 1. Load & clean dataset ----
    print(f"Loading dataset from {args.data} ...")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)
    df[args.label_col] = df[args.label_col].str.lower().str.strip()
    df = df[df[args.label_col].isin(LABEL2IDX.keys())].reset_index(drop=True)
    df["clean_text"] = df[args.text_col].apply(clean_text)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    cleaned_data_path = os.path.join(args.output_dir, "cleaned_dataset.csv")
    df.to_csv(cleaned_data_path, index=False, encoding="utf-8")
    print(f"Cleaned dataset saved to {cleaned_data_path} ({len(df)} rows)")
    print(df[args.label_col].value_counts())

    # ---- 2. Preprocess (tokenize, remove stopwords, lemmatize/stem) ----
    print("\nRunning preprocessing pipeline (tokenize -> stopwords -> lemma/stem) ...")
    tokenized = preprocess_corpus(df[args.text_col].tolist(), use_stemming=args.stem)
    df["tokens"] = tokenized
    labels_idx = encode_labels(df[args.label_col].tolist())

    # ---- Train / val / test split (stratified) ----
    from sklearn.model_selection import train_test_split

    idx_all = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=args.test_size, random_state=SEED, stratify=labels_idx
    )
    labels_trainval = [labels_idx[i] for i in idx_trainval]
    val_relative_size = args.val_size / (1 - args.test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_relative_size, random_state=SEED, stratify=labels_trainval
    )

    def subset(indices):
        return df.iloc[indices].reset_index(drop=True)

    train_df, val_df, test_df = subset(idx_train), subset(idx_val), subset(idx_test)
    print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ---- 3. Vocabulary + embeddings ----
    print("\nBuilding vocabulary ...")
    word2idx, idx2word = build_vocab(train_df["tokens"].tolist(), min_freq=1)
    print(f"Vocabulary size: {len(word2idx)}")

    with open(os.path.join(args.model_dir, "vocab.pkl"), "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word}, f)

    if args.embedding_type == "word2vec":
        print("\nTraining Word2Vec embeddings on the training corpus ...")
        w2v_model = train_word2vec(train_df["tokens"].tolist(), vector_size=args.embedding_dim)
        w2v_model.save(os.path.join(args.model_dir, "word2vec.model"))
        embedding_matrix = build_embedding_matrix_from_w2v(w2v_model, word2idx, args.embedding_dim)
    else:
        if not args.glove_path:
            raise ValueError("--glove_path is required when --embedding_type=glove")
        glove_index = load_glove_embeddings(args.glove_path, args.embedding_dim)
        embedding_matrix = build_embedding_matrix_from_glove(glove_index, word2idx, args.embedding_dim)

    np.save(os.path.join(args.model_dir, "embedding_matrix.npy"), embedding_matrix)

    # ---- 4. Baseline model: TF-IDF + Logistic Regression ----
    print("\nTraining baseline (TF-IDF + Logistic Regression) ...")
    baseline = build_baseline_pipeline()
    train_baseline(baseline, train_df["clean_text"].tolist(), [LABEL2IDX[l] for l in train_df[args.label_col]])

    with open(os.path.join(args.model_dir, "baseline_pipeline.pkl"), "wb") as f:
        pickle.dump(baseline, f)

    baseline_test_preds = baseline.predict(test_df["clean_text"].tolist())
    baseline_test_true = [LABEL2IDX[l] for l in test_df[args.label_col]]

    # ---- 5. LSTM model ----
    print("\nBuilding datasets for the LSTM ...")
    train_labels = [LABEL2IDX[l] for l in train_df[args.label_col]]
    val_labels = [LABEL2IDX[l] for l in val_df[args.label_col]]
    test_labels = [LABEL2IDX[l] for l in test_df[args.label_col]]

    train_ds = SentimentDataset(train_df["tokens"].tolist(), train_labels, word2idx, args.max_len)
    val_ds = SentimentDataset(val_df["tokens"].tolist(), val_labels, word2idx, args.max_len)
    test_ds = SentimentDataset(test_df["tokens"].tolist(), test_labels, word2idx, args.max_len)

    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = (class_counts.sum() / (3 * np.maximum(class_counts, 1))).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining LSTM classifier on device: {device}")
    model = LSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=3,
        embedding_matrix=embedding_matrix,
        bidirectional=True,
        dropout=0.4,
    )
    model, history = train_lstm(
        model, train_ds, val_ds,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, class_weights=class_weights,
    )
    torch.save(model.state_dict(), os.path.join(args.model_dir, "lstm_model.pt"))

    # LSTM test predictions
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    lstm_preds, lstm_true = [], []
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x = x.to(device)
            logits = model(x, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            lstm_preds.extend(preds.tolist())
            lstm_true.extend(y.numpy().tolist())

    # ---- 6. Evaluation ----
    label_names = [IDX2LABEL[i] for i in range(3)]

    baseline_metrics = evaluate_predictions(baseline_test_true, baseline_test_preds, label_names)
    lstm_metrics = evaluate_predictions(lstm_true, lstm_preds, label_names)

    save_classification_report(
        baseline_metrics["classification_report"],
        os.path.join(args.output_dir, "classification_report_baseline.txt"),
    )
    save_classification_report(
        lstm_metrics["classification_report"],
        os.path.join(args.output_dir, "classification_report_lstm.txt"),
    )

    plot_confusion_matrix(
        baseline_metrics["confusion_matrix"], label_names, "Baseline (TF-IDF + LogReg)",
        os.path.join(args.output_dir, "confusion_matrix_baseline.png"),
    )
    plot_confusion_matrix(
        lstm_metrics["confusion_matrix"], label_names, "LSTM + Word Embeddings",
        os.path.join(args.output_dir, "confusion_matrix_lstm.png"),
    )

    compare_models(
        {"baseline_tfidf_logreg": baseline_metrics, "lstm_embeddings": lstm_metrics},
        os.path.join(args.output_dir, "model_comparison.csv"),
    )

    # ---- 7. Error analysis (on the LSTM model) ----
    print("\nRunning error analysis on LSTM predictions ...")
    lstm_pred_names = [IDX2LABEL[i] for i in lstm_preds]
    lstm_true_names = [IDX2LABEL[i] for i in lstm_true]
    err_df = build_error_dataframe(
        test_df[args.text_col].tolist(), lstm_true_names, lstm_pred_names, test_df["tokens"].tolist()
    )
    write_error_report(
        err_df,
        os.path.join(args.output_dir, "error_analysis.txt"),
        os.path.join(args.output_dir, "misclassified_examples.csv"),
    )

    print("\nPipeline complete. See the 'results/' and 'models/' directories for all outputs.")


if __name__ == "__main__":
    main()
