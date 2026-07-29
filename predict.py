"""
Run predictions on new, arbitrary sentences using the trained models
saved by main.py (in the models/ directory).

Usage:
    # single sentence, LSTM model (default)
    python predict.py --text "I absolutely loved this product!"

    # single sentence, baseline model
    python predict.py --text "The food was okay I guess." --model baseline

    # interactive mode - type sentences one at a time, Ctrl+C / 'quit' to exit
    python predict.py --interactive

Requires that you already ran `python main.py ...` at least once, so that
models/vocab.pkl, models/lstm_model.pt (or models/baseline_pipeline.pkl)
exist.
"""
import argparse
import os
import pickle

import torch

from src.preprocessing import preprocess_text, clean_text
from src.dataset import encode_tokens, IDX2LABEL
from src.models import LSTMClassifier


def load_lstm(model_dir, embedding_dim, hidden_dim):
    with open(os.path.join(model_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    word2idx = vocab["word2idx"]

    model = LSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=3,
        embedding_matrix=None,  # weights are overwritten by load_state_dict below
        bidirectional=True,
        dropout=0.4,
    )
    state_path = os.path.join(model_dir, "lstm_model.pt")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    return model, word2idx


def load_baseline(model_dir):
    with open(os.path.join(model_dir, "baseline_pipeline.pkl"), "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict_lstm(text, model, word2idx, max_len, use_stemming=False):
    tokens = preprocess_text(text, use_stemming=use_stemming)
    ids, length = encode_tokens(tokens, word2idx, max_len)
    x = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)
    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    label = IDX2LABEL[pred_idx]
    prob_dict = {IDX2LABEL[i]: round(float(probs[i]), 4) for i in range(3)}
    return label, prob_dict


def predict_baseline(text, pipeline):
    cleaned = clean_text(text)
    pred_idx = pipeline.predict([cleaned])[0]
    probs = pipeline.predict_proba([cleaned])[0]
    label = IDX2LABEL[int(pred_idx)]
    prob_dict = {IDX2LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))}
    return label, prob_dict


def parse_args():
    p = argparse.ArgumentParser(description="Predict sentiment for new text")
    p.add_argument("--text", type=str, default=None, help="Sentence to classify")
    p.add_argument("--interactive", action="store_true", help="Enter interactive prompt mode")
    p.add_argument("--model", choices=["lstm", "baseline"], default="lstm")
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--embedding_dim", type=int, default=100, help="Must match training run")
    p.add_argument("--hidden_dim", type=int, default=128, help="Must match training run")
    p.add_argument("--max_len", type=int, default=40, help="Must match training run")
    p.add_argument("--stem", action="store_true", help="Must match training run (--stem flag used or not)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.text and not args.interactive:
        raise SystemExit("Provide --text \"your sentence\" or use --interactive")

    if args.model == "lstm":
        model, word2idx = load_lstm(args.model_dir, args.embedding_dim, args.hidden_dim)
    else:
        pipeline = load_baseline(args.model_dir)

    def run(sentence):
        if args.model == "lstm":
            label, probs = predict_lstm(sentence, model, word2idx, args.max_len, args.stem)
        else:
            label, probs = predict_baseline(sentence, pipeline)
        print(f"\nText: {sentence}")
        print(f"Predicted sentiment: {label.upper()}")
        print(f"Probabilities: {probs}")

    if args.interactive:
        print(f"Interactive mode ({args.model} model). Type a sentence and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                sentence = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if sentence.lower() in {"quit", "exit"}:
                break
            if not sentence:
                continue
            run(sentence)
    else:
        run(args.text)


if __name__ == "__main__":
    main()
