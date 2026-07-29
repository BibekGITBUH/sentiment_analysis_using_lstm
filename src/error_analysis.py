"""
Qualitative + quantitative error analysis: which examples are
misclassified, which label pairs get confused most, and whether errors
correlate with text length or vocabulary novelty (unseen/OOV tokens).
"""
import os

import pandas as pd


def build_error_dataframe(texts, y_true_names, y_pred_names, tokenized_texts=None):
    df = pd.DataFrame({
        "text": texts,
        "true_label": y_true_names,
        "pred_label": y_pred_names,
    })
    if tokenized_texts is not None:
        df["n_tokens"] = [len(t) for t in tokenized_texts]
    df["correct"] = df["true_label"] == df["pred_label"]
    return df


def error_pair_counts(df):
    errors = df[~df["correct"]]
    pair_counts = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return pair_counts


def length_comparison(df):
    if "n_tokens" not in df.columns:
        return None
    return df.groupby("correct")["n_tokens"].agg(["mean", "median", "std", "count"])


def sample_errors(df, n_per_pair=3):
    errors = df[~df["correct"]]
    parts = []
    for _, group in errors.groupby(["true_label", "pred_label"]):
        parts.append(group.head(n_per_pair))
    if not parts:
        return errors.head(0)
    return pd.concat(parts, ignore_index=True)


def write_error_report(df, out_txt_path, out_csv_path, n_per_pair=5):
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)

    errors = df[~df["correct"]]
    pair_counts = error_pair_counts(df)
    length_stats = length_comparison(df)
    samples = sample_errors(df, n_per_pair=n_per_pair)

    lines = []
    lines.append("=" * 70)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total examples: {len(df)}")
    lines.append(f"Correct: {df['correct'].sum()}  |  Incorrect: {len(errors)}")
    lines.append(f"Overall accuracy: {df['correct'].mean():.4f}")
    lines.append("")

    lines.append("-- Confusion pairs (true -> predicted), most frequent first --")
    for _, row in pair_counts.iterrows():
        lines.append(f"  {row['true_label']:>9} -> {row['pred_label']:<9} : {row['count']} errors")
    lines.append("")

    if length_stats is not None:
        lines.append("-- Token-length comparison: correct vs incorrect predictions --")
        lines.append(length_stats.to_string())
        lines.append("")
        lines.append("Interpretation: if the mean length for incorrect predictions is")
        lines.append("notably higher/lower than for correct ones, sentence length is")
        lines.append("likely correlated with model errors (e.g. very short or very long")
        lines.append("reviews may lack/contain enough signal for the LSTM's fixed context).")
        lines.append("")

    lines.append("-- Sample misclassified examples (up to "
                  f"{n_per_pair} per true/pred pair) --")
    for _, row in samples.iterrows():
        text_preview = row["text"][:160].replace("\n", " ")
        lines.append(f"  [true={row['true_label']}, pred={row['pred_label']}] {text_preview}")
    lines.append("")

    lines.append("-- Common error themes to check manually --")
    lines.append("  * Neutral vs positive/negative confusion: neutral is the hardest")
    lines.append("    class since it lacks strong polarity words; check if the model")
    lines.append("    is biased toward predicting positive/negative for neutral text.")
    lines.append("  * Negation handling: look for misclassified examples containing")
    lines.append("    'not', 'never', \"n't\" - LSTMs need enough training signal to")
    lines.append("    learn that negation flips polarity.")
    lines.append("  * Sarcasm / mixed sentiment: sentences with both positive and")
    lines.append("    negative cues are a common source of error for both models.")
    lines.append("  * OOV / rare words: words absent from the embedding vocabulary")
    lines.append("    fall back to the <UNK> vector and lose sentiment signal.")

    report_text = "\n".join(lines)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    errors.to_csv(out_csv_path, index=False, encoding="utf-8")

    print(report_text)
    return report_text
