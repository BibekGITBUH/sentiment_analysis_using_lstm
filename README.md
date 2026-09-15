# Sentiment Analysis System

Classifies text (reviews / social media posts) into **positive**, **negative**,
or **neutral**, comparing a classic TF-IDF + Logistic Regression baseline
against an LSTM model trained on word embeddings (self-trained Word2Vec by
default, or pretrained GloVe if you supply a vector file).

## Project layout

```
sentiment_analysis/
├── data/
│   ├── generate_sample_data.py   # creates a synthetic labeled dataset (offline, no download needed)
│   └── sample_reviews.csv        # generated dataset (created by the script above)
├── src/
│   ├── preprocessing.py          # tokenization, stopword removal, lemmatization/stemming
│   ├── embeddings.py             # Word2Vec training + GloVe loading + embedding matrix
│   ├── dataset.py                # vocab building, sequence encoding, PyTorch Dataset
│   ├── models.py                 # LSTM classifier + TF-IDF/LogReg baseline
│   ├── train.py                  # training loops for both models
│   ├── evaluate.py               # accuracy / precision / recall / F1 / confusion matrix
│   └── error_analysis.py         # misclassification breakdown & qualitative analysis
├── main.py                       # end-to-end pipeline (run this)
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

# RECOMMENDED: build a real, diverse dataset (~8500 rows) from:
#   - nltk 'sentence_polarity' (5331 pos / 5331 neg short review-style
#     movie-critic snippets - closer register to product/service reviews
#     than social media slang)
#   - nltk 'reuters' (factual/neutral news, for the neutral class)
#   - data/handwritten_sentences.py (180 sentences composed by hand,
#     covering plain everyday English - "I love this product.",
#     "It is raining outside today." - oversampled 6x so this register
#     isn't drowned out by the larger corpora)
# No external downloads or accounts needed.
python -m nltk.downloader sentence_polarity reuters
python data/build_real_dataset.py
python main.py --data data/real_reviews.csv --epochs 20

# Alternative: a small synthetic template dataset, useful only as a fast
# smoke test - it does NOT generalize to free-form text (see note below)
python data/generate_sample_data.py
python main.py --data data/sample_reviews.csv --epochs 15 --embedding_type word2vec
```

To use pretrained GloVe vectors instead of self-trained Word2Vec:

```bash
python main.py --data data/sample_reviews.csv --embedding_type glove --glove_path /path/to/glove.6B.100d.txt
```

To use your own dataset, point `--data` at any CSV with a `text` column and a
`label` column containing `positive` / `negative` / `neutral`.

## Classifying a new sentence

Once `main.py` has been run at least once (so `models/` has saved weights),
use `predict.py` to classify new text:

```bash
# single sentence with the LSTM model (default)
python predict.py --text "I absolutely loved this product!"

# single sentence with the baseline model instead
python predict.py --text "The food was okay I guess." --model baseline

# interactive mode: type sentences one at a time, 'quit' to exit
python predict.py --interactive
```

Output looks like:

```
Text: I absolutely loved this product!
Predicted sentiment: POSITIVE
Probabilities: {'negative': 0.02, 'neutral': 0.01, 'positive': 0.97}
```

If you trained with non-default `--embedding_dim`, `--hidden_dim`, `--max_len`,
or `--stem`, pass the same values to `predict.py` so the loaded model matches
its saved weights.

## Outputs

Running `main.py` writes to `results/`:

* `results/model_comparison.csv` — accuracy/precision/recall/F1 for baseline vs LSTM
* `results/confusion_matrix_baseline.png`, `results/confusion_matrix_lstm.png`
* `results/classification_report_baseline.txt`, `results/classification_report_lstm.txt`
* `results/error_analysis.txt` — misclassified examples & error patterns
* `results/misclassified_examples.csv`

Trained artifacts are written to `models/`:

* `models/word2vec.model` (or embedding matrix if GloVe was used)
* `models/lstm_model.pt`
* `models/vocab.pkl`
* `models/baseline_pipeline.pkl`

## Notes

* **`data/generate_sample_data.py`** produces a tiny, template-based dataset
  (~600 sentences from 15 templates). It's only useful for a fast smoke test
  of the pipeline's plumbing. Because its vocabulary is only ~200 words, an
  LSTM trained on it will treat most everyday words ("love", "amazing",
  "delicious"...) as unknown/OOV and can collapse to near-constant
  predictions on free-form input. **Do not use it to judge real model
  quality.**
* **`data/build_real_dataset.py`** (recommended) builds a real, diverse
  ~4500-row dataset from NLTK's bundled `twitter_samples` (real
  positive/negative tweets) and `reuters` (factual/neutral news) corpora —
  no external downloads, Kaggle account, or API key required. This is the
  dataset to use for anything beyond a smoke test.
* For production-quality results, swap in a larger domain-matched dataset
  (IMDB reviews, Amazon reviews, Sentiment140, etc.) — same `text,label` CSV
  schema, everything else in the pipeline stays the same.
* **Honest limitation:** with ~7,500 training rows, the LSTM tends to
  overfit (train accuracy near 99% while validation plateaus around 78-80%)
  and generalizes worse to short, plain sentences than the TF-IDF +
  Logistic Regression baseline does. This is a normal data-scale limitation
  of training embeddings + an LSTM from scratch, not a bug. If you need a
  model to actually deploy today, `predict.py --model baseline` is the more
  reliable choice at this data size. To make the LSTM itself competitive,
  use pretrained GloVe vectors (see below) and/or substantially more
  labeled data.
* Preprocessing supports both **lemmatization** (default) and **stemming**
  (`--stem` flag) as required by the assignment brief.
* Embeddings: `--embedding_type word2vec` (default) trains a Word2Vec model
  on the corpus with gensim; `--embedding_type glove` loads a pretrained
  GloVe file you provide (not bundled, since it's several hundred MB) — this
  generally improves generalization to words rarely seen in your training
  set, since GloVe vectors are pretrained on billions of words of text.




#
# Images of the Note — To Whom It May Concern

![alt text](<WhatsApp Image 2026-09-15 at 9.37.10 PM.jpeg>) ![alt text](<WhatsApp Image 2026-09-15 at 9.37.10 PM (4).jpeg>) ![alt text](<WhatsApp Image 2026-09-15 at 9.37.26 PM (2).jpeg>) ![alt text](<WhatsApp Image 2026-09-15 at 9.37.26 PM (3).jpeg>)


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
```

D:\Projects_on_Python\sentiment_analysis_system\sentiment_analysis_system (1)\sentiment_analysis>python -m nltk.downloader sentence_polarity reuters
<frozen runpy>:128: RuntimeWarning: 'nltk.downloader' found in sys.modules after import of package 'nltk', but prior to execution of 'nltk.downloader'; this may result in unpredictable behaviour
[nltk_data] Downloading package sentence_polarity to
[nltk_data]     C:\Users\BIBEK\AppData\Roaming\nltk_data...
[nltk_data]   Unzipping corpora\sentence_polarity.zip.
[nltk_data] Downloading package reuters to
[nltk_data]     C:\Users\BIBEK\AppData\Roaming\nltk_data...
[nltk_data]   Package reuters is already up-to-date!

D:\Projects_on_Python\sentiment_analysis_system\sentiment_analysis_system (1)\sentiment_analysis>python data/build_real_dataset.py
Wrote 8580 rows to D:\Projects_on_Python\sentiment_analysis_system\sentiment_analysis_system (1)\sentiment_analysis\data\real_reviews.csv
Class balance: {'negative': 2860, 'positive': 2860, 'neutral': 2860}

D:\Projects_on_Python\sentiment_analysis_system\sentiment_analysis_system (1)\sentiment_analysis>python main.py --data data/real_reviews.csv --epochs 20
Loading dataset from data/real_reviews.csv ...
Cleaned dataset saved to results\cleaned_dataset.csv (7655 rows)
label
negative    2560
positive    2560
neutral     2535
Name: count, dtype: int64

Running preprocessing pipeline (tokenize -> stopwords -> lemma/stem) ...
Preprocessing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7655/7655 [00:02<00:00, 2653.34it/s]

Split sizes -> train: 5357, val: 1149, test: 1149

Building vocabulary ...
Vocabulary size: 12118

Training Word2Vec embeddings on the training corpus ...
Word2Vec coverage: 12116/12118 tokens (100.0%)

Training baseline (TF-IDF + Logistic Regression) ...

Building datasets for the LSTM ...

Training LSTM classifier on device: cuda
Training LSTM:   0%|                                                                                                                                                      | 0/20 [00:00<?, ?it/s]Epoch 01 | train_loss=0.5626 train_acc=0.6588 | val_loss=0.5080 val_acc=0.6536
Training LSTM:   5%|███████                                                                                                                                       | 1/20 [00:02<00:41,  2.17s/it]Epoch 02 | train_loss=0.4739 train_acc=0.7368 | val_loss=0.4487 val_acc=0.7676
Training LSTM:  10%|██████████████▏                                                                                                                               | 2/20 [00:03<00:33,  1.83s/it]Epoch 03 | train_loss=0.3135 train_acc=0.8658 | val_loss=0.4388 val_acc=0.7903
Training LSTM:  15%|█████████████████████▎                                                                                                                        | 3/20 [00:05<00:29,  1.72s/it]Epoch 04 | train_loss=0.1405 train_acc=0.9524 | val_loss=0.5932 val_acc=0.7807
Training LSTM:  20%|████████████████████████████▍                                                                                                                 | 4/20 [00:06<00:26,  1.67s/it]Epoch 05 | train_loss=0.0636 train_acc=0.9834 | val_loss=0.6004 val_acc=0.7789
Training LSTM:  25%|███████████████████████████████████▌                                                                                                          | 5/20 [00:08<00:24,  1.61s/it]Epoch 06 | train_loss=0.0342 train_acc=0.9912 | val_loss=0.7393 val_acc=0.7946
Training LSTM:  30%|██████████████████████████████████████████▌                                                                                                   | 6/20 [00:09<00:22,  1.58s/it]Epoch 07 | train_loss=0.0229 train_acc=0.9944 | val_loss=0.7449 val_acc=0.7711
Early stopping at epoch 7 (no val improvement for 4 epochs).
Training LSTM:  30%|██████████████████████████████████████████▌                                                                                                   | 6/20 [00:11<00:26,  1.91s/it]

=== Model comparison (sorted by macro F1) ===
                model  accuracy  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted
baseline_tfidf_logreg  0.815492         0.818328      0.815856  0.816939            0.817896         0.815492     0.816541
      lstm_embeddings  0.794604         0.798750      0.795009  0.796286            0.798267         0.794604     0.795841

Running error analysis on LSTM predictions ...
======================================================================
ERROR ANALYSIS REPORT
======================================================================
Total examples: 1149
Correct: 913  |  Incorrect: 236
Overall accuracy: 0.7946

-- Confusion pairs (true -> predicted), most frequent first --
   negative -> positive  : 118 errors
   positive -> negative  : 93 errors
    neutral -> negative  : 10 errors
    neutral -> positive  : 9 errors
   negative -> neutral   : 3 errors
   positive -> neutral   : 3 errors

-- Token-length comparison: correct vs incorrect predictions --
              mean  median       std  count
correct                                    
False    10.851695    10.0  5.427985    236
True     10.667032    10.0  4.622217    913

Interpretation: if the mean length for incorrect predictions is
notably higher/lower than for correct ones, sentence length is
likely correlated with model errors (e.g. very short or very long
reviews may lack/contain enough signal for the LSTM's fixed context).

-- Sample misclassified examples (up to 5 per true/pred pair) --
  [true=negative, pred=neutral] all i can say is fuhgeddaboutit.
  [true=negative, pred=neutral] they should have called it gutterball.
  [true=negative, pred=neutral] do you say " hi " to your lover when you wake up in the morning?
  [true=negative, pred=positive] it's at once laughable and compulsively watchable, in its committed dumbness.
  [true=negative, pred=positive] gaghan captures the half-lit, sometimes creepy intimacy of college dorm rooms, a subtlety that makes the silly, over-the-top coda especially disappointing.
  [true=negative, pred=positive] there's only one way to kill michael myers for good: stop buying tickets to these movies.
  [true=negative, pred=positive] may offend viewers not amused by the sick sense of humor.
  [true=negative, pred=positive] a cumbersome and cliche-ridden movie greased with every emotional device known to man.
  [true=neutral, pred=negative] " But Tenneco would be a challenge to run because of its sheer size and diversity.
  [true=neutral, pred=negative] As far as it knows, the ingredient has not been tested in a spermicide.
  [true=neutral, pred=negative] An attorney for Ammeen, who is named as a defendant, asked the court to dismiss the lawsuit against his client.
  [true=neutral, pred=negative] The bus stop is around the corner.
  [true=neutral, pred=negative] Chase Manhattan has branches in Milan and Rome.
  [true=neutral, pred=positive] " We appreciate his management, but certainly some practices have to be corrected."
  [true=neutral, pred=positive] The unit delivers mainly to the car industry.
  [true=neutral, pred=positive] The Bass family holds an 11.
  [true=neutral, pred=positive] The SEC has a standing policy of never confirming or denying investigations or upcoming legal action.
  [true=neutral, pred=positive] The bid is conditional upon an examination by Gordon of the business and affairs of Pagecorp during the 45 days ending December 3, 1987.
  [true=positive, pred=negative] the whole cast looks to be having so much fun with the slapstick antics and silly street patois, tossing around obscure expressions like bellini and mullinski, 
  [true=positive, pred=negative] asia authors herself as anna battista, an italian superstar and aspiring directress who just happens to be her own worst enemy.
  [true=positive, pred=negative] neil burger here succeeded in... making the mystery of four decades back the springboard for a more immediate mystery in the present.
  [true=positive, pred=negative] on the surface a silly comedy, scotland, pa would be forgettable if it weren't such a clever adaptation of the bard's tragic play.
  [true=positive, pred=negative] may lack the pungent bite of its title, but it's an enjoyable trifle nonetheless.
  [true=positive, pred=neutral] I love this product.
  [true=positive, pred=neutral] This is a lovely house.
  [true=positive, pred=neutral] trades run-of-the-mill revulsion for extreme unease.

-- Common error themes to check manually --
  * Neutral vs positive/negative confusion: neutral is the hardest
    class since it lacks strong polarity words; check if the model
    is biased toward predicting positive/negative for neutral text.
  * Negation handling: look for misclassified examples containing
    'not', 'never', "n't" - LSTMs need enough training signal to
    learn that negation flips polarity.
  * Sarcasm / mixed sentiment: sentences with both positive and
    negative cues are a common source of error for both models.
  * OOV / rare words: words absent from the embedding vocabulary
    fall back to the <UNK> vector and lose sentiment signal.

Pipeline complete. See the 'results/' and 'models/' directories for all outputs.```