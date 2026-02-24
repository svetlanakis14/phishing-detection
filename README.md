# Phishing Detection with Deep Learning

## Project Description
Phishing emails represent one of the most common entry points for cyberattacks. This project implements and compares two deep learning approaches for automated phishing detection, then evaluates their robustness against adversarial text modifications — small, deliberate changes to email content designed to fool the classifier while preserving the email's meaning for a human reader.

The pipeline covers three stages:
1. **Basic training and evaluation** - how well the models perform on clean data
2. **Adversarial attack evaluation** - how much performance drops when models are 'attacked' (input data is modified)
3. **Adversarial retraining** - adding attacked examples to training set and evaluating performance again

## Dataset
[Kaggle — Phishing Emails Dataset](https://www.kaggle.com/datasets/subhajournal/phishingemails)
- ~18,650 emails total
- Labels: `Safe Email` (11,322) and `Phishing Email` (7,328)
- 16 emails with missing text were dropped during preprocessing
- Final dataset after cleaning: 18,634 examples

The dataset is not included in this repository. Download it from Kaggle and place `Phishing_Email.csv` in the `data/` folder.

## Project structure 
```
phishing-detection/
├── data/
│   └── Phishing_Email.csv        # not included, download from Kaggle
├── src/
│   ├── preprocessing.py          # cleaning, TF-IDF, tokenization, split
│   ├── model_tfidf.py            # feedforward neural network definition
│   ├── model_lstm.py             # LSTM model definition
│   ├── train.py                  # full training pipeline
│   ├── evaluation.py             # metrics, plots, confusion matrices
|   └── adversarial_attacks.py    # methods for modifying inputs (email text)
├── notebooks/
│   └── analysis.ipynb            # exploratory analysis and result visualization
├── results/
│   ├── models/                   # saved model files and vectorizers
│   ├── *_history.png             # training curves
│   ├── *_confusion_matrix.png    # confusion matrices
│   ├── *_metrics.json            # saved metrics per model
│   └── all_models_comparison.png     # bar chart comparing all models
└── README.md
```

## Models
### Model 1 — TF-IDF + Feedforward Neural Network
Text is represented as a TF-IDF vector (10,000 features, unigrams and bigrams). The vector is passed through a feedforward network:
```
Input (10,000)
→ Dense(128, ReLU) → BatchNormalization → Dropout(0.5)
→ Dense(64, ReLU)  → BatchNormalization → Dropout(0.5)
→ Dense(1, Sigmoid)
```
Total parameters: ~1,289,000

### Model 2 — Embedding + Bidirectional LSTM
Text is tokenized into integer sequences (vocabulary size 10,000, max length 200 tokens) and passed through a sequential model:
```
Input (200 tokens)
→ Embedding(10000, 128)
→ Bidirectional LSTM(64 units)
→ Dropout(0.5)
→ Dense(32, ReLU)
→ Dense(1, Sigmoid)
```
Total parameters: ~1,383,000
- Both models use Binary Cross-Entropy loss and the Adam optimizer. Training uses Early Stopping to prevent overfitting.

## Results
### Baseline Performance
Both models were trained on 80% of the data and evaluated on the remaining 20% (3,727 emails).

| Model  | Accuracy | Precision | Recall | F1     |
|--------|----------|-----------|--------|--------|
| TF-IDF | 0.9624   | 0.9715    | 0.9316 | 0.9511 |
| LSTM   | 0.9678   | 0.9409    | 0.9795 | 0.9598 |

LSTM achieves a slightly higher F1 score. Also, LSTM has higher recall (0.98 vs 0.93), meaning it misses fewer phishing emails — which is the more important metric in a security context.

- Training curves (loss and accuracy per epoch) are saved in `results/` as `tfidf_loss.png` and `lstm_loss.png`.
- Confusion matrices for both models before attack are saved as `tfidf_confusion_matrix.png` and `lstm_confusion_matrix.png`.

---

### Adversarial Attacks
Three types of adversarial text modifications were applied to the test set:

| Attack | Description | Example |
|--------|-------------|---------|
| `char_swap` | Replaces characters with visually similar symbols | `click` → `c1ick`, `verify` → `ver!fy` |
| `synonym` | Replaces phishing keywords with synonyms | `urgent` → `immediate`, `verify` → `confirm` |
| `whitespace` | Inserts spaces inside trigger words | `verify` → `v e r i f y` |

**Performance after each attack:**

| Model  | Original | char_swap | synonym | whitespace |
|--------|----------|-----------|---------|------------|
| TF-IDF | 0.9624   | 0.9555    | 0.9616  | 0.9622     |
| LSTM   | 0.9678   | 0.9174    | 0.9683  | 0.9681     |

**Key findings:**

- `char_swap` was the only attack that had a meaningful effect, and only on LSTM (−5.0% accuracy drop)
- `synonym` and `whitespace` had negligible impact on both models
- Counterintuitively, **TF-IDF proved more robust to attacks than LSTM** — because TF-IDF ignores word order and context, character-level modifications affect it less. LSTM, which reads sequences token by token, treats `c1ick` as a completely unknown word, losing contextual information
- LSTM was essentially unaffected by synonym and whitespace attacks, suggesting these modifications were not aggressive enough for a sequence model that captures broader context
- Confusion matrices for each attack scenario are saved in `results/` (e.g. `tfidf_char_swap_confusion_matrix.png`).

---

### After Retraining
Attacked versions of the test emails were added to the training set (augmenting it from ~14,900 to ~59,600 examples across all three attack types), and both models were retrained from scratch.

| Model  | Before retraining | After retraining | Improvement |
|--------|-------------------|------------------|-------------|
| TF-IDF | 0.9624            | 0.9887           | +0.0263     |
| LSTM   | 0.9678            | 0.9885           | +0.0207     |

Both models reached ~99% accuracy after retraining, with recall of 0.9993 — meaning virtually every phishing email in the test set was correctly identified.
This confirms the core hypothesis: **exposing models to adversarial examples during training significantly improves robustness and overall performance.**

- Confusion matrices after retraining are saved as `tfidf_retrain_confusion_matrix.png` and `lstm_retrain_confusion_matrix.png`.
- A bar chart comparing all models across all stages is saved as `results/all_models_comparison.png`.

---

## How to Run
**1. Install dependencies**
- pip install -r requirements.txt
  
**2. Place the dataset**
- Download `Phishing_Email.csv` from Kaggle and place it in the `data/` folder.
  
**3. Run the full pipeline**
- python src/train.py
- This will run preprocessing, train both models, apply adversarial attacks, retrain, and save all results to `results/`.
  
**4. Explore results interactively**
- Open `notebooks/analysis.ipynb` to visualize training curves, confusion matrices, and metric comparisons side by side.
---
