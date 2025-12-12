# 📋 Complete Pipeline Guide — Social Media Classification

## 🎯 **Quick Overview: How Your Model Works**

```
Raw CSV (comments) 
    ↓
[1] PREPROCESSING: Clean text (remove URLs, normalize Arabizi, etc.)
    ↓
[2] FEATURE EXTRACTION: Convert text → numbers (TF-IDF or Transformer embeddings)
    ↓
[3] TRAINING: Learn patterns (Logistic Regression / SVM / Transformer)
    ↓
[4] EVALUATION: Cross-validation (5-fold) → OOF F1-score
    ↓
[5] HYPERPARAMETER TUNING: Search best configs (C, ngrams, etc.)
    ↓
[6] ENSEMBLE: Combine top models (average probabilities)
    ↓
[7] BLENDING: Mix TF-IDF + Transformer probabilities
    ↓
Final Submission CSV
```

---

## 📝 **Step-by-Step Code**

### **STEP 1: PREPROCESSING**

**What it does:**
- Cleans text (removes URLs, normalizes phone numbers, handles Arabizi)
- Creates multi-view features (Arabic-only, Latin-only)
- Adds metadata (text length, punctuation counts)

**Main Code:**
```python
from forca_hack.src.social.preprocess import preprocess_social_df
from forca_hack.src.social.text_cleaning import TextNormConfig

# Load raw data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Configure text normalization
norm_cfg = TextNormConfig(
    keep_url_token=True,        # Replace URLs with <URL>
    keep_mention_token=True,    # Replace @user with <MENTION>
    mask_phone_numbers=True,    # Replace phones with <PHONE>
    reduce_elongations=True,    # "sooo" → "so"
    lowercase_latin=True,       # "Hello" → "hello"
    keep_digits=True,           # Keep numbers
    arabizi_map_digits=True,    # "3lach" → "علاش" (context-aware)
)

# Apply preprocessing
train_df = preprocess_social_df(train_df, norm_cfg=norm_cfg)
test_df = preprocess_social_df(test_df, norm_cfg=norm_cfg)

# New columns created:
# - text_clean: cleaned text
# - text_ar: Arabic-only text
# - text_lat: Latin-only text
# - has_ar, has_lat: binary flags
# - n_exclam, n_question, n_digits: counts
```

**File:** `forca-hack/src/social/preprocess.py`

---

### **STEP 2: FEATURE EXTRACTION**

**What it does:**
- Converts text → numerical vectors
- **TF-IDF**: Character ngrams (3-6) + Word ngrams (1-2)
- **Transformer**: Token embeddings from XLM-RoBERTa

**Main Code (TF-IDF):**
```python
from forca_hack.src.social.preprocess import build_text_feature

# Build combined text feature (main text + metadata)
X_train = build_text_feature(train_df)  # Returns Series[str]
X_test = build_text_feature(test_df)

# Inside build_text_feature:
# - Combines: text_clean + " " + text_ar + " " + text_lat
# - Adds metadata as tokens: "LEN_50", "HAS_AR", etc.
```

**Main Code (Transformer):**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
X_train = build_text_feature(train_df).astype(str).tolist()
X_test = build_text_feature(test_df).astype(str).tolist()

# Tokenize
train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=192,
    return_tensors="pt"
)
```

**Files:**
- TF-IDF: `forca-hack/src/social/preprocess.py` → `build_text_feature()`
- Transformer: `forca-hack/scripts/train_social_transformer.py`

---

### **STEP 3: TRAINING**

**What it does:**
- Trains a classifier on features
- Uses 5-fold cross-validation (StratifiedKFold)
- Saves OOF (out-of-fold) predictions for evaluation

**Main Code (TF-IDF + Logistic Regression):**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Build model pipeline
vectorizer = TfidfVectorizer(
    ngram_range=(3, 6),      # Character ngrams
    max_features=200_000,
    min_df=2
)
clf = LogisticRegression(C=6.0, max_iter=2000, class_weight="balanced")
model = Pipeline([("tfidf", vectorizer), ("clf", clf)])

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros_like(y_train)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    oof_pred[va_idx] = model.predict(X_train.iloc[va_idx])

# Evaluate
oof_f1 = f1_score(y_train, oof_pred, average="macro")
print(f"OOF macro F1: {oof_f1:.5f}")
```

**Main Code (Transformer):**
```python
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=9
)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    fp16=True,  # Mixed precision (faster on GPU)
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

**Files:**
- TF-IDF: `forca-hack/scripts/train_social_tfidf.py`
- Transformer: `forca-hack/scripts/train_social_transformer.py`

---

### **STEP 4: EVALUATION**

**What it does:**
- Computes macro F1-score on OOF predictions
- Reports per-class F1 and confusion matrix

**Main Code:**
```python
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Overall macro F1
macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"Macro F1: {macro_f1:.5f}")

# Per-class F1
per_class_f1 = {}
for label in np.unique(y_true):
    per_class_f1[label] = f1_score(
        (y_true == label).astype(int),
        (y_pred == label).astype(int)
    )

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
```

**File:** Both training scripts compute this automatically.

---

### **STEP 5: HYPERPARAMETER TUNING**

**What it does:**
- Searches best configs (C, ngram ranges, min_df, etc.)
- Tests multiple classifiers (LR, SVM, SGD)
- Keeps top-K models for ensemble

**Main Code:**
```python
# Search space
configs = [
    {"C": 4.0, "char_ngram": (3, 6), "min_df": 2, "clf": "lr"},
    {"C": 6.0, "char_ngram": (3, 7), "min_df": 1, "clf": "lr"},
    {"C": 8.0, "char_ngram": (3, 6), "min_df": 2, "clf": "svm"},
    # ... more configs
]

scores = []
for cfg in configs:
    model = make_model(**cfg)
    oof_f1 = run_cv(model, X_train, y_train)
    scores.append((oof_f1, cfg))

# Sort by score
scores.sort(key=lambda x: x[0], reverse=True)

# Keep top 8
top_k = 8
best_configs = [cfg for _, cfg in scores[:top_k]]
```

**Command:**
```bash
python forca-hack/scripts/train_social_tfidf.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --out_dir outputs \
  --search --search_mode medium \
  --include_svm \
  --ensemble_top_k 8
```

**File:** `forca-hack/scripts/train_social_tfidf.py` (lines 325-443)

---

### **STEP 6: ENSEMBLE**

**What it does:**
- Averages probabilities from top-K models
- Predicts final class = argmax(averaged probabilities)

**Main Code:**
```python
# Train top-K models, get test probabilities
test_proba_sum = np.zeros((len(X_test), num_classes))

for cfg in best_configs:
    model = make_model(**cfg)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    test_proba_sum += proba

# Average
test_proba_avg = test_proba_sum / len(best_configs)
test_pred = np.argmax(test_proba_avg, axis=1)
```

**File:** `forca-hack/scripts/train_social_tfidf.py` (lines 444-458)

---

### **STEP 7: BLENDING (TF-IDF + Transformer)**

**What it does:**
- Combines TF-IDF probabilities + Transformer probabilities
- Finds best weights (e.g., 0.6 TF-IDF + 0.4 Transformer)
- Uses OOF predictions to tune weights

**Main Code:**
```python
# Load probabilities
tfidf_proba = np.load("tfidf_proba.npy")      # Shape: (n_test, 9)
transformer_proba = np.load("transformer_proba.npy")  # Shape: (n_test, 9)

# Try different weights
best_f1 = 0
best_w = 0.5

for w_tfidf in np.arange(0.0, 1.01, 0.05):
    w_trans = 1.0 - w_tfidf
    blended = w_tfidf * tfidf_proba + w_trans * transformer_proba
    pred = np.argmax(blended, axis=1)
    f1 = f1_score(y_true, pred, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_w = w_tfidf

# Apply best weights to test
final_proba = best_w * tfidf_test_proba + (1 - best_w) * transformer_test_proba
final_pred = np.argmax(final_proba, axis=1)
```

**Command:**
```bash
python forca-hack/scripts/tune_blend_social.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --oof_proba outputs/tfidf_oof_proba.npy outputs/transformer_oof_proba.npy \
  --oof_meta outputs/tfidf_oof_meta.json outputs/transformer_oof_meta.json \
  --test_proba outputs/tfidf_test_proba.npy outputs/transformer_test_proba.npy \
  --test_meta outputs/tfidf_test_meta.json outputs/transformer_test_meta.json \
  --out_csv outputs/submission_blend.csv
```

**File:** `forca-hack/scripts/tune_blend_social.py`

---

## 🚀 **Complete Workflow (Copy-Paste Ready)**

### **Phase 1: TF-IDF Training (with probability saving)**

```bash
python forca-hack/scripts/train_social_tfidf.py \
  --train_csv /content/clients-satisfaction/train.csv \
  --test_csv /content/clients-satisfaction/test_file.csv \
  --out_dir forca-hack/outputs \
  --n_splits 5 \
  --seed 42 \
  --search --search_mode medium \
  --include_svm \
  --ensemble_top_k 8 \
  --predict_strategy fold_ensemble \
  --save_test_proba \
  --save_oof_proba \
  --save_artifacts
```

**Outputs:**
- `submission_social_best_top8.csv` (ready to submit)
- `submission_social_best_top8_proba.npy` (test probabilities)
- `submission_social_best_top8_oof_proba.npy` (OOF probabilities)
- `submission_social_best_top8_proba_meta.json` (metadata)

---

### **Phase 2: Transformer Training (GPU required)**

```bash
# First: Enable GPU in Colab (Runtime → Change runtime type → GPU)

python forca-hack/scripts/train_social_transformer.py \
  --train_csv /content/clients-satisfaction/train.csv \
  --test_csv /content/clients-satisfaction/test_file.csv \
  --out_dir forca-hack/outputs \
  --model_name xlm-roberta-base \
  --n_splits 5 \
  --fold_ensemble \
  --fp16 \
  --lr 2e-5 \
  --epochs 3 \
  --max_len 192 \
  --save_test_proba \
  --save_oof_proba
```

**Outputs:**
- `submission_social_transformer.csv` (ready to submit)
- `submission_social_transformer_proba.npy`
- `submission_social_transformer_oof_proba.npy`
- `submission_social_transformer_proba_meta.json`

---

### **Phase 3: Auto-Blend (after both finish)**

```bash
python forca-hack/scripts/tune_blend_social.py \
  --train_csv /content/clients-satisfaction/train.csv \
  --test_csv /content/clients-satisfaction/test_file.csv \
  --oof_proba \
    forca-hack/outputs/submission_social_best_top8_oof_proba.npy \
    forca-hack/outputs/submission_social_transformer_oof_proba.npy \
  --oof_meta \
    forca-hack/outputs/submission_social_best_top8_oof_proba_meta.json \
    forca-hack/outputs/submission_social_transformer_oof_proba_meta.json \
  --test_proba \
    forca-hack/outputs/submission_social_best_top8_proba.npy \
    forca-hack/outputs/submission_social_transformer_proba.npy \
  --test_meta \
    forca-hack/outputs/submission_social_best_top8_proba_meta.json \
    forca-hack/outputs/submission_social_transformer_proba_meta.json \
  --step 0.05 \
  --out_csv forca-hack/outputs/submission_social_blend_auto.csv \
  --out_weights_json forca-hack/outputs/blend_weights.json
```

**Output:**
- `submission_social_blend_auto.csv` (best submission, ready for Kaggle)

---

## 📊 **Expected Results**

- **TF-IDF alone:** ~0.54-0.56 macro F1
- **Transformer alone:** ~0.55-0.58 macro F1 (if GPU-trained properly)
- **Blended (TF-IDF + Transformer):** ~0.57-0.59 macro F1

---

## ⚠️ **Common Issues**

1. **"FileNotFoundError" for CSV:**
   - Check your file paths (use absolute paths in Colab)
   - Mount Google Drive if files are there: `from google.colab import drive; drive.mount('/content/drive')`

2. **"No OOF probability files":**
   - Run training scripts with `--save_oof_proba` flag
   - Files are created during training, not before

3. **Transformer running on CPU:**
   - Enable GPU: Runtime → Change runtime type → GPU
   - Restart runtime after changing
   - Verify: `import torch; print(torch.cuda.is_available())`

4. **Low F1-score:**
   - Make sure Transformer is actually using GPU (check logs for `device=cuda`)
   - Try different blend weights manually
   - Increase `--ensemble_top_k` (e.g., 8 → 12)

---

## 📁 **File Structure**

```
forca-hack/
├── src/social/
│   ├── preprocess.py          # Step 1: Preprocessing
│   └── text_cleaning.py        # Text normalization
├── scripts/
│   ├── train_social_tfidf.py  # Steps 2-6: TF-IDF training
│   ├── train_social_transformer.py  # Steps 2-6: Transformer training
│   └── tune_blend_social.py   # Step 7: Blending
└── outputs/
    ├── submission_*.csv        # Final submissions
    ├── *_proba.npy            # Test probabilities
    └── *_oof_proba.npy         # OOF probabilities
```

---

## 🎯 **Quick Reference: What Each Script Does**

| Script | Purpose | Key Flags |
|--------|---------|-----------|
| `train_social_tfidf.py` | Train TF-IDF + LR/SVM models | `--search`, `--ensemble_top_k`, `--save_oof_proba` |
| `train_social_transformer.py` | Train XLM-RoBERTa | `--fold_ensemble`, `--fp16`, `--save_oof_proba` |
| `tune_blend_social.py` | Find best blend weights | `--oof_proba`, `--test_proba`, `--step` |

---

**That's it!** Run Phase 1 → Phase 2 → Phase 3 in order, and you'll get your best submission. 🚀

