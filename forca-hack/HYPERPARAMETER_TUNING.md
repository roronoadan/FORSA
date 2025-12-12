# 🔍 Hyperparameter Tuning Guide

## Overview

The training script now supports **two hyperparameter tuning approaches**:

1. **Grid Search** (original) - Exhaustive search over predefined parameter combinations
2. **Bayesian Optimization (Optuna)** - Smart search that learns from previous trials ⭐ **NEW**

---

## 🎯 **Option 1: Grid Search** (Original Approach)

### How it works:
- Tests all combinations from predefined parameter grids
- **Small mode**: 5 configs (fast)
- **Medium mode**: ~80 configs (balanced)
- **Large mode**: ~160 configs (thorough)

### Usage:
```bash
python forca-hack/scripts/train_social_tfidf.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --out_dir outputs \
  --search --search_mode medium \
  --include_svm \
  --ensemble_top_k 8
```

### Pros:
- ✅ Deterministic (same results every time)
- ✅ No extra dependencies
- ✅ Good for small search spaces

### Cons:
- ❌ Slow for large search spaces
- ❌ Wastes time on bad regions
- ❌ Doesn't learn from previous trials

---

## 🚀 **Option 2: Bayesian Optimization (Optuna)** ⭐ **RECOMMENDED**

### How it works:
- Uses **TPE (Tree-structured Parzen Estimator)** to learn from previous trials
- Focuses search on promising parameter regions
- Typically finds better configs in **fewer trials** than grid search

### Installation:
```bash
pip install optuna
```

### Usage:
```bash
python forca-hack/scripts/train_social_tfidf.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --out_dir outputs \
  --search --search_mode optuna \
  --n_trials 50 \
  --include_svm \
  --ensemble_top_k 8
```

### Parameters:
- `--n_trials`: Number of trials to run (default: 50)
  - **30-50 trials**: Fast, good for quick exploration
  - **50-100 trials**: Balanced (recommended)
  - **100+ trials**: Thorough, best results
- `--optuna_timeout`: Maximum time in seconds (optional)
  - Example: `--optuna_timeout 3600` (1 hour)

### Pros:
- ✅ **Much faster** - finds good configs in 30-50 trials vs 80-160 grid search
- ✅ **Smarter** - learns which parameters matter
- ✅ **Better results** - often finds better configs than grid search
- ✅ **Flexible** - can set time limits or trial counts

### Cons:
- ❌ Requires `optuna` package
- ❌ Non-deterministic (results vary slightly each run)

---

## 📊 **Comparison**

| Approach | Trials/Configs | Time (approx) | Best for |
|----------|----------------|---------------|----------|
| Grid (small) | 5 | ~10 min | Quick baseline |
| Grid (medium) | ~80 | ~2-3 hours | Thorough search |
| Grid (large) | ~160 | ~4-6 hours | Exhaustive search |
| **Optuna (50 trials)** | **50** | **~1-2 hours** | **Best balance** ⭐ |
| Optuna (100 trials) | 100 | ~2-4 hours | Maximum quality |

---

## 🎯 **Recommended Workflow**

### Step 1: Quick Optuna Search (30 trials)
```bash
python forca-hack/scripts/train_social_tfidf.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --out_dir outputs \
  --search --search_mode optuna \
  --n_trials 30 \
  --include_svm \
  --ensemble_top_k 8 \
  --save_test_proba --save_oof_proba
```

**Purpose**: Fast exploration to find promising regions

### Step 2: Refined Optuna Search (50-100 trials)
```bash
python forca-hack/scripts/train_social_tfidf.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --out_dir outputs \
  --search --search_mode optuna \
  --n_trials 80 \
  --include_svm \
  --ensemble_top_k 8 \
  --save_test_proba --save_oof_proba
```

**Purpose**: Fine-tune around best configs from Step 1

---

## 🔧 **Search Space (Optuna)**

The Optuna search explores:

- **use_word**: `[True, False]` - Whether to use word ngrams
- **C**: `[0.5, 16.0]` (log scale) - Regularization strength
- **min_df**: `[1, 3]` - Minimum document frequency
- **char_ngram**: `(3-4, 5-8)` - Character ngram range
- **word_ngram**: `(1-2, 2-4)` - Word ngram range (if use_word=True)
- **char_max_features**: `[150k, 200k, 300k, 400k]`
- **word_max_features**: `[60k, 80k, 120k, 180k]` (if use_word=True)
- **clf**: `["lr", "sgd", "svm"]` - Classifier type
- **alpha**: `[1e-6, 1e-4]` (log scale, for SGD only)

---

## 💡 **Tips**

1. **Start with Optuna (50 trials)** - Usually better than grid search
2. **Use `--include_svm`** - SVM often performs best on TF-IDF
3. **Save probabilities** - Use `--save_test_proba --save_oof_proba` for blending
4. **Monitor progress** - Optuna shows progress bar and best score so far
5. **Time limits** - Use `--optuna_timeout` if you have limited time

---

## 📈 **Expected Results**

- **Grid search (medium)**: OOF F1 ~0.635-0.640
- **Optuna (50 trials)**: OOF F1 ~0.638-0.645 ⭐
- **Optuna (100 trials)**: OOF F1 ~0.640-0.648

*Note: Actual results depend on your dataset and random seed*

---

## 🐛 **Troubleshooting**

### "Optuna not installed"
```bash
pip install optuna
```

### "Too slow"
- Reduce `--n_trials` (try 30 instead of 50)
- Use `--optuna_timeout` to set a maximum time
- Skip `--include_svm` (SVM is slower)

### "Results vary each run"
- This is normal for Optuna (Bayesian optimization is stochastic)
- Use `--seed` to make it more reproducible
- For deterministic results, use grid search instead

---

## 🎓 **How Bayesian Optimization Works**

1. **Initial trials**: Randomly sample a few configs
2. **Learn**: Build a probabilistic model of "which configs are good"
3. **Suggest**: Use the model to suggest promising new configs
4. **Repeat**: Continue learning and suggesting until trials are done

This is **much smarter** than grid search, which blindly tests all combinations without learning.

---

**That's it!** Use Optuna for faster, better hyperparameter tuning. 🚀

