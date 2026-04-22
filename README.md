# Toxic Comment Classification

Multi-class classification of social media comments as normal, toxic, or severely toxic using Bag-of-Words representations and logistic regression, with strategies to address class imbalance.

## Overview

This project builds a content moderation classifier on a real-world dataset of social media comments originally released by the [Google Jigsaw team](https://jigsaw.google.com). The pipeline covers text featurization via Bag-of-Words, multi-class logistic regression with cross-validated hyperparameter selection, and two strategies — resampling and class reweighting — to address the severe class imbalance inherent in toxicity data.

**Note:** The dataset contains offensive language. Comments are analyzed only to train a detection model.

## Dataset

`toxic.csv` — A subset of the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

**Labels used:**
- `normal` (0): Neither toxic nor severely toxic
- `toxic` (1): Toxic but not severely toxic  
- `severely toxic` (2): Both toxic and severely toxic flags set

Labels are derived by summing the `toxic` and `severe_toxic` binary columns, yielding a single three-class target.

## Methods

### Text Featurization (Bag-of-Words)

Comments are vectorized using scikit-learn's `CountVectorizer` with:
- `token_pattern=r"(?u)\b[a-zA-Z][a-zA-z]+\b"` — English alphabetic tokens only (excludes numbers, punctuation, and non-English characters that are unlikely to contribute to toxicity)
- `min_df=0.0002` — Drops tokens appearing in fewer than 0.02% of comments (likely typos or noise)
- Binary counts disabled — word frequency is preserved since repetition may signal stronger toxicity

This produces a feature matrix `X` with at most 10,000 vocabulary tokens per row.

### Multi-class Logistic Regression

Train/test split (70/30, `random_state=2024`). A validation curve over `C ∈ {0.001, 0.01, 0.1, 0.3, 1.0}` using 5-fold cross-validation guides regularization selection. **`C=0.3`** is selected — it achieves the best validation performance without a large generalization gap.

### Addressing Class Imbalance

The original model suffers on severely toxic comments, which are far rarer than normal ones. Two strategies are compared:

**Resampling:** Upsampling toxic and severely toxic training examples (or downsampling normal ones) to create a balanced training set.

**Class reweighting:** Setting `class_weight='balanced'` in `LogisticRegression` to upweight rare classes during optimization, without modifying the data.

**Selected approach: Class reweighting** — it better detects toxic comments while maintaining comparable performance on normal and severely toxic classes. In a content moderation context, false negatives (missed toxic comments) are more harmful than false positives, making recall on toxic classes the priority metric.

### Model Inspection

The top 50 vocabulary tokens by coefficient magnitude reveal a key limitation of BOW models: **lack of context**. Words like `"son"` or `"head"` appear in the top toxicity-associated vocabulary yet are routinely used in non-toxic contexts. Without sentence-level context, the model will generate false positives for comments that happen to include these words innocuously.

## Results Summary

| Model | Normal Accuracy | Toxic Accuracy | Severe Accuracy |
|-------|----------------|----------------|-----------------|
| Baseline logistic (no balance) | High | Moderate | Low |
| Resampled | Moderate | Higher | Moderate |
| **Class reweighting** | **High** | **Highest** | **Moderate** |

## Key Findings

- BOW + logistic regression achieves >90% overall accuracy but struggles with rare classes without intervention.
- Class reweighting outperforms resampling for improving minority class detection without degrading majority class performance.
- BOW models are inherently context-free, making certain common words reliable false-positive generators.

## Tech Stack

- Python 3
- scikit-learn (`CountVectorizer`, `LogisticRegression`, `ValidationCurveDisplay`, `ConfusionMatrixDisplay`, `resample`)
- pandas, NumPy

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook toxicity.ipynb
```

The `toxic.csv` data file must be in the same directory as the notebook.
