# Sentiment Analysis of Swiggy Food Reviews using RNN

A Natural Language Processing (NLP) mini project that classifies Swiggy food-delivery reviews as **positive** or **negative** using a Simple Recurrent Neural Network (RNN) built with TensorFlow/Keras.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

Natural Language Processing (NLP) enables computers to understand, interpret and generate human language. This project applies NLP to **sentiment analysis**: given the text of a customer review, predict whether the sentiment is positive or negative.

The notebook walks through the full pipeline: loading data, cleaning text, creating sentiment labels, tokenizing and padding sequences, training an RNN, evaluating it, and predicting sentiment for new reviews.

## Dataset

The project uses a Swiggy restaurant/food-delivery dataset (`swiggy.csv`) with the following columns:

| Column | Description |
|---|---|
| `ID` | Unique record identifier |
| `Area` | Locality of the restaurant |
| `City` | City of the restaurant |
| `Restaurant Price` | Price information for the restaurant |
| `Avg Rating` | Average rating of the restaurant |
| `Total Rating` | Total number of ratings |
| `Food Item` | Food item ordered |
| `Food Type` | Category/type of food |
| `Delivery Time` | Delivery time |
| `Review` | Free-text customer review |

> The dataset file is not included in this repository. Place your copy of the CSV in the project directory (or in Google Drive if running on Colab) and update the file path in the notebook.

**Sentiment labelling:** Labels are derived from `Avg Rating`:

- `Avg Rating > 3.5` → **1 (Positive)**
- `Avg Rating <= 3.5` → **0 (Negative)**

## Project Workflow

1. **Load data**: read the CSV with Pandas (Google Drive mounted in Colab).
2. **Text cleaning**: convert reviews to lowercase and remove special characters and punctuation using regex.
3. **Label creation**: generate binary sentiment labels from ratings.
4. **Handle missing values**: drop rows containing nulls.
5. **Tokenization**: convert text to integer sequences with the Keras `Tokenizer` (vocabulary size of 5,000).
6. **Padding**: pad/truncate every sequence to a fixed length of 200 tokens.
7. **Data splitting**: stratified 80/20 train-test split, then 10% of the training set held out for validation.
8. **Model building and training**: train a SimpleRNN classifier for 5 epochs.
9. **Evaluation and prediction**: measure test accuracy and predict sentiment on custom text.

## Model Architecture

| Layer | Configuration |
|---|---|
| `Embedding` | vocabulary 5,000, embedding dimension 16, input length 200 |
| `SimpleRNN` | 64 units, `tanh` activation |
| `Dense` | 1 unit, `sigmoid` activation |

**Training configuration**

- Loss: binary cross-entropy
- Optimizer: Adam
- Epochs: 5
- Batch size: 32
- Split: stratified, `random_state=42`

## Results

| Metric | Value |
|---|---|
| Test accuracy | **~0.72** |
| Validation accuracy | ~0.72 (stable across epochs) |

**Sample prediction**

```
Review: The food was great.
Sentiment: Positive (Probability: 0.59)
```

> **Note:** Validation accuracy stays flat at about 71.6% across all epochs. This is typical of a model that is mostly predicting the majority class rather than learning from the text. See [Limitations](#limitations) for likely causes and fixes.

## Getting Started

### Prerequisites

- Python 3.9+
- Google Colab (recommended) or a local Jupyter environment

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install pandas numpy scikit-learn tensorflow
```

## Usage

1. Open `Sentiment_Analysis_Mini_Project.ipynb` in Google Colab or Jupyter.
2. Upload the Swiggy CSV, or mount Google Drive and update the path:
   ```python
   data = pd.read_csv('/content/swiggy.csv')
   ```
3. Run all cells in order.
4. Predict on your own text:
   ```python
   sample_review = "The food was great."
   print(predict_sentiment(sample_review))
   ```

## Limitations

- **Labels come from restaurant ratings, not the review text.** `Avg Rating` describes the restaurant overall, so a negative review for a highly rated restaurant gets a positive label. This adds label noise and limits what the model can learn.
- **Class imbalance.** About 72% of samples are positive, so a model that always predicts "positive" would score roughly the same accuracy as this one. Accuracy alone is not a reliable metric here.
- **SimpleRNN is a weak sequence model.** It struggles with long sequences because of vanishing gradients.
- **Short training.** Only 5 epochs with a small embedding size.
- **Deprecation warning.** `input_length` in the `Embedding` layer is deprecated in newer Keras versions and can be removed.

## Future Improvements

- Replace `SimpleRNN` with **LSTM**, **GRU** or **Bidirectional LSTM**.
- Evaluate with **precision, recall, F1-score and a confusion matrix**, not just accuracy.
- Handle class imbalance using class weights, oversampling or undersampling.
- Use review-level sentiment labels, or a pre-trained sentiment model/lexicon (e.g. VADER, TextBlob) for more reliable labels.
- Add **stop-word removal** and **lemmatization** in preprocessing.
- Use pre-trained embeddings (GloVe, Word2Vec) or transformer models such as **BERT**.
- Add `EarlyStopping` and tune hyperparameters (sequence length, vocabulary size, units).
- Plot training and validation accuracy/loss curves.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, scikit-learn, TensorFlow/Keras, `re`
- **Environment:** Google Colab

## Author

**Arvind Kumar**
**MBA Business Analytics**
**Chandigarh University**

---

*This project was built for learning purposes to demonstrate an end-to-end NLP sentiment analysis pipeline.*
