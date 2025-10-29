# 💹 Financial News Sentiment Analysis using Transformers

This project performs **sentiment analysis on financial news headlines** using **Hugging Face Transformers**.  
It combines multiple Kaggle financial sentiment datasets and fine-tunes a transformer-based model (BERT/FinBERT) to classify news as **Positive**, **Negative**, or **Neutral**.

---

## 📘 Google Colab Notebook

Run the complete workflow in Google Colab:  
🔗 [Open Notebook](https://colab.research.google.com/drive/15C4B8XyTEksGCxsAG0m0LZT022fbYQmD?usp=sharing)

---

## 📊 Datasets Used

1. [Sentiment Analysis Labelled Financial News Data](https://www.kaggle.com/datasets/aravsood7/sentiment-analysis-labelled-financial-news-data)  
2. [Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)  
3. [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

These datasets are **merged and cleaned** into one combined dataset for training and evaluation.

---

## ⚙️ Workflow Overview

### 1️⃣ Data Preprocessing
- Load and merge all Kaggle datasets.  
- Clean text (remove URLs, symbols, extra spaces).  
- Normalize sentiment labels (`positive`, `neutral`, `negative`).

### 2️⃣ Model Preparation
- Tokenize text using a Hugging Face tokenizer.  
- Split dataset into train/test (e.g., 80/20).  
- Convert to Hugging Face `Dataset` objects.

### 3️⃣ Model Training
- Fine-tune a transformer (e.g., **BERT**, **FinBERT**, or **DistilBERT**) using:
  ```python
  AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
