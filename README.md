# 💹 Financial News Sentiment Analysis using Transformers

This project performs **sentiment analysis on financial news headlines** using **Hugging Face Transformers**.  
It combines multiple Kaggle financial sentiment datasets and fine-tunes a transformer-based model (**FinBERT**) to classify news as **Positive**, **Negative**, or **Neutral**.

---

## 📘 Google Colab Notebook

Run the complete workflow in Google Colab:  
🔗 [Open Notebook](https://colab.research.google.com/drive/15C4B8XyTEksGCxsAG0m0LZT022fbYQmD?usp=sharing)

---

## 📊 Datasets Used

1. [Sentiment Analysis Labelled Financial News Data](https://www.kaggle.com/datasets/aravsood7/sentiment-analysis-labelled-financial-news-data)  
2. [Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)  
3. [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

All datasets are merged, cleaned, and normalized into a single corpus for training and evaluation.

---

## ⚙️ Workflow Overview

### 1️⃣ Data Preprocessing
- Load and merge all Kaggle datasets.  
- Clean text (remove URLs, special characters, extra spaces).  
- Normalize sentiment labels to **`positive`**, **`neutral`**, and **`negative`**.

### 2️⃣ Model Preparation
- Use **FinBERT (`yiyanghkust/finbert-tone`)** — a financial-domain variant of BERT.  
- Tokenize text with the FinBERT tokenizer.  
- Split the dataset into training and testing sets (e.g., 80/20).  
- Convert data into `Dataset` objects from Hugging Face `datasets`.

### 3️⃣ Model Training
Fine-tune **FinBERT** using the Hugging Face `Trainer` API:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
