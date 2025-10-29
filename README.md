# 🧠 Financial News Sentiment Analysis

This project performs **sentiment analysis on financial news headlines** using a combination of three public Kaggle datasets.  
The goal is to classify financial news into **Positive**, **Negative**, or **Neutral** sentiments using NLP and deep learning techniques.

---

## 📂 Datasets Used

1. [Sentiment Analysis Labelled Financial News Data](https://www.kaggle.com/datasets/aravsood7/sentiment-analysis-labelled-financial-news-data)
2. [Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)
3. [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

All three datasets were merged, cleaned, and preprocessed to create a **comprehensive dataset** for financial sentiment classification.

---

## ⚙️ Model Workflow

1. **Data Loading & Merging**
   - Loads all three datasets from Kaggle.
   - Standardizes sentiment labels (`positive`, `negative`, `neutral`).
   - Combines and cleans the text data.

2. **Text Preprocessing**
   - Lowercasing, removing punctuation, numbers, and stopwords.
   - Tokenization and padding for deep learning input.

3. **Model Training**
   - Trained using LSTM / BiLSTM (or transformer if implemented in Colab).
   - Model evaluated using accuracy, precision, recall, and F1-score.

4. **Inference**
   - Allows sentiment prediction for new unseen financial news.

---

## 🚀 Getting Started

### 🔹 Option 1 — Run on Google Colab
You can directly open and run the Colab notebook here:  
[📘 Google Colab Notebook](https://colab.research.google.com/drive/15C4B8XyTEksGCxsAG0m0LZT022fbYQmD?usp=sharing)

Colab already comes with most dependencies preinstalled, but you can still use the setup script if needed.

### 🔹 Option 2 — Run Locally

#### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
