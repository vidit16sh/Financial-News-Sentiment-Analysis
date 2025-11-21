# 💹 Financial News Sentiment Analysis (LSTM-Based Deep Learning Model)

This project performs **sentiment analysis on financial news headlines** using a custom-built **Bidirectional LSTM model in PyTorch**.  
It combines multiple publicly available financial sentiment datasets, preprocesses and merges them, and trains an LSTM model to classify news as **Positive**, **Negative**, or **Neutral**.
The pipeline includes text cleaning, tokenization, vocabulary creation, padded sequence generation, model training, and evaluation.

---

## 📘 Google Colab Notebook

Run the full training and evaluation workflow in Google Colab:  
🔗 **https://colab.research.google.com/drive/15C4B8XyTEksGCxsAG0m0LZT022fbYQmD?usp=sharing**

---

## 📊 Datasets Used

This project uses the following financial sentiment datasets:

1. **Sentiment Analysis Labelled Financial News Data**  
2. **Financial Sentiment Analysis Dataset**  
3. **Sentiment Analysis for Financial News**

All datasets are **merged**, **cleaned**, and **label-normalized** before training.

Files used in the notebook include:
- `Data.csv`  
- `Data_Train.csv`  
- `Data_Test.csv`

---

## ⚙️ Workflow Overview

### 🔹 1. **Data Preprocessing**
- Load and combine all dataset CSV files  
- Convert text to lowercase and remove punctuation  
- Clean spacing, digits, and special characters  
- Encode labels into integers → `{negative: 0, neutral: 1, positive: 2}`  
- Build a **custom tokenizer + vocabulary**  
- Convert sentences into padded sequences of fixed-length integers  

---

### 🔹 2. **Model Architecture (PyTorch)**

The project implements a custom **BiLSTM-based classifier**:

**Embedding → 2-Layer BiLSTM → Dropout → Fully Connected Layer → Softmax**

Key features:
- Embedding layer trained from scratch  
- Bidirectional LSTMs for richer context  
- Dropout for regularization  
- Final linear layer for classification  
- GPU support for faster training  
- Class imbalance handled using `WeightedRandomSampler`  
- Best model saved automatically using validation F1-score  

---

### 🔹 3. **Model Training**

The model is trained using:

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (lr = 1e-3)  
- **Epochs:** 15  
- **Metrics:** Accuracy & Macro F1-score  
- **Batching:** DataLoader with weighted sampling  

During training, the best model is stored as: models/best_model.pth


---

## 📈 Results (Extracted from Training Notebook)

### **Validation Metrics**
| Metric | Value |
|--------|--------|
| **Validation Accuracy** | **71.29%** |
| **Validation F1-Score** | **0.7098** |

### **Test Set Performance**
| Metric | Score |
|--------|--------|
| **Test Accuracy** | ~0.69 |
| **Weighted Precision** | 0.68 |
| **Weighted Recall** | 0.69 |
| **Weighted F1-Score** | 0.68 |

Additional outputs (available in the notebook):
- Full **Classification Report**
- **Confusion Matrix**
- Training & Validation Curves

---

## 🧪 Example Training Snippet

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc, val_f1 = evaluate(...)
```
