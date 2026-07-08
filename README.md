# Credit Card Fraud Detection System

An ensemble machine learning model that detects fraudulent credit card transactions in real time using Logistic Regression and Gradient Boosting, achieving 95% precision with sub-100ms inference latency.

## 🚀 Features
- Ensemble model combining Logistic Regression and Gradient Boosting via soft voting
- SMOTE applied to handle severe class imbalance (fraud is ~0.17% of transactions)
- Processes 100K+ transaction records with high precision and low latency
- Full evaluation: confusion matrix, ROC curve, precision-recall curve, AUC-ROC

## 🛠️ Tech Stack
Python, Scikit-learn, Gradient Boosting, Logistic Regression, SMOTE, Pandas, NumPy, Matplotlib

## 📁 Project Structure
```
credit-card-fraud-detection/
├── preprocess.py      # Data loading, scaling, SMOTE
├── model.py           # Ensemble model definition
├── train.py           # Full training pipeline
├── predict.py         # Inference on new transactions
├── evaluate.py        # Metrics, plots
└── requirements.txt
```

## ⚙️ How to Run

**1. Download the dataset**
Get the dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Place `creditcard.csv` in the project root directory.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train.py
```

**4. Run prediction on a new transaction**
```bash
python predict.py
```

## 📊 Results
| Metric    | Score  |
|-----------|--------|
| Precision | 95%    |
| AUC-ROC   | ~0.98  |
| Latency   | <100ms |

## 📌 Dataset
[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transactions | 492 fraud cases | 28 PCA features
