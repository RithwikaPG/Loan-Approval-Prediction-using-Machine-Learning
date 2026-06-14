# 🏦 Loan Approval Prediction using Machine Learning

> A machine learning system that predicts loan approval status based on applicant financial data, automating decisions for financial institutions.

---

## 📌 Overview

This project classifies loan applications as **Approved** or **Not Approved** using a **Random Forest Classifier**. It processes applicant details — income, credit history, employment status, and more — through a complete ML pipeline including preprocessing, encoding, and prediction.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML model & preprocessing |
| Google Colab | Development environment |

---

## 🤖 Model

- **Algorithm:** Random Forest Classifier (`n_estimators=100`, `random_state=42`)
- **Task:** Binary Classification
- **Output:** `Y` (Approved) / `N` (Not Approved)

---

## 📂 Dataset Features

| Feature | Type |
|--------|------|
| Gender | Categorical |
| Married | Categorical |
| Dependents | Categorical |
| Education | Categorical |
| Self-Employed | Categorical |
| Applicant Income | Numerical |
| Coapplicant Income | Numerical |
| Loan Amount | Numerical |
| Loan Amount Term | Numerical |
| Credit History | Numerical |
| Property Area | Categorical |

---

## ⚙️ Pipeline

```
1. Load Datasets        →  train.csv / test.csv
2. Handle Missing Values →  Mode (categorical), Median (numerical)
3. Label Encoding        →  Convert text → numbers
4. Feature/Target Split  →  X_train, y_train
5. Train Model           →  RandomForestClassifier.fit()
6. Predict               →  model.predict(X_test)
7. Export Results        →  loan_predictions.csv
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn
```

### 3. Run the notebook
Open in **Google Colab** or Jupyter and run all cells.

---

## 📋 Key Code Snippets

**Load Data**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('traindata.csv')
test  = pd.read_csv('testdata.csv')
```

**Handle Missing Values**
```python
# Categorical → mode | Numerical → median
for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
for col in numerical_cols:
    train[col] = train[col].fillna(train[col].median())
```

**Label Encoding**
```python
le = LabelEncoder()
# Example: Male → 1, Female → 0 | Urban → 2, Rural → 1
```

**Train & Predict**
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Export Results**
```python
submission.to_csv('loan_predictions.csv', index=False)
```

---

## 📊 Output

| Prediction | Meaning |
|-----------|---------|
| `Y` | Loan Approved ✅ |
| `N` | Loan Not Approved ❌ |

---

## 🧠 What I Learned

- End-to-end ML pipeline: data loading → preprocessing → training → prediction
- Handling real-world missing values using statistical imputation
- Encoding categorical variables for ML compatibility
- Using ensemble models (Random Forest) for classification tasks

---

## 📁 Project Type

`Machine Learning` &nbsp;·&nbsp; `Classification` &nbsp;·&nbsp; `Predictive Analytics`

---

> *Built with Python & Scikit-learn | Developed on Google Colab*
