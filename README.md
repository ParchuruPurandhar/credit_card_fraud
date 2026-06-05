# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Project Overview

Credit card fraud is one of the most significant challenges faced by financial institutions. This project uses Machine Learning algorithms to identify fraudulent credit card transactions based on transaction features.

The goal is to build a classification model capable of distinguishing between legitimate and fraudulent transactions with high accuracy.

---

## 🎯 Problem Statement

Fraudulent credit card transactions result in substantial financial losses every year. Detecting fraud in real-time is critical for banks and payment systems.

This project leverages machine learning techniques to analyze transaction data and predict whether a transaction is fraudulent.

---

## 📂 Dataset Information

The dataset contains anonymized credit card transaction records.

### Features

| Feature  | Description                          |
| -------- | ------------------------------------ |
| Time     | Seconds elapsed between transactions |
| V1 – V28 | PCA-transformed transaction features |
| Amount   | Transaction amount                   |
| Class    | Target Variable                      |

### Target Variable

| Value | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Joblib
* Jupyter Notebook

---

## 🚀 Project Workflow

### 1. Data Collection

* Loaded dataset using Pandas
* Explored transaction records

### 2. Data Preprocessing

* Checked missing values
* Verified data types
* Performed exploratory analysis

### 3. Train-Test Split

Dataset split into:

* Training Set (70%)
* Testing Set (30%)

```python
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
```

---

## 🤖 Machine Learning Models Used

### Logistic Regression

A baseline classification model.

### Decision Tree Classifier

Captures nonlinear transaction patterns.

### Random Forest Classifier

An ensemble model using multiple decision trees.

### Support Vector Machine (SVM)

Effective for high-dimensional data.

### XGBoost Classifier

Gradient boosting algorithm for enhanced predictive performance.

---

## 📊 Model Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
confusion_matrix()
```

---

## 🏆 Model Comparison

| Model               | Evaluated |
| ------------------- | --------- |
| Logistic Regression | ✅         |
| Decision Tree       | ✅         |
| Random Forest       | ✅         |
| SVM                 | ✅         |
| XGBoost             | ✅         |

The best-performing model was selected and saved for deployment.

---

# 🎯 Sample Prediction

## Sample Input

| Feature | Value  |
| ------- | ------ |
| Time    | 406    |
| V1      | -1.35  |
| V2      | -0.07  |
| V3      | 2.53   |
| V4      | 1.37   |
| V5      | -0.34  |
| Amount  | 149.62 |

```python
sample_transaction = [[
406,
-1.35,
-0.07,
2.53,
1.37,
-0.34,
149.62
]]
```

*(Illustrative example; actual model uses all transaction features.)*

---

## Prediction

```python
prediction = model.predict(sample_transaction)
print(prediction)
```

## Output

```text
[0]
```

### Interpretation

✅ Transaction is **Legitimate**

---

# 🚨 Fraudulent Transaction Example

## Input

| Feature | Value  |
| ------- | ------ |
| Time    | 472    |
| V1      | -5.32  |
| V2      | 4.61   |
| V3      | -7.13  |
| V4      | 3.52   |
| V5      | -2.84  |
| Amount  | 999.99 |

```python
sample_transaction = [[
472,
-5.32,
4.61,
-7.13,
3.52,
-2.84,
999.99
]]
```

## Output

```text
[1]
```

### Interpretation

🚨 Transaction is **Fraudulent**

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── Credit_Card_Fraud_Detection.ipynb
├── credit_model.pkl
├── README.md
├── requirements.txt
│
└── images/
```

---

## 💾 Saving the Model

```python
import joblib

joblib.dump(RFC, "credit_model.pkl")
```

### Load Model

```python
model = joblib.load("credit_model.pkl")
```

---

## 📈 Key Insights

* Fraudulent transactions represent a very small portion of total transactions.
* Class imbalance is a major challenge.
* Ensemble models such as Random Forest and XGBoost provide strong performance.
* Precision and Recall are more important than Accuracy in fraud detection tasks.

---

## 🔮 Future Improvements

* Handle class imbalance using SMOTE.
* Deploy model using Streamlit.
* Integrate real-time transaction monitoring.
* Implement anomaly detection techniques.
* Improve model explainability using SHAP.

---



## 📜 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
jupyter
```


## 👨‍💻 Author

**P. Purandhar**

* Aspiring Data Scientist
* Machine Learning Enthusiast
