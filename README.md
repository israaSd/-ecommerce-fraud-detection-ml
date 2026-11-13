# -ecommerce-fraud-detection-ml

# E-Commerce Fraud Detection Using Machine Learning
### Comparative Feature Selection and Predictive Modeling

**Master's Thesis — Applied Mathematics & Statistics**  
**Lebanese University | 2026**  
**Author:** Israa Saad

---

## 1. Project Overview

This project investigates the effectiveness of feature selection techniques and machine learning classifiers for detecting fraudulent e-commerce transactions.

Fraud detection presents several challenges, including highly imbalanced datasets, high-dimensional feature spaces, missing values, and complex relationships between transaction characteristics.

The objective of this research was to evaluate how different feature selection strategies influence predictive performance while reducing the number of input features.

The study compares **8 feature selection conditions and 6 machine learning classifiers**, resulting in **48 experimental configurations**.

Particular attention is given to:

- Data preprocessing and feature engineering
- Handling class imbalance
- Filter, embedded, wrapper, and metaheuristic feature selection
- Machine learning classification
- Model evaluation using appropriate metrics
- Statistical comparison of experimental results

## 2. Dataset

**Dataset:** IEEE-CIS Fraud Detection  
**Source:** Kaggle — IEEE-CIS Fraud Detection Competition

[Access the dataset on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

### Dataset Characteristics

| Characteristic | Description |
|---|---|
| Total transactions | 590,540 |
| Fraudulent transactions | 20,663 |
| Fraud prevalence | Approximately 3.5% |
| Initial data sources | Transaction and identity tables |
| Final feature space | 433 predictors |
| Target variable | `isFraud` |
| Prediction task | Binary classification |

The dataset contains transaction attributes, anonymized numerical variables, categorical information, and missing values.

The transaction and identity tables were merged using a left join on `TransactionID`.

**Data availability:** The original dataset is not redistributed in this repository. It can be obtained through Kaggle, subject to its access conditions.

---

## 3. Research Workflow

The methodology follows a structured experimental pipeline:

**Data Integration → Chronological Splitting → Preprocessing → Feature Selection → Model Training → Evaluation → Statistical Comparison**

### 3.1 Data Preprocessing

Data preprocessing was performed to prepare the dataset for machine learning while preventing data leakage.

The main steps included:

**1. Chronological Data Splitting**

Transactions were ordered using `TransactionDT` and divided into:

- Training set: 80% (472,432 transactions)
- Test set: 20% (118,108 transactions)

A chronological split was selected to better reflect the temporal nature of fraud detection.

**2. Missing Value Handling**

- Numerical variables: Median imputation
- Categorical variables: Missing values encoded as `-1`
- Unseen categorical levels in the test set: Encoded as `-2`

Preprocessing parameters were determined using the training data.

**3. Categorical Encoding**

Categorical variables were converted into numerical representations using mappings learned from the training data.

**4. Feature Engineering**

A transaction-hour feature was derived from `TransactionDT` to capture daily transaction patterns.

**5. Feature Scaling**

Standardization was applied for Logistic Regression, using parameters estimated from the training data.

**6. Class Imbalance**

The dataset contains approximately 3.5% fraudulent transactions.

Cost-sensitive learning and class weighting were used to account for class imbalance, without applying SMOTE or undersampling.

The preprocessing methodology is documented in the accompanying preprocessing PDF.

---

## 4. Feature Selection Methods

Seven feature selection techniques were compared against a baseline without feature selection.

Each selection method retained **120 predictors**, allowing comparisons under a common feature budget.

| Method | Category | Description |
|---|---|---|
| No Feature Selection | Baseline | Uses all 433 predictors |
| Mutual Information (MI) | Filter | Measures statistical dependency between features and the target |
| LASSO | Embedded | Uses L1 regularization for
