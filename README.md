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
| LASSO | Embedded | Uses L1 regularization for feature selection |
| Random Forest Importance (RFI) | Embedded | Ranks predictors according to Random Forest feature importance |
| Recursive Feature Elimination (RFE) | Wrapper | Iteratively eliminates less important predictors |
| Genetic Algorithm (GA) | Metaheuristic | Uses evolutionary search to identify feature subsets |
| Particle Swarm Optimization (PSO) | Metaheuristic | Uses swarm-based optimization for feature selection |
| Grey Wolf Optimizer (GWO) | Metaheuristic | Uses a population-based optimization strategy inspired by grey wolf hunting behavior |

The comparison investigates whether reducing the feature space improves predictive performance, computational efficiency, or both.

### Feature Selection Implementation

The Python scripts implement the feature selection procedures and generate selected feature subsets for downstream classification experiments.

The metaheuristic methods use XGBoost-based fitness evaluation.

---

## 5. Machine Learning Classifiers

Six classification algorithms were evaluated under each feature selection condition.

| Classifier | Description |
|---|---|
| Logistic Regression (LR) | Linear probabilistic classification |
| Random Forest (RF) | Ensemble of decision trees |
| XGBoost (XGB) | Gradient-boosted decision trees |
| LightGBM (LGBM) | Efficient gradient boosting framework |
| CatBoost (CAT) | Gradient boosting with categorical-feature handling capabilities |
| Multi-Layer Perceptron (MLP) | Neural network classifier |

The classifiers were evaluated using consistent experimental conditions to support comparisons across feature selection methods.

Class weighting or equivalent cost-sensitive adjustments were used to address class imbalance.

Classifier hyperparameters were kept fixed across feature selection conditions.

---

## 6. Model Evaluation

Because fraudulent transactions represent a small proportion of the dataset, classification accuracy alone is insufficient to evaluate model effectiveness.

### Primary Evaluation Metric

**Precision–Recall Area Under the Curve (PR-AUC)**

Average Precision (AP) was used as the primary performance measure.

PR-AUC is particularly informative for imbalanced classification problems because it focuses on the trade-off between precision and recall for the positive class.

The random-ranking baseline corresponds approximately to the positive-class prevalence.

For the test dataset:

**PR-AUC baseline ≈ 0.0344**

### Secondary Evaluation Metrics

Additional metrics included:

- ROC-AUC
- Matthews Correlation Coefficient (MCC)
- Precision
- Recall
- F1-score
- Accuracy

### Classification Thresholds

Performance was evaluated using:

1. Default classification threshold of 0.50
2. Validation-based MCC-optimal thresholds

This enabled analysis of the trade-off between fraud detection and false positives.

---

## 7. Statistical Comparison

Statistical tests were applied to evaluate differences among feature selection methods and classifiers.

The analysis included:

**Friedman Test with Iman–Davenport Correction**

Used to assess overall differences among competing methods.

**Shaffer's Post-Hoc Procedure**

Used for multiple pairwise comparisons while controlling family-wise error.

**Kendall's W**

Used to quantify agreement in method rankings.

The statistical analysis was performed using Python in Jupyter notebooks.

Comparisons were conducted for both PR-AUC and ROC-AUC.

---

## 8. Key Results

### Best Individual Performance

| Metric | Best Configuration | Result |
|---|---|---|
| PR-AUC | RFE + Random Forest | 0.5523 |
| ROC-AUC | RFE + XGBoost | 0.9112 |

The best PR-AUC was approximately **16 times the random-ranking baseline**.

### Feature Selection Efficiency

The no-selection baseline achieved the highest mean PR-AUC across the six classifiers:

**Mean PR-AUC = 0.4425**

RFE achieved:

**Mean PR-AUC = 0.4390**

This corresponds to approximately **99.2% of the no-selection performance**, while reducing the feature space from 433 to 120 predictors.

**Feature reduction: 72.3%**

### Main Findings

- Tree-based ensemble classifiers generally demonstrated the strongest predictive performance.
- Feature selection did not consistently improve average predictive performance across all classifiers.
- RFE provided a favorable balance between dimensionality reduction and predictive performance.
- The effectiveness of feature selection depended on the classifier.
- Metaheuristic approaches required additional computational effort without consistently outperforming simpler methods.
- Statistical comparisons did not identify a significant difference between RFE and the no-selection baseline under the applied post-hoc testing procedure.

Overall, the study demonstrates that feature selection should be evaluated in relation to the classifier, performance metric, and computational requirements rather than assumed to improve predictive accuracy universally.

---

## 9. Technologies and Libraries

**Programming Language:** Python

**Data Processing:**
- pandas
- NumPy

**Machine Learning:**
- scikit-learn
- XGBoost
- LightGBM
- CatBoost

**Statistical Analysis:**
- SciPy
- Statistical testing and multiple-comparison procedures

**Development Environment:**
- Jupyter Notebook
- Python scripts

**Visualization:**
- Matplotlib
- Seaborn

---

## 10. Repository Contents

The repository contains the materials associated with the research.

### Data Preprocessing

Python scripts and documentation describing:

- Data integration
- Chronological train/test splitting
- Missing value handling
- Categorical encoding
- Feature engineering
- Data preparation for modeling

### Feature Selection

Python implementations of:

- Mutual Information
- LASSO
- Random Forest Importance
- Recursive Feature Elimination
- Genetic Algorithm
- Particle Swarm Optimization
- Grey Wolf Optimizer

### Classification Models

Scripts implementing the six machine learning classifiers and their evaluation across the feature selection conditions.

### Statistical Analysis

Jupyter notebooks containing:

- Performance comparisons
- Statistical significance testing
- Post-hoc comparisons
- Model ranking analysis

### Results

Experimental results, summary tables, and figures illustrating model performance.

### Master's Thesis Report

The complete research report provides the theoretical background, literature review, methodology, experimental results, statistical analyses, discussion, and conclusions.

---

## 11. Research Limitations

The study was conducted using the IEEE-CIS dataset; results may differ for other fraud detection datasets or transaction environments.

Classifier hyperparameters were held constant across feature selection conditions to support controlled comparisons. Therefore, the reported results do not necessarily represent the maximum achievable performance of individually optimized models.

The computational cost of wrapper and metaheuristic approaches is also an important consideration when evaluating their practical usefulness.

---

## 12. Author

**Israa Saad**

Master's research in Applied Mathematics – Statistics  
Lebanese University | 2026

**Areas of Interest:** Data Science, Machine Learning, Statistical Modeling, Data Engineering, Predictive Analytics

[LinkedIn](https://www.linkedin.com/in/israa-saad-b1181a164/) | [GitHub](https://github.com/israaSd)

---

*This repository documents academic research conducted as part of the Master's program in Applied Mathematics – Statistics at the Lebanese University.*
