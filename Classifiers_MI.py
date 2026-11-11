import json
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import os
import random

os.makedirs("results", exist_ok=True)

random.seed(42)
np.random.seed(42)

start = time.time()

tx  = pd.read_csv('C:\\Users\\admin\\Desktop\\M2 thesis\\ieee-fraud-detection\\train_transaction.csv')
idn = pd.read_csv('C:\\Users\\admin\\Desktop\\M2 thesis\\ieee-fraud-detection\\train_identity.csv')
df = tx.merge(idn, how="left", on="TransactionID")
del tx, idn
print(f"Merged shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

df = df.drop(columns=["TransactionID"])

df["TransactionHour"] = ((df["TransactionDT"] // 3600) % 24)

TARGET = "isFraud"

num_cols = [c for c in df.select_dtypes(include=["number"]).columns
            if c not in (TARGET, "TransactionDT")]
cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
print(f"{len(num_cols)} numeric  +  {len(cat_cols)} categorical  =  {len(num_cols)+len(cat_cols)} predictors")

df = df.sort_values("TransactionDT").reset_index(drop=True)
cut = int(len(df) * 0.80)
train = df.iloc[:cut].copy()
test  = df.iloc[cut:].copy()
del df
print(f"Train: {train.shape[0]:,} rows  (fraud {train['isFraud'].mean():.2%})")
print(f"Test : {test.shape[0]:,} rows  (fraud {test['isFraud'].mean():.2%})")


# Calculate medians from training data only
train_medians = train[num_cols].median()

# Fill missing values in train using train medians
train[num_cols] = train[num_cols].fillna(train_medians)

# Fill missing values in test using train medians
test[num_cols] = test[num_cols].fillna(train_medians)


encoders = {}
for col in cat_cols:
    tr_na, te_na = train[col].isna(), test[col].isna()
    tr_s,  te_s  = train[col].astype(str), test[col].astype(str)

    levels  = sorted(tr_s[~tr_na].unique())
    mapping = {lvl: i for i, lvl in enumerate(levels)}

    train[col] = tr_s.map(mapping).where(~tr_na, -1).astype("int32")

    te_code = te_s.map(mapping).where(~te_na, -1)     # missing -> -1
    test[col] = te_code.fillna(-2).astype("int32")    # unseen  -> -2
    encoders[col] = mapping


# Original fixed split
X_train = train.drop(columns=[TARGET]).copy()
y_train = train[TARGET].copy()

X_test = test.drop(columns=[TARGET]).copy()
y_test = test[TARGET].copy()


with open("FS_Results/mi_selected_features.json", "r") as f:
    selected_features = json.load(f)


selected_features = list(selected_features)

X_train_fs = X_train.loc[:, selected_features].copy()
X_test_fs = X_test.loc[:, selected_features].copy()

print("X_train_fs shape:", X_train_fs.shape)
print("X_test_fs shape:", X_test_fs.shape)


negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError("The training set contains no fraud observations.")

class_ratio = negative_count / positive_count

print("Non-fraud training observations:", negative_count)
print("Fraud training observations:", positive_count)
print(f"Negative/Fraud ratio: {class_ratio:.4f}")
# fraud observation receives approximately 27.5 times the weight of a non-fraud observation in XGBoost and CatBoost


# for Logistic Regression and Multilayer Perceptron MLP only
# Logistic Regression is affected by feature magnitude, so use StandardScaler.
scaler = StandardScaler()

X_train_fs_scaled = scaler.fit_transform(X_train_fs)
X_test_fs_scaled = scaler.transform(X_test_fs)

print("Scaled train shape:", X_train_fs_scaled.shape)
print("Scaled test shape :", X_test_fs_scaled.shape)
# Random Forest, XGBoost, and CatBoost do not require this scaling, so they will use the original X_train_fs and X_test_fs.

sample_weights = np.where(
    y_train == 1,
    class_ratio,
    1.0
)

# 1- predictive model : Logistic Regression
lr_model = LogisticRegression(
    class_weight="balanced",
    solver="saga",
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)
# class_weight="balanced" lets Logistic Regression calculate inverse-frequency class weights automatically


# 2- predictive model : Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
# balanced_subsample recalculates balanced weights for each bootstrap sample used to build a tree.


# 3- predictive model : XGBoost
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",

    scale_pos_weight=class_ratio,

    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=1.0,

    tree_method="hist",
    random_state=42,
    n_jobs=-1
)


# 4- predictive model : CatBoost
cat_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC",

    class_weights=[1.0, class_ratio],

    iterations=500,
    learning_rate=0.05,
    depth=6,

    random_seed=42,
    verbose=100,
    allow_writing_files=False
)
# For binary classification, CatBoost documents [1, negative_count / positive_count] as a valid class-weighting approach


# 5- predictive model : LightGBM
lgbm_model = LGBMClassifier(
    objective="binary",

    # Class imbalance
    scale_pos_weight=class_ratio,

    # Boosting parameters
    n_estimators=500,
    learning_rate=0.05,

    # Tree complexity
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,

    # Sampling
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,

    # Regularization
    reg_alpha=0.0,
    reg_lambda=1.0,

    # Reproducibility / computation
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# 6- predictive model : MLP
mlp_model = MLPClassifier(

    # Network architecture
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",

    # Optimizer
    solver="adam",
    learning_rate_init=0.001,

    # Regularization
    alpha=0.0001,

    # Training
    batch_size=2048,
    max_iter=100,

    # Early stopping
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=8,

    # Reproducibility
    random_state=42,
    verbose=True
)


# This function calculates: Accuracy, Precision, Recall, F1-score. ROC-AUC, PR-AUC, Specificity, Confusion-matrix values, Runtime
def evaluate_model(
    model_name,
    model,
    X_train_model,
    X_test_model,
    y_train,
    y_test,
    threshold=0.50,
    sample_weight=None
):
    """
    Fit one classifier and calculate binary classification metrics.

    Parameters
    ----------
    model_name : str
        Name displayed in the results table.
    model : classifier
        A fitted-compatible sklearn, XGBoost, or CatBoost classifier.
    X_train_model, X_test_model :
        Training and test features.
    y_train, y_test :
        Binary labels.
    threshold : float
        Probability threshold used to convert probabilities to 0/1 labels.

    Returns
    -------
    result : dict
        Evaluation metrics and runtime.
    fitted_model :
        Trained classifier.
    """

    start_time = time.time()

    if sample_weight is None:
        model.fit(X_train_model, y_train)
    else:
        model.fit(
            X_train_model,
            y_train,
            sample_weight=sample_weight
        )

    training_time = time.time() - start_time

    # Probability assigned to fraud class = 1
    y_probability = model.predict_proba(X_test_model)[:, 1]

    # Convert probability to predicted class
    y_prediction = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_prediction,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    result = {
        "Model": model_name,
        "Features": X_train_model.shape[1],
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_prediction),
        "Precision": precision_score(
            y_test,
            y_prediction,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_prediction,
            zero_division=0
        ),
        "Specificity": specificity,
        "F1": f1_score(
            y_test,
            y_prediction,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            y_probability
        ),
        "PR_AUC": average_precision_score(
            y_test,
            y_probability
        ),
        "MCC": matthews_corrcoef(y_test, y_prediction),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Training_Time_Seconds": training_time
    }

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Decision threshold: {threshold:.2f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_prediction,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            y_prediction,
            labels=[0, 1]
        )
    )

    print(f"\nROC-AUC: {result['ROC_AUC']:.6f}")
    print(f"PR-AUC : {result['PR_AUC']:.6f}")

    return result, model


# Train and evaluate Logistic Regression
results = []
trained_models = {}

lr_result, trained_lr = evaluate_model(
    model_name="MI + Logistic Regression",
    model=lr_model,
    X_train_model=X_train_fs_scaled,
    X_test_model=X_test_fs_scaled,
    y_train=y_train,
    y_test=y_test,
    threshold=0.50
)

results.append(lr_result)
trained_models["Logistic Regression"] = trained_lr


# Train and evaluate Random Forest
rf_result, trained_rf = evaluate_model(
    model_name="MI + Random Forest",
    model=rf_model,
    X_train_model=X_train_fs,
    X_test_model=X_test_fs,
    y_train=y_train,
    y_test=y_test,
    threshold=0.50
)

results.append(rf_result)
trained_models["Random Forest"] = trained_rf


# Train and evaluate XGBoost
xgb_result, trained_xgb = evaluate_model(
    model_name="MI + XGBoost",
    model=xgb_model,
    X_train_model=X_train_fs,
    X_test_model=X_test_fs,
    y_train=y_train,
    y_test=y_test,
    threshold=0.50
)

results.append(xgb_result)
trained_models["XGBoost"] = trained_xgb


# Train and evaluate CatBoost
cat_result, trained_cat = evaluate_model(
    model_name="MI + CatBoost",
    model=cat_model,
    X_train_model=X_train_fs,
    X_test_model=X_test_fs,
    y_train=y_train,
    y_test=y_test,
    threshold=0.50
)

results.append(cat_result)
trained_models["CatBoost"] = trained_cat

# Train and evaluate LightGBM
lightgbm_result, trained_lightgbm = evaluate_model(
    model_name="MI + lightgbm",
    model=lgbm_model,
    X_train_model=X_train_fs,
    X_test_model=X_test_fs,
    y_train=y_train,
    y_test=y_test,
    threshold=0.50
)

results.append(lightgbm_result)
trained_models["LightGBM"] = trained_lightgbm

# Train and evaluate Multilayer Perceptron MLP
mlp_result, trained_mlp = evaluate_model(
    model_name="MI + MLP",
    model=mlp_model,
    X_train_model=X_train_fs_scaled,
    X_test_model=X_test_fs_scaled,
    y_train=y_train,
    y_test=y_test,
    threshold=0.50,
    sample_weight=sample_weights
)
results.append(mlp_result)
trained_models["MLP"] = trained_mlp

# the final comparison table
results_df = pd.DataFrame(results)

metric_columns = [
    "Model",
    "Features",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC_AUC",
    "PR_AUC",
    "MCC",
    "TN",
    "FP",
    "FN",
    "TP",
    "Training_Time_Seconds"
]

results_df = results_df[metric_columns]

results_df = results_df.sort_values(
    by="PR_AUC",
    ascending=False
).reset_index(drop=True)

print("\nFinal model comparison:")
print(results_df.round(6))


results_df.to_csv('results/Mi_results.csv', index=False)   