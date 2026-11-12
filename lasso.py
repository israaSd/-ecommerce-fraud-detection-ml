import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import time
from sklearn.preprocessing import LabelEncoder
import platform
import sklearn
import sys
import os
import random

os.makedirs("FS_results", exist_ok=True)

random.seed(42)
np.random.seed(42)

start = time.time()

tx  = pd.read_csv('data\\ieee-fraud-detection\\train_transaction.csv')
idn = pd.read_csv('data\\ieee-fraud-detection\\train_identity.csv')
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

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


lasso = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=0.000383,                 
    class_weight="balanced",
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    verbose=1
)

lasso.fit(X_train_scaled, y_train)

print("Iterations used:", lasso.n_iter_)

coef = lasso.coef_.ravel()

# use threshold to avoid tiny non-zero coefficients
threshold = 1e-4
selected_idx = np.where(np.abs(coef) > threshold)[0]

selected_features = X_train.columns[selected_idx]

print("Original features:", X_train.shape[1])
print("Selected features:", len(selected_features))
print(list(selected_features))



selected_features = list(selected_features)

with open("FS_results/lasso_selected_features.json", "w") as f:
    json.dump(selected_features, f, indent=4)

print("Feature list saved!")

runtime = time.time() - start

print(f"Runtime: {runtime/60:.2f} minutes")

results = {
    "method": "Lasso",

    "runtime_seconds": runtime,
    "runtime_minutes": runtime/60,

    "dataset": "IEEE-CIS Fraud Detection",

    "n_samples": len(X_train),
    "n_features_before": X_train.shape[1],
    "n_features_selected": len(selected_features),

    "selected_features": list(selected_features),

    "python_version": sys.version,

    "sklearn_version": sklearn.__version__,
      "C": 0.00035,
    "coefficient_threshold": 1e-4,
    "n_iter": int(lasso.n_iter_[0]),

    "platform": platform.platform(),
    "processor": platform.processor(),

    "random_seed": 42
}

with open("FS_results/Lasso_results.json","w") as f:
    json.dump(results,f,indent=4)
