"""Exploratory 120-feature RFE experiment using the ORIGINAL selected subset.

Train RF, XGBoost, LightGBM and CatBoost with/without class weighting, using
native missing-value support where available. XGB/LGBM/CAT use categorical
support; RF in scikit-learn 1.4+ supports NaN numerically but, in commonly
installed versions, still requires encoding string categorical variables.

IMPORTANT: The RFE features were selected in the original thesis using training
information and its original preprocessing. This script REUSES the selected list;
it does not rerun RFE or use test information to choose features.

Run from VS Code .py, or paste into a notebook cell and execute:
    python RFE_Selected_Native_Comparison.py

Outputs: results/rfe_native_comparison/rfe_native_results.csv

Optional from command line:
    python RFE_Selected_Native_Comparison.py --models RF XGB LGBM CAT
    python RFE_Selected_Native_Comparison.py --weight-mode unweighted
    python RFE_Selected_Native_Comparison.py --data-dir "D:\\ieee-fraud-detection"
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
DEFAULT_DATA_DIR = Path(r"C:\Users\admin\Desktop\M2 thesis\ieee-fraud-detection")

# Copied from Appendix B.3 of the submitted thesis, NOT refit on test data.
RFE_FEATURES = """
TransactionDT TransactionAmt ProductCD card1 card2 card3 card4 card5 card6 addr1 dist1
P_emaildomain R_emaildomain C1 C2 C3 C4 C5 C6 C8 C9 C10 C11 C12 C13 C14
D1 D2 D3 D4 D5 D6 D8 D10 D11 D15 M2 M3 M4 M5 M6 M9
V6 V12 V13 V19 V20 V30 V37 V38 V45 V49 V53 V54 V55 V56 V61 V62 V64
V67 V69 V70 V72 V75 V76 V82 V83 V87 V90 V91 V96 V99 V102 V130 V131
V133 V136 V149 V152 V162 V165 V173 V177 V187 V197 V201 V217 V258
V261 V281 V282 V283 V285 V293 V294 V296 V307 V308 V309 V310 V311
V312 V313 V314 V315 V317 V320 V324 V333 V335 V339
id_01 id_03 id_06 id_09 id_17 id_30 id_34 DeviceType TransactionHour
""".split()

if len(RFE_FEATURES) != 120 or len(RFE_FEATURES) != len(set(RFE_FEATURES)):
    raise RuntimeError(f"RFE feature list should contain 120 distinct columns, got {len(RFE_FEATURES)}")

RESULT_COLUMNS = [
    "Feature_Set", "Classifier", "Class_Weighting", "Missing_Numeric", "Categorical_Handling",
    "Features", "Train_Rows", "Test_Rows", "Threshold", "PR_AUC", "ROC_AUC",
    "Accuracy", "Precision", "Recall", "F1", "MCC", "Specificity",
    "TP", "FN", "FP", "TN", "Training_Seconds"
]


def script_folder():
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=script_folder() / "results" / "rfe_native_comparison")
    p.add_argument("--weight-mode", choices=["both", "weighted", "unweighted"], default="both")
    p.add_argument("--models", nargs="+", choices=["RF", "XGB", "LGBM", "CAT"], default=["RF", "XGB", "LGBM", "CAT"])
    p.add_argument("--trees", type=int, default=500,
                   help="Number of boosting iterations; RF always uses original 300 trees. "
                        "Use the default 500 for thesis-comparable boosting results.")
    return p.parse_args([] if "ipykernel" in sys.modules else None)


def get_partitions(data_dir):
    transfile = data_dir / "train_transaction.csv"
    identfile = data_dir / "train_identity.csv"
    if not transfile.is_file() or not identfile.is_file():
        raise FileNotFoundError("Missing IEEE-CIS files in: " + str(data_dir) +
                                ". Set --data-dir to the folder containing train_transaction.csv "
                                "and train_identity.csv.")
    tx = pd.read_csv(transfile, low_memory=False)
    ident = pd.read_csv(identfile, low_memory=False)
    df = tx.merge(ident, how="left", on="TransactionID")
    del tx, ident
    df = df.drop(columns=["TransactionID"])
    df["TransactionHour"] = ((df["TransactionDT"] // 3600) % 24)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    missing = sorted(set(RFE_FEATURES) - set(df.columns))
    if missing:
        raise ValueError("RFE feature names are missing from merged data: " + repr(missing))
    cut = int(len(df) * 0.80)
    X_train = df.loc[:cut-1, RFE_FEATURES].copy()
    X_test = df.loc[cut:, RFE_FEATURES].copy()
    y_train = df.loc[:cut-1, "isFraud"].copy()
    y_test = df.loc[cut:, "isFraud"].copy()
    del df
    cat_cols = list(X_train.select_dtypes(include=["object", "category", "string"]).columns)
    return X_train, X_test, y_train, y_test, cat_cols


def set_native_categories(X_train, X_test, cats):
    train, test = X_train.copy(deep=False), X_test.copy(deep=False)
    for col in cats:
        tr = X_train[col].astype("string")
        te = X_test[col].astype("string")
        dtype = CategoricalDtype(categories=tr.dropna().unique().tolist())
        train[col] = tr.astype(dtype)
        test[col] = te.astype(dtype)  # unseen test categories become missing
    return train, test


def set_catboost_categories(X_train, X_test, cats):
    train, test = X_train.copy(deep=False), X_test.copy(deep=False)
    for col in cats:
        train[col] = X_train[col].astype("string").fillna("__MISSING_CATEGORY__").astype(str)
        test[col] = X_test[col].astype("string").fillna("__MISSING_CATEGORY__").astype(str)
    return train, test


def set_rf_categories(X_train, X_test, cats):
    """Only ordinal-encode string categoricals; leave numeric NaN untouched.

    Same training-only, lexicographic mapping convention as the original classifier
    script, with -1 for missing and -2 for an unknown test category.
    """
    train, test = X_train.copy(deep=False), X_test.copy(deep=False)
    for col in cats:
        tr_na, te_na = X_train[col].isna(), X_test[col].isna()
        tr_str, te_str = X_train[col].astype(str), X_test[col].astype(str)
        levels = sorted(tr_str[~tr_na].unique())
        mapping = {label: i for i, label in enumerate(levels)}
        train[col] = tr_str.map(mapping).where(~tr_na, -1).astype("int32")
        test[col] = te_str.map(mapping).where(~te_na, -1).fillna(-2).astype("int32")
    return train, test


def model_for(name, weighted, ratio, iterations, categorical_columns):
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=-1, random_state=SEED
        )
    if name == "XGB":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=ratio if weighted else 1.0, n_estimators=iterations,
            learning_rate=0.05, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
            tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=-1
        )
    if name == "LGBM":
        return LGBMClassifier(
            objective="binary", scale_pos_weight=ratio if weighted else 1.0,
            n_estimators=iterations, learning_rate=0.05, max_depth=6, num_leaves=31,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=-1
        )
    if name == "CAT":
        return CatBoostClassifier(
            loss_function="Logloss", eval_metric="PRAUC",
            class_weights=[1.0, ratio] if weighted else None,
            iterations=iterations, learning_rate=0.05, depth=6,
            cat_features=categorical_columns, random_seed=SEED,
            verbose=False, allow_writing_files=False
        )
    raise ValueError(name)


def train_evaluate(name, weighted, ratio, iterations, cats, train, test, y_train, y_test):
    model = model_for(name, weighted, ratio, iterations, cats)
    started = time.perf_counter()
    model.fit(train, y_train)
    training_time = time.perf_counter() - started
    proba = model.predict_proba(test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = [int(n) for n in confusion_matrix(y_test, pred, labels=[0, 1]).ravel()]
    result = {
        "Feature_Set": "Original thesis RFE 120",
        "Classifier": name,
        "Class_Weighting": "Weighted" if weighted else "Unweighted",
        "Missing_Numeric": "Native NaN (no median imputation)",
        "Categorical_Handling": "Original training-only ordinal encoding (RF requires numeric inputs)" if name == "RF"
        else ("Native CatBoost with missing-category text sentinel" if name == "CAT"
              else "Native category dtype"),
        "Features": train.shape[1], "Train_Rows": len(y_train), "Test_Rows": len(y_test),
        "Threshold": 0.5,
        "PR_AUC": average_precision_score(y_test, proba),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Training_Seconds": training_time
    }
    del model
    return result


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.trees < 1:
        raise ValueError("--trees must be >= 1")
    X_train, X_test, y_train, y_test, cats = get_partitions(args.data_dir)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"RFE features: {len(RFE_FEATURES)}, training rows: {len(y_train)}, test rows: {len(y_test)}")
    print(f"Categorical features in RFE subset: {cats}")
    print("Keeping original 120 RFE features fixed (RFE is NOT rerun).")
    print("For RF, only string categoricals are encoded; numeric NaNs remain native.")
    print(f"Train class ratio: {ratio:.4f} | sklearn: {SKLEARN_VERSION}")

    rows, errors = [], []
    out = args.output_dir / "rfe_native_results.csv"
    failures_path = args.output_dir / "errors.txt"
    weights = [True, False] if args.weight_mode == "both" else [args.weight_mode == "weighted"]
    # Build model-specific data; free copies after each group.
    for name in [n for n in ["RF", "CAT", "XGB", "LGBM"] if n in args.models]:
        if name == "RF":
            tr, te = set_rf_categories(X_train, X_test, cats)
        elif name == "CAT":
            tr, te = set_catboost_categories(X_train, X_test, cats)
        else:
            tr, te = set_native_categories(X_train, X_test, cats)
        for weighted in weights:
            mode = "Weighted" if weighted else "Unweighted"
            print(f"\n===== RFE | {name} | {mode} =====", flush=True)
            try:
                row = train_evaluate(name, weighted, ratio, args.trees, cats, tr, te, y_train, y_test)
                rows.append(row)
                pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out, index=False)
                print(f"PR-AUC {row['PR_AUC']:.4f} | ROC-AUC {row['ROC_AUC']:.4f} | "
                      f"MCC {row['MCC']:.4f} | Recall {row['Recall']:.4f} | "
                      f"Precision {row['Precision']:.4f}")
                print(f"Saved completed runs: {out}")
            except Exception as err:
                msg = f"RFE | {name} | {mode}: {type(err).__name__}: {err}"
                print("FAILED:", msg, file=sys.stderr)
                errors.append(msg)
                failures_path.write_text("\n\n".join(errors), encoding="utf-8")
            finally:
                gc.collect()
        del tr, te
        gc.collect()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Exploratory only; separate from thesis' original 48 models. "
                "Feature selection is not rerun and the RFE list was selected using "
                "the thesis' original preprocessing; native preprocessing at the "
                "classifier stage is tested separately. ROC/PR AUC are across thresholds; "
                "F1/MCC/precision/recall use fixed threshold 0.50.",
        "train_rows": len(y_train), "test_rows": len(y_test),
        "fraud_train": int((y_train == 1).sum()),
        "fraud_test": int((y_test == 1).sum()),
        "features": RFE_FEATURES,
        "categorical_features": cats,
        "class_weight_ratio": float(ratio),
        "sklearn_version": SKLEARN_VERSION,
        "boosting_iterations": args.trees,
        "random_seed": SEED,
        "successes": len(rows), "failures": errors
    }
    (args.output_dir / "run_details.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if rows:
        print("\nFinished runs (real scores; not predictions):")
        print(pd.DataFrame(rows)[["Classifier", "Class_Weighting", "PR_AUC", "ROC_AUC", "MCC", "Recall", "Precision"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"Results file: {out}")
    if errors:
        print("See errors.txt for failed runs. Successful results were preserved.", file=sys.stderr)
    if not rows:
        raise RuntimeError("No experiments completed; check errors.txt.")


if __name__ == "__main__":
    main()
