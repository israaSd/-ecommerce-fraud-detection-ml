"""Exploratory NO-FS boosting experiment: native missing/categorical support.

Compares the original thesis' fixed XGB/LGBM/CatBoost parameters on the same
chronological 80/20 split, with and without class weighting. The study's original
model/selection results are NOT modified.

Prerequisites:
    pip install pandas numpy scikit-learn xgboost lightgbm catboost

Example on the thesis computer:
    python No_FS_Boosting_Native_Comparison.py

If the data are stored elsewhere:
    python No_FS_Boosting_Native_Comparison.py --data-dir "D:\\datasets\\ieee-fraud-detection"

Optional quicker trial:
    python No_FS_Boosting_Native_Comparison.py --models CAT --trees 50

Outputs in results/no_fs_native_boosting/ beside this script.
"""

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
DEFAULT_DATA_DIR = Path(r"C:\Users\admin\Desktop\M2 thesis\ieee-fraud-detection")
RESULT_COLUMNS = [
    "Classifier", "Class_Weighting", "Categorical_Handling", "Missing_Numeric",
    "Features", "Train_Rows", "Test_Rows", "Threshold", "PR_AUC", "ROC_AUC",
    "Accuracy", "Precision", "Recall", "F1", "MCC", "Specificity",
    "TP", "FN", "FP", "TN", "Training_Seconds",
]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Folder containing train_transaction.csv and train_identity.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "no_fs_native_boosting")
    parser.add_argument("--weight-mode", choices=["both", "weighted", "unweighted"], default="both",
                        help="Both (default) isolates the effect of class imbalance weighting")
    parser.add_argument("--models", nargs="+", choices=["XGB", "LGBM", "CAT"],
                        default=["XGB", "LGBM", "CAT"])
    parser.add_argument("--trees", type=int, default=500,
                        help="500 matches original thesis; smaller values only for a smoke test")
    parser.add_argument("--original-results", type=Path, default=None,
                        help="Optional path to original results/No_FS_results_boosting.csv")
    return parser.parse_args()


def load_data(folder):
    transaction_path = folder / "train_transaction.csv"
    identity_path = folder / "train_identity.csv"
    missing = [str(p) for p in [transaction_path, identity_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot run the actual IEEE-CIS experiment because these input files are missing:\n"
            + "\n".join(missing)
            + "\nSupply the correct --data-dir. No real metrics have been generated."
        )

    tx = pd.read_csv(transaction_path, low_memory=False)
    ident = pd.read_csv(identity_path, low_memory=False)
    df = tx.merge(ident, how="left", on="TransactionID")
    del tx, ident
    df = df.drop(columns=["TransactionID"])
    df["TransactionHour"] = ((df["TransactionDT"] // 3600) % 24)

    # Preserve exact thesis chronological splitting convention.
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    y_train = df["isFraud"].iloc[:cut].copy()
    y_test = df["isFraud"].iloc[cut:].copy()
    df.drop(columns=["isFraud"], inplace=True)
    X_train = df.iloc[:cut].copy()
    X_test = df.iloc[cut:].copy()
    del df

    cat_cols = list(X_train.select_dtypes(include=["object", "category", "string"]).columns)
    numeric_cols = list(X_train.select_dtypes(include=["number"]).columns)
    # IMPORTANT: no numerical imputation, no numeric categorical encoding, no scaling.
    assert X_train.isna().any().any(), "Expected natural missing values in IEEE-CIS data."
    return X_train, X_test, y_train, y_test, cat_cols, numeric_cols


def native_pandas_categoricals(X_train, X_test, cat_cols):
    """Declare raw categoricals; do not manually encode/impute them.

    Pandas category dtype is required by XGBoost and LightGBM. The model itself
    handles categorical splits. Any test-only categories become missing; crucially,
    categories are defined using TRAINING data only (no future vocabulary leakage).
    """
    for name in cat_cols:
        tr_text = X_train[name].astype("string")
        te_text = X_test[name].astype("string")
        dtype = CategoricalDtype(categories=tr_text.dropna().unique().tolist())
        X_train[name] = tr_text.astype(dtype)
        X_test[name] = te_text.astype(dtype)
    return X_train, X_test


def catboost_inputs(X_train, X_test, cat_cols):
    """Preserve NaN for numeric inputs; minimal string replacement for cat features.

    CatBoost supports missing NUMERIC values but does not accept NaN as a
    categorical feature value, so missing categorical labels must use a sentinel.
    This is NOT a global imputation or ordinal encoding step.
    """
    train_cat = X_train.copy(deep=False)
    test_cat = X_test.copy(deep=False)
    for name in cat_cols:
        train_cat[name] = X_train[name].astype("string").fillna("__MISSING_CATEGORY__").astype(str)
        test_cat[name] = X_test[name].astype("string").fillna("__MISSING_CATEGORY__").astype(str)
    return train_cat, test_cat


def make_model(name, weighted, class_ratio, n_trees, cat_cols):
    if name == "XGB":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=(class_ratio if weighted else 1.0),
            n_estimators=n_trees, learning_rate=0.05, max_depth=6,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=-1,
        )
    if name == "LGBM":
        return LGBMClassifier(
            objective="binary", scale_pos_weight=(class_ratio if weighted else 1.0),
            n_estimators=n_trees, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.0,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
        )
    if name == "CAT":
        return CatBoostClassifier(
            loss_function="Logloss", eval_metric="PRAUC",
            class_weights=([1.0, class_ratio] if weighted else None),
            iterations=n_trees, learning_rate=0.05, depth=6,
            cat_features=cat_cols, random_seed=SEED,
            verbose=False, allow_writing_files=False,
        )
    raise ValueError(name)


def fit_and_measure(name, weighted, model, X_train, X_test, y_train, y_test, cat_cols):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel())
    return {
        "Classifier": name,
        "Class_Weighting": "Weighted" if weighted else "Unweighted",
        "Categorical_Handling": ("Native CatBoost (missing categorical label sentinel)"
                                 if name == "CAT" else "Native pandas category dtype"),
        "Missing_Numeric": "Unimputed (NaN left for model)",
        "Features": int(X_train.shape[1]),
        "Train_Rows": len(y_train), "Test_Rows": len(y_test),
        "Threshold": 0.50,
        "PR_AUC": average_precision_score(y_test, y_prob),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Training_Seconds": elapsed,
    }


def save_comparison(results_df, original_results, output_dir):
    if not original_results or not original_results.is_file():
        return
    original = pd.read_csv(original_results)
    needed = ["Model", "PR_AUC", "ROC_AUC", "MCC", "Precision", "Recall"]
    if any(col not in original.columns for col in needed):
        warnings.warn("Original baseline CSV did not have the expected columns; comparison skipped")
        return
    names = {"XGB": "XGBoost", "LGBM": "lightgbm", "CAT": "CatBoost"}
    original["Classifier"] = original["Model"].str.replace("NO FS + ", "", regex=False)
    original["Classifier"] = original["Classifier"].str.casefold()
    base = original.set_index("Classifier")
    compare = results_df.copy()
    for met in ["PR_AUC", "ROC_AUC", "MCC", "Precision", "Recall"]:
        compare[f"Original_{met}"] = compare["Classifier"].map(
            {abbr: float(base.loc[full.casefold(), met])
             for abbr, full in names.items() if full.casefold() in base.index}
        )
        compare[f"Delta_{met}_Native_minus_Original"] = (
            compare[met] - compare[f"Original_{met}"]
        )
    target = output_dir / "native_vs_original_no_fs.csv"
    compare.to_csv(target, index=False)
    print(f"Saved optional original-baseline comparison: {target}")


def main():
    args = get_args()
    if args.original_results is None:
        # Automatically compare with the original script output when it is available.
        candidate = Path(__file__).resolve().parent / "results" / "No_FS_results.csv"
        if candidate.is_file():
            args.original_results = candidate
    if args.trees < 1:
        raise ValueError("--trees must be at least 1")
    np.random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # No artificial scores are prefilled. If this run fails, partial completed runs stay saved.
    X_train, X_test, y_train, y_test, cat_cols, numeric_cols = load_data(args.data_dir)
    train_npos = int((y_train == 1).sum())
    class_ratio = int((y_train == 0).sum()) / train_npos
    print(f"Train: {len(y_train):,} / Test: {len(y_test):,}; "+
          f"test fraud {sum(y_test):,}; predictors {X_train.shape[1]}; "+
          f"categorical predictors {len(cat_cols)}")
    print("No numerical imputation, manual label encoding, feature selection, or scaling.")
    print("CatBoost requires a missing-category text sentinel; numeric NaN remain.")
    print(f"Class ratio: {class_ratio:.4f}; modes: {args.weight_mode}")

    info = {
        "status": "in_progress", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_folder": str(args.data_dir),
        "comparison": "NO FS native missing/categorical processing, optional class weights",
        "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
        "train_fraud": int(train_npos), "test_fraud": int((y_test == 1).sum()),
        "features": int(X_train.shape[1]), "categorical_features": cat_cols,
        "numeric_columns_with_nan": int(X_train[numeric_cols].isna().any().sum()),
        "train_class_ratio": class_ratio, "n_trees": args.trees, "random_seed": SEED,
        "weight_mode": args.weight_mode, "models": args.models,
        "test_split": "exactly chronological 80/20 as original thesis script",
        "threshold": 0.5,
        "caveat": "An exploratory follow-up, not part of the original 48 experiments. "
                  "CatBoost's categorical missing labels use a sentinel. "
                  "Using the old baseline CSV permits a clean preprocessing comparison "
                  "only for native weighted variants.",
        "library_versions": {
            "pandas": pd.__version__,
            "xgboost": __import__("xgboost").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "catboost": __import__("catboost").__version__,
        },
    }
    info_path = args.output_dir / "run_details.json"
    results_path = args.output_dir / "no_fs_native_boosting_results.csv"
    errors_path = args.output_dir / "errors.txt"
    weights = [True, False] if args.weight_mode == "both" else [args.weight_mode == "weighted"]
    rows, errors = [], []

    # Fit CatBoost before converting categorical text columns to pandas category dtype,
    # avoiding a large duplicate of all numeric arrays.
    ordered_models = [m for m in ["CAT", "XGB", "LGBM"] if m in args.models]
    for name in ordered_models:
        if name == "CAT":
            tr, te = catboost_inputs(X_train, X_test, cat_cols)
        else:
            # Convert only once; test categories must follow the training vocabulary.
            if X_train[cat_cols].select_dtypes(include=["category"]).shape[1] != len(cat_cols):
                native_pandas_categoricals(X_train, X_test, cat_cols)
            tr, te = X_train, X_test
        for weighted in weights:
            label = "weighted" if weighted else "unweighted"
            print(f"\n===== {name} | {label} | native missing/categories =====", flush=True)
            try:
                model = make_model(name, weighted, class_ratio, args.trees, cat_cols)
                row = fit_and_measure(name, weighted, model, tr, te, y_train, y_test, cat_cols)
                rows.append(row)
                pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(results_path, index=False)
                print(f"PR-AUC: {row['PR_AUC']:.6f} | ROC-AUC: {row['ROC_AUC']:.6f} | "
                      f"MCC: {row['MCC']:.6f} | Recall: {row['Recall']:.6f} | "
                      f"Precision: {row['Precision']:.6f} | {row['Training_Seconds']:.1f}s")
                print(f"Saved real finished-run results to: {results_path}")
            except Exception as exc:
                message = f"{name} | {label}: {type(exc).__name__}: {exc}"
                print("FAILED: " + message, file=sys.stderr)
                errors.append(message)
                errors_path.write_text("\n\n".join(errors) + "\n", encoding="utf-8")
            finally:
                if "model" in locals():
                    del model
                gc.collect()
        if name == "CAT":
            del tr, te
            gc.collect()

    info["status"] = "completed" if not errors else "completed_with_errors"
    info["successful_runs"] = len(rows)
    info["failed_runs"] = len(errors)
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        summary = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        save_comparison(summary, args.original_results, args.output_dir)
        print("\nFinished runs (actual, not estimated):")
        print(summary[["Classifier", "Class_Weighting", "PR_AUC", "ROC_AUC", "MCC", "Recall", "Precision"]]
              .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    else:
        print("No runs completed; see errors.txt", file=sys.stderr)
        sys.exit(2)
    if errors:
        print(f"Some runs failed. See {errors_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
