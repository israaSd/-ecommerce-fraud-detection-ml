import time
import json
import pandas as pd
import random
import numpy as np
import random
import os
import gc
import joblib
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

os.makedirs("FS_results", exist_ok=True)

random.seed(42)
np.random.seed(42)

os.makedirs("PSO", exist_ok=True)

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

c = int(len(X_train) * 0.80)
X_train_GWO, X_val_GWO = X_train.iloc[:c], X_train.iloc[c:]
y_train_GWO, y_val_GWO = y_train.iloc[:c], y_train.iloc[c:]



negative_count = int((y_train_GWO == 0).sum())
positive_count = int((y_train_GWO == 1).sum())

class_ratio = negative_count / max(positive_count, 1)

print("BGWO fitness data")
print(f"Training rows:   {len(X_train_GWO):,}")
print(f"Validation rows: {len(X_val_GWO):,}")
print(f"Fraud rate in training: {y_train_GWO.mean():.6f}")
print(f"Fraud rate in validation: {y_val_GWO.mean():.6f}")
print(f"class_ratio: {class_ratio:.4f}")

# GWO PARAMETERS
# =============================================================================
 
N_WOLVES = 30
N_ITERATIONS = 50
 
# 'a' decreases linearly from A_MAX to A_MIN; it controls exploration ->
# exploitation (the GWO analogue of PSO's decreasing inertia).
A_MAX = 2.0
A_MIN = 0.0
 
MIN_FEATURES = 30      # same bounds as GA / PSO
MAX_FEATURES = 120

FEATURE_PENALTY = 0.01
 
INITIAL_SELECTION_PROB = 0.28
POS_MIN, POS_MAX = -6.0, 6.0   # clip before sigmoid to avoid saturation
 
CHECKPOINT_FILE = "gwo_checkpoint.pkl"
os.makedirs("FS_results", exist_ok=True)
 
n_features = X_train.shape[1]  # defined explicitly

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

rng = np.random.default_rng(SEED)

# 3. XGBOOST FITNESS  ---  IDENTICAL METRIC + MODEL TO YOUR GA / PSO
# =============================================================================
 
def fitness(position):
    """
    Evaluate one wolf (one binary feature subset: 1 = selected, 0 = not).
    GWO minimizes cost, so:  cost = 1 - ROC_AUC.
    Returns (cost, auc) where auc is the validation ROC-AUC.
    """
 
    selected_idx = np.where(position == 1)[0]
    n_selected = len(selected_idx)
 
    if n_selected < MIN_FEATURES or n_selected > MAX_FEATURES:
        return 1.0, 0.0
 
    X_tr = X_train_GWO.iloc[:, selected_idx]
    X_va = X_val_GWO.iloc[:, selected_idx]
 
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,     # add the same line to GA / PSO for reproducibility
        n_jobs=-1,
        scale_pos_weight= class_ratio
    )
 
    try:
        model.fit(X_tr, y_train_GWO)
        preds = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_val_GWO, preds)
        cost = 1.0 - auc
 
    except Exception as error:
        print(f"Wolf evaluation error: {error}")
        auc = 0.0
        cost = 1.0
 
    finally:
        del model, X_tr, X_va
        gc.collect()
 
    return float(cost), float(auc)

# 4. REPAIR  (keep selected count within [MIN_FEATURES, MAX_FEATURES])
# =============================================================================
 
def repair_wolf(position):
    position = position.astype(np.int8).copy()
    selected_count = int(position.sum())
 
    if selected_count < MIN_FEATURES:
        unselected_idx = np.where(position == 0)[0]
        add_idx = rng.choice(unselected_idx, MIN_FEATURES - selected_count, replace=False)
        position[add_idx] = 1
 
    elif selected_count > MAX_FEATURES:
        selected_idx = np.where(position == 1)[0]
        remove_idx = rng.choice(selected_idx, selected_count - MAX_FEATURES, replace=False)
        position[remove_idx] = 0
 
    return position

# 5. INITIALIZE PACK
# =============================================================================
 
def initialize_pack():
    pack = (rng.random((N_WOLVES, n_features)) < INITIAL_SELECTION_PROB).astype(np.int8)
    for wolf_idx in range(N_WOLVES):
        pack[wolf_idx] = repair_wolf(pack[wolf_idx])
    return pack

# 6. SIGMOID TRANSFER FUNCTION  (binarizes the continuous GWO position)
# =============================================================================
 
def sigmoid(x):
    x = np.clip(x, POS_MIN, POS_MAX)
    return 1.0 / (1.0 + np.exp(-x))

# 7. CHECKPOINTS  (atomic write, same lesson as your GA/PSO fixes)
# =============================================================================
 
def save_checkpoint(pack, alpha_pos, alpha_cost, alpha_auc,
                    beta_pos, beta_cost, delta_pos, delta_cost,
                    iteration, history):
    checkpoint = {
        "pack": pack,
        "alpha_pos": alpha_pos, "alpha_cost": alpha_cost, "alpha_auc": alpha_auc,
        "beta_pos": beta_pos,   "beta_cost": beta_cost,
        "delta_pos": delta_pos, "delta_cost": delta_cost,
        "iteration": iteration, "history": history,
        "rng_state": rng.bit_generator.state,
    }
    tmp = CHECKPOINT_FILE + ".tmp"
    joblib.dump(checkpoint, tmp)
    os.replace(tmp, CHECKPOINT_FILE)
 
 
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        return joblib.load(CHECKPOINT_FILE)
    except Exception as error:
        print(f"Could not load checkpoint: {error}")
        return None
 
 

# 8. LEADER SELECTION
# =============================================================================
 
def rank_leaders(pack, costs):
    """Return alpha, beta, delta positions and costs (three lowest cost)."""
    order = np.argsort(costs)
    a, b, d = order[0], order[1], order[2]
    return (pack[a].copy(), costs[a],
            pack[b].copy(), costs[b],
            pack[d].copy(), costs[d])

# 9. INITIALIZE OR RESUME
# =============================================================================
 
checkpoint = load_checkpoint()
 
if checkpoint is not None:
    print("GWO checkpoint found.")
    pack = checkpoint["pack"]
    alpha_pos, alpha_cost, alpha_auc = checkpoint["alpha_pos"], checkpoint["alpha_cost"], checkpoint["alpha_auc"]
    beta_pos,  beta_cost  = checkpoint["beta_pos"],  checkpoint["beta_cost"]
    delta_pos, delta_cost = checkpoint["delta_pos"], checkpoint["delta_cost"]
    history = checkpoint["history"]
    start_iteration = checkpoint["iteration"] + 1     # real resume (GA hardcoded this to 0)
    if "rng_state" in checkpoint:
        rng.bit_generator.state = checkpoint["rng_state"]
    print(f"Resuming from iteration {start_iteration + 1} | best AUC: {alpha_auc:.5f}")
 
else:
    print("Starting a new GWO search.")
    pack = initialize_pack()
 
    init_costs = np.zeros(N_WOLVES)
    for wolf_idx in range(N_WOLVES):
        c, _ = fitness(pack[wolf_idx])
        init_costs[wolf_idx] = c
 
    (alpha_pos, alpha_cost,
     beta_pos, beta_cost,
     delta_pos, delta_cost) = rank_leaders(pack, init_costs)
    alpha_auc = 1.0 - alpha_cost
 
    history = []
    start_iteration = 0

# 10. RUN GWO
# =============================================================================
 
gwo_start = time.time()
 
try:
 
    for iteration in range(start_iteration, N_ITERATIONS):
 
        iteration_start = time.time()
 
        a = A_MAX - iteration * (A_MAX - A_MIN) / (N_ITERATIONS - 1) if N_ITERATIONS > 1 else A_MIN
 
        print("\n" + "=" * 70)
        print(f"Iteration {iteration + 1}/{N_ITERATIONS} | a={a:.3f}")
        print("=" * 70)
 
        # ---- move each wolf toward alpha/beta/delta, binarize, repair -------
        for wolf_idx in range(N_WOLVES):
 
            X = pack[wolf_idx].astype(np.float32)
            X_new = np.zeros(n_features, dtype=np.float32)
 
            for leader_pos in (alpha_pos, beta_pos, delta_pos):
                A_vec = 2.0 * a * rng.random(n_features) - a   # in [-a, a]
                C_vec = 2.0 * rng.random(n_features)           # in [0, 2]
                D = np.abs(C_vec * leader_pos - X)
                X_new += (leader_pos - A_vec * D)
 
            X_new /= 3.0
 
            prob = sigmoid(X_new)
            new_position = (rng.random(n_features) < prob).astype(np.int8)
            pack[wolf_idx] = repair_wolf(new_position)
 
        # ---- re-evaluate and refresh leaders --------------------------------
        costs = np.zeros(N_WOLVES)
        aucs = np.zeros(N_WOLVES)
 
        for wolf_idx in range(N_WOLVES):
            c, s = fitness(pack[wolf_idx])
            costs[wolf_idx], aucs[wolf_idx] = c, s
            print(f"Wolf {wolf_idx + 1:02d}/{N_WOLVES} | "
                  f"AUC={s:.5f} | features={int(pack[wolf_idx].sum())}")
 
        # keep current leaders as candidates so the best is never lost
        cand_pack = np.vstack([pack, alpha_pos, beta_pos, delta_pos])
        cand_costs = np.concatenate([costs, [alpha_cost, beta_cost, delta_cost]])
        (alpha_pos, alpha_cost,
         beta_pos, beta_cost,
         delta_pos, delta_cost) = rank_leaders(cand_pack, cand_costs)
        alpha_auc = 1.0 - alpha_cost
 
        # ---- log + checkpoint -----------------------------------------------
        iteration_runtime = (time.time() - iteration_start) / 60
        history.append({
            "iteration": iteration + 1,
            "best_auc_iteration": float(aucs.max()),
            "mean_auc_iteration": float(aucs.mean()),
            "alpha_auc": float(alpha_auc),
            "alpha_features": int(alpha_pos.sum()),
            "runtime_minutes": float(iteration_runtime),
        })
 
        print(f"\nIteration {iteration + 1} completed")
        print(f"Alpha AUC: {alpha_auc:.5f} | features: {int(alpha_pos.sum())} | "
              f"runtime: {iteration_runtime:.2f} min")
 
        save_checkpoint(pack, alpha_pos, alpha_cost, alpha_auc,
                        beta_pos, beta_cost, delta_pos, delta_cost,
                        iteration, history)
        print("Checkpoint saved.")
 
 
except KeyboardInterrupt:
    print("\nGWO stopped manually. Re-run to resume from the last iteration.")


# 11. FINAL RESULTS
# =============================================================================
 
runtime = time.time() - start
 
selected_indices = np.where(alpha_pos == 1)[0]
selected_feature_names = X_train_GWO.columns[selected_indices].tolist()
 
print("\n" + "=" * 70)
print("FINAL GWO RESULTS")
print("=" * 70)
print(f"Best validation AUC: {alpha_auc:.5f}")
print(f"Selected features:   {len(selected_feature_names)}")
print(selected_feature_names)

# 12. SAVE OUTPUTS  (same format/paths style as your GA / PSO scripts)
# =============================================================================
 
with open("FS_results/gwo_selected_features.json", "w") as f:
    json.dump(selected_feature_names, f, indent=4)
 
results = {
    "method": "Binary Grey Wolf Optimization with XGBoost",
    "runtime_seconds": runtime,
    "runtime_minutes": runtime / 60,
    "dataset": "IEEE-CIS Fraud Detection",
    "n_samples": len(X_train_GWO),
    "n_features_before": X_train_GWO.shape[1],
    "n_features_selected": len(selected_feature_names),
    "selected_features": selected_feature_names,
    "best_validation_auc": float(alpha_auc),   # selection criterion, not final performance
    "n_wolves": N_WOLVES,
    "n_iterations": N_ITERATIONS,
    "a_max": A_MAX, "a_min": A_MIN,
    "min_features": MIN_FEATURES, "max_features": MAX_FEATURES,
    "random_seed": SEED,
}
 
with open("FS_results/GWO_results.json", "w") as f:
    json.dump(results, f, indent=4)
 
pd.DataFrame(history).to_csv("FS_results/gwo_history.csv", index=False)
 
print("\nGWO files saved successfully.")
print(f"Runtime: {runtime / 60:.2f} minutes")
 
