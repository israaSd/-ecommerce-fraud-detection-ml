import time
import json
import platform
import sklearn
import xgboost
import sys
import pandas as pd
import random
import numpy as np
import random
import joblib
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import os
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

c = int(len(X_train) * 0.80)
X_train_GA, X_val_GA = X_train.iloc[:c], X_train.iloc[c:]
y_train_GA, y_val_GA = y_train.iloc[:c], y_train.iloc[c:]

n_features = X_train.shape[1]


POP_SIZE = 30
N_GENERATIONS = 50

CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.03
ELITE_RATE = 0.10

MIN_FEATURES = 30
MAX_FEATURES = 120

CHECKPOINT_FILE = "ga_checkpoint.pkl"

negative_count = int((y_train_GA == 0).sum())
positive_count = int((y_train_GA == 1).sum())

if positive_count == 0:
    raise ValueError("The GA fitness training slice contains no fraud observations.")

class_ratio = negative_count / positive_count


def fitness(individual):
    selected_idx = [i for i in range(n_features) if individual[i] == 1]

    if len(selected_idx) < MIN_FEATURES:
        return 0,

    X_tr = X_train_GA.iloc[:, selected_idx]
    X_va = X_val_GA.iloc[:, selected_idx]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=class_ratio
    )

    model.fit(X_tr, y_train_GA)

    preds = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_val_GA, preds)

    return auc,

def init_population():
    population = []
    for _ in range(POP_SIZE):
        ind = np.random.randint(0, 2, n_features)

        # enforce max_features constraint
        if ind.sum() > MAX_FEATURES:
            idx = np.where(ind == 1)[0]
            remove = np.random.choice(idx, ind.sum() - MAX_FEATURES, replace=False)
            ind[remove] = 0

        population.append(ind)
    return population

def tournament_selection(pop, scores):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if scores[i] > scores[j] else pop[j]

def crossover(p1, p2):
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()

    point = random.randint(1, n_features - 1)

    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])

    return c1, c2

def mutate(ind):
    for i in range(n_features):
        if random.random() < MUTATION_RATE:
            ind[i] = 1 - ind[i]

    # enforce max constraint
    if ind.sum() > MAX_FEATURES:
        idx = np.where(ind == 1)[0]
        remove = np.random.choice(idx, ind.sum() - MAX_FEATURES, replace=False)
        ind[remove] = 0

    return ind

def save_checkpoint(pop, gen, best_ind, best_score):
    joblib.dump({
        "population": pop,
        "generation": gen,
        "best_ind": best_ind,
        "best_score": best_score
    }, CHECKPOINT_FILE)

def load_checkpoint():
    try:
        return joblib.load(CHECKPOINT_FILE)
    except:
        return None


checkpoint = load_checkpoint()

if checkpoint:
    population = checkpoint["population"]
    start_gen = checkpoint["generation"] + 1
    best_ind = checkpoint["best_ind"]
    best_score = checkpoint["best_score"]
else:
    population = init_population()
    start_gen = 0
    best_ind = None
    best_score = 0

for gen in range(start_gen, N_GENERATIONS):
    
    scores = [fitness(ind)[0] for ind in population]

    # track best
    gen_best_idx = np.argmax(scores)

    if scores[gen_best_idx] > best_score:
        best_score = scores[gen_best_idx]
        best_ind = population[gen_best_idx].copy()

    print(f"Generation {gen} | Best AUC: {best_score:.5f}")

    # ELITE
    elite_num = int(ELITE_RATE * POP_SIZE)
    elite_idx = np.argsort(scores)[-elite_num:]
    new_population = [population[i].copy() for i in elite_idx]

    # GENERATE NEW POPULATION
    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, scores)
        p2 = tournament_selection(population, scores)

        c1, c2 = crossover(p1, p2)

        c1 = mutate(c1)
        c2 = mutate(c2)

        new_population.append(c1)
        if len(new_population) < POP_SIZE:
            new_population.append(c2)

    population = new_population
    save_checkpoint(population, gen, best_ind, best_score)

# FINAL OUTPUT
selected_features = np.where(best_ind == 1)[0]

print("Best AUC:", best_score)
print("Selected features:", len(selected_features))
print(selected_features)

selected_features = np.where(best_ind == 1)[0]

selected_feature_names = X_train.columns[selected_features]

print("Best AUC:", best_score)
print("Selected features:", len(selected_feature_names))
print(selected_feature_names)


selected_features = list(selected_feature_names)

with open("FS_results/ga_selected_features.json", "w") as f:
    json.dump(selected_features, f, indent=4)

print("Feature list saved!")


runtime = time.time() - start

print(f"Runtime: {runtime/60:.2f} minutes")

results = {
    "method": "GA",

    "runtime_seconds": runtime,
    "runtime_minutes": runtime/60,

    "dataset": "IEEE-CIS Fraud Detection",

    "n_samples": len(X_train),
    "n_features_before": X_train.shape[1],
    "n_features_selected": len(selected_features),

    "fitness_metric": "roc_auc",
    "best_fitness": float(best_score),
    "pop_size": POP_SIZE,
    "n_generations": N_GENERATIONS,
    "crossover_rate": CROSSOVER_RATE,
    "mutation_rate": MUTATION_RATE,
    "elite_rate": ELITE_RATE,
    "min_features": MIN_FEATURES,
    "max_features": MAX_FEATURES,

    "selected_features": list(selected_features),

    "python_version": sys.version,

    "sklearn_version": sklearn.__version__,
    "xgboost_version": xgboost.__version__,

    "platform": platform.platform(),
    "processor": platform.processor(),

    "random_seed": 42
}

with open("FS_results/GA_results.json","w") as f:
    json.dump(results,f,indent=4)
