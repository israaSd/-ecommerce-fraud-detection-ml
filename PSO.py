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
X_train_PSO, X_val_PSO = X_train.iloc[:c], X_train.iloc[c:]
y_train_PSO, y_val_PSO = y_train.iloc[:c], y_train.iloc[c:]

# 1. REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

rng = np.random.default_rng(SEED)


# 2. PSO PARAMETERS
# =============================================================================

N_PARTICLES = 30
N_ITERATIONS = 50

# Cognitive and social learning coefficients
C1 = 1.8
C2 = 1.8

# Linearly decreasing inertia
W_MAX = 0.9
W_MIN = 0.4

MIN_FEATURES = 30
MAX_FEATURES = 120

# Initial probability of selecting each feature
# Each feature has a 28% initial chance of being selected.
INITIAL_SELECTION_PROB = 0.28

# Limit velocity to avoid sigmoid saturation
# This prevents probabilities from becoming permanently fixed near 0 or 1.
VELOCITY_MIN = -6.0
VELOCITY_MAX = 6.0

CHECKPOINT_FILE = "pso_checkpoint.pkl"

os.makedirs("FS_results", exist_ok=True)

negative_count = int((y_train_PSO == 0).sum())
positive_count = int((y_train_PSO == 1).sum())
if positive_count == 0:
    raise ValueError("The PSO fitness training slice contains no fraud observations.")
class_ratio = negative_count / positive_count

# 3. XGBOOST FITNESS MODEL
# =============================================================================

def fitness(position):
    """
    Evaluate one PSO particle.

    Each binary particle represents one feature subset:
        1 = selected
        0 = not selected

    PSO minimizes cost, so:
        cost = 1 - ROC-AUC
    """

    selected_idx = np.where(position == 1)[0]
    n_selected = len(selected_idx)

    if n_selected < MIN_FEATURES or n_selected > MAX_FEATURES:
        return 1.0, 0.0

    X_tr = X_train_PSO.iloc[:, selected_idx]
    X_va = X_val_PSO.iloc[:, selected_idx]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=class_ratio
    )

    try:
        model.fit(X_tr, y_train_PSO)

        predictions = model.predict_proba(X_va)[:, 1]

        auc = roc_auc_score(y_val_PSO, predictions)
        cost = 1.0 - auc

    except Exception as error:
        print(f"Particle evaluation error: {error}")

        auc = 0.0
        cost = 1.0

    finally:
        del model, X_tr, X_va
        gc.collect()

    return float(cost), float(auc)


# 4. PARTICLE REPAIR
# =============================================================================
# This function is a constraint-repair step. Its job is to make sure every PSO particle selects between: MIN_FEATURES = 30 and MAX_FEATURES = 120

def repair_particle(position):
    """
    Keep the selected feature count between MIN_FEATURES and MAX_FEATURES.
    """

    position = position.astype(np.int8).copy()

    selected_count = int(position.sum())

    if selected_count < MIN_FEATURES:
        unselected_idx = np.where(position == 0)[0]

        add_idx = rng.choice(
            unselected_idx,
            size=MIN_FEATURES - selected_count,
            replace=False
        )

        position[add_idx] = 1

    elif selected_count > MAX_FEATURES:
        selected_idx = np.where(position == 1)[0]

        remove_idx = rng.choice(
            selected_idx,
            size=selected_count - MAX_FEATURES,
            replace=False
        )

        position[remove_idx] = 0

    return position


# 5. INITIALIZE SWARM
# =============================================================================
n_features = X_train.shape[1]

# This function creates the starting PSO population before the optimization begins.
# It initializes two things:
# positions: which features each particle initially selects.
# velocities: the initial movement tendency of each particle.

def initialize_swarm():
    """
    Initialize binary positions and continuous velocities.
    """

    positions = (
        rng.random((N_PARTICLES, n_features))
        < INITIAL_SELECTION_PROB
    ).astype(np.int8)

    for particle_idx in range(N_PARTICLES):
        positions[particle_idx] = repair_particle(
            positions[particle_idx]
        )

    velocities = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(N_PARTICLES, n_features)
    ).astype(np.float32)

    return positions, velocities


# 6. SIGMOID TRANSFER FUNCTION
# =============================================================================

# Converts the continuous velocity of each feature into a probability of selecting that feature.

def sigmoid(velocity):
    """
    Convert continuous velocities into selection probabilities.
    """

    velocity = np.clip(
        velocity,
        VELOCITY_MIN,
        VELOCITY_MAX
    )

    return 1.0 / (1.0 + np.exp(-velocity))

# 7. CHECKPOINT FUNCTIONS
# =============================================================================

# Saves the complete state of the PSO algorithm after each iteration, allowing the optimization to continue later without restarting.

def save_checkpoint(
    positions,
    velocities,
    personal_best_positions,
    personal_best_costs,
    personal_best_aucs,
    global_best_position,
    global_best_cost,
    global_best_auc,
    iteration,
    history
):
    """
    Save the complete PSO state after every iteration.
    """

    checkpoint = {
        "positions": positions,
        "velocities": velocities,

        "personal_best_positions": personal_best_positions,
        "personal_best_costs": personal_best_costs,
        "personal_best_aucs": personal_best_aucs,

        "global_best_position": global_best_position,
        "global_best_cost": global_best_cost,
        "global_best_auc": global_best_auc,

        "iteration": iteration,
        "history": history,

        "rng_state": rng.bit_generator.state
    }

    joblib.dump(checkpoint, CHECKPOINT_FILE)


def load_checkpoint():
    """
    Return the checkpoint when it exists.
    """

    if not os.path.exists(CHECKPOINT_FILE):
        return None

    try:
        return joblib.load(CHECKPOINT_FILE)

    except Exception as error:
        print(f"Could not load checkpoint: {error}")
        return None


# 8. INITIALIZE OR RESUME
# =============================================================================

# The global best is the best solution found by the entire swarm.
# Personal Best (pBest) : Each particle remembers the best solution it has ever found during the search.
checkpoint = load_checkpoint()

if checkpoint is not None:

    print("PSO checkpoint found.")

    positions = checkpoint["positions"]
    velocities = checkpoint["velocities"]

    personal_best_positions = checkpoint[
        "personal_best_positions"
    ]

    personal_best_costs = checkpoint[
        "personal_best_costs"
    ]

    personal_best_aucs = checkpoint[
        "personal_best_aucs"
    ]

    global_best_position = checkpoint[
        "global_best_position"
    ]

    global_best_cost = checkpoint[
        "global_best_cost"
    ]

    global_best_auc = checkpoint[
        "global_best_auc"
    ]

    history = checkpoint["history"]

    start_iteration = checkpoint["iteration"] + 1

    if "rng_state" in checkpoint:
        rng.bit_generator.state = checkpoint["rng_state"]

    print(
        f"Resuming from iteration {start_iteration + 1}"
    )

    print(
        f"Current best AUC: {global_best_auc:.5f}"
    )

else:

    print("Starting a new PSO search.")

    positions, velocities = initialize_swarm()

    personal_best_positions = positions.copy()

    personal_best_costs = np.full(
        N_PARTICLES,
        np.inf
    )

    personal_best_aucs = np.zeros(
        N_PARTICLES
    )

    global_best_position = None
    global_best_cost = np.inf
    global_best_auc = 0.0

    history = []

    start_iteration = 0


# 9. RUN PSO
# =============================================================================

pso_start = time.time()

try:

    for iteration in range(
        start_iteration,
        N_ITERATIONS
    ):

        iteration_start = time.time()

        # Linear decrease from W_MAX to W_MIN
        if N_ITERATIONS > 1:
            progress = iteration / (N_ITERATIONS - 1)
        else:
            progress = 1.0

        inertia = W_MAX - progress * (W_MAX - W_MIN)

        particle_costs = np.zeros(N_PARTICLES)
        particle_aucs = np.zeros(N_PARTICLES)

        print("\n" + "=" * 70)

        print(
            f"Iteration {iteration + 1}/{N_ITERATIONS} "
            f"| inertia={inertia:.3f}"
        )

        print("=" * 70)

        # ---------------------------------------------------------------------
        # Evaluate particles
        # ---------------------------------------------------------------------

        for particle_idx in range(N_PARTICLES):

            cost, auc = fitness(
                positions[particle_idx]
            )

            particle_costs[particle_idx] = cost
            particle_aucs[particle_idx] = auc

            selected_count = int(
                positions[particle_idx].sum()
            )

            # Update personal best
            # Checks whether the current feature subset is better than the best subset previously found by that same particle
            if cost < personal_best_costs[particle_idx]:

                personal_best_costs[particle_idx] = cost
                personal_best_aucs[particle_idx] = auc

                personal_best_positions[particle_idx] = (
                    positions[particle_idx].copy()
                )

            # Update global best
            # Checks whether this particle is better than every solution found by the entire swarm so far
            if cost < global_best_cost:

                global_best_cost = cost
                global_best_auc = auc

                global_best_position = (
                    positions[particle_idx].copy()
                )

                print(
                    f"New global best | "
                    f"AUC={global_best_auc:.5f} | "
                    f"features={selected_count}"
                )

            print(
                f"Particle {particle_idx + 1:02d}/"
                f"{N_PARTICLES} | "
                f"AUC={auc:.5f} | "
                f"features={selected_count}"
            )

        # ---------------------------------------------------------------------
        # Save iteration results
        # ---------------------------------------------------------------------

        iteration_runtime = (
            time.time() - iteration_start
        ) / 60

        iteration_result = {
            "iteration": iteration + 1,
            "best_auc_iteration": float(
                particle_aucs.max()
            ),
            "mean_auc_iteration": float(
                particle_aucs.mean()
            ),
            "global_best_auc": float(
                global_best_auc
            ),
            "global_best_features": int(
                global_best_position.sum()
            ),
            "runtime_minutes": float(
                iteration_runtime
            )
        }

        history.append(iteration_result)

        print(
            f"\nIteration {iteration + 1} completed"
        )

        print(
            f"Iteration best AUC: "
            f"{particle_aucs.max():.5f}"
        )

        print(
            f"Global best AUC: "
            f"{global_best_auc:.5f}"
        )

        print(
            f"Global best features: "
            f"{int(global_best_position.sum())}"
        )

        print(
            f"Iteration runtime: "
            f"{iteration_runtime:.2f} minutes"
        )

        # ---------------------------------------------------------------------
        # Update velocities and positions
        # ---------------------------------------------------------------------
        # r1 and r2 introduce randomness into particle movement.

        r1 = rng.random(
            (N_PARTICLES, n_features)
        )

        r2 = rng.random(
            (N_PARTICLES, n_features)
        )

        # This pulls each particle toward its own personal best subset.
        cognitive_component = (
            C1
            * r1
            * (personal_best_positions - positions)
        )

        # pulls particles toward the best feature subset found by the entire swarm. It allows particles to learn from the strongest solution.
        social_component = (
            C2
            * r2
            * (global_best_position - positions)
        )

        velocities = (
            inertia * velocities
            + cognitive_component
            + social_component
        )

        velocities = np.clip(
            velocities,
            VELOCITY_MIN,
            VELOCITY_MAX
        ).astype(np.float32)

        probabilities = sigmoid(velocities)

        # Generate new binary positions
        positions = (
            rng.random(probabilities.shape)
            < probabilities
        ).astype(np.int8)

        for particle_idx in range(N_PARTICLES):

            # Ensures each new particle selects between 30 and 120 features.
            positions[particle_idx] = repair_particle(
                positions[particle_idx]
            )

        # ---------------------------------------------------------------------
        # Save checkpoint after every completed iteration
        # ---------------------------------------------------------------------

        save_checkpoint(
            positions=positions,
            velocities=velocities,

            personal_best_positions=personal_best_positions,
            personal_best_costs=personal_best_costs,
            personal_best_aucs=personal_best_aucs,

            global_best_position=global_best_position,
            global_best_cost=global_best_cost,
            global_best_auc=global_best_auc,

            iteration=iteration,
            history=history
        )

        print("Checkpoint saved.")

# If you stop the code, it prints a message explaining that the next run will resume from the last fully saved iteration.
except KeyboardInterrupt:

    print("\nPSO stopped manually.")

    print(
        "Run the script again to continue from the "
        "last completed iteration."
    )


# 10. FINAL RESULTS
# =============================================================================

runtime = time.time() - pso_start

selected_indices = np.where(
    global_best_position == 1
)[0]

selected_feature_names = X_train.columns[
    selected_indices
].tolist()

print("\n" + "=" * 70)
print("FINAL PSO RESULTS")
print("=" * 70)

print(
    f"Best validation AUC: "
    f"{global_best_auc:.5f}"
)

print(
    f"Selected features: "
    f"{len(selected_feature_names)}"
)

print(selected_feature_names)

# 11. SAVE SELECTED FEATURES
# =============================================================================

with open(
    "FS_results/pso_selected_features.json",
    "w"
) as file:

    json.dump(
        selected_feature_names,
        file,
        indent=4
    )

runtime = time.time() - start

print(f"Runtime: {runtime/60:.2f} minutes")

# 12. SAVE FULL PSO RESULTS
# =============================================================================

results = {
    "method": "Binary PSO with XGBoost",

    "runtime_seconds": runtime,
    "runtime_minutes": runtime / 60,

    "dataset": "IEEE-CIS Fraud Detection",

    "n_samples": len(X_train),
    "n_features_before": X_train.shape[1],
    "n_features_selected": len(
        selected_feature_names
    ),

    "selected_features": selected_feature_names,

    "best_validation_auc": float(
        global_best_auc
    ),

    "n_particles": N_PARTICLES,
    "n_iterations": N_ITERATIONS,

    "c1": C1,
    "c2": C2,

    "inertia_max": W_MAX,
    "inertia_min": W_MIN,

    "min_features": MIN_FEATURES,
    "max_features": MAX_FEATURES,

    "random_seed": SEED
}

with open(
    "FS_results/PSO_results.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )



# 13. SAVE ITERATION HISTORY
# =============================================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    "FS_results/pso_history.csv",
    index=False
)

print("\nPSO files saved successfully.")
print("1. FS_results/pso_selected_features.json")
print("2. FS_results/PSO_results.json")
print("3. FS_results/pso_history.csv")
print(f"Runtime: {runtime / 60:.2f} minutes")