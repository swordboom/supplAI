"""
delay_model.py
---------------
Trains a shipment delay prediction model on the real logistics dataset
(Is_delayed_prediction_Train_*.csv) and uses it to compute per-route
delay probabilities for the supply chain graph.

Model choice (auto-detected at runtime):
  1. XGBoost with GPU  (device='cuda', tree_method='hist')  â† preferred
  2. RandomForest CPU  (n_jobs=-1)                          â† fallback

The trained model is saved to models/delay_model.pkl and reloaded on
subsequent runs â€” so training happens only ONCE.

Key features used (all present in the dataset, no target leakage):
  distance, shipment_weight, SLA, pickup_metro, pickup_non_metro,
  drop_metro, drop_non_metro, cp_delay_per_quarter, cp_ontime_per_quarter,
  cp_delay_per_month, cp_ontime_per_month, holiday_in_between,
  is_sunday_in_between
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH   = PROJECT_ROOT / "datasets" / "Is_delayed_prediction_Train_2_Avatar_2_Version_1_05_09_2019.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_PATH   = MODEL_DIR / "delay_model.pkl"

# ---------------------------------------------------------------------------
# Feature columns used for training (13 features)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "distance",
    "shipment_weight",
    "SLA",
    "pickup_metro",
    "pickup_non_metro",
    "drop_metro",
    "drop_non_metro",
    "cp_delay_per_quarter",
    "cp_ontime_per_quarter",
    "cp_delay_per_month",
    "cp_ontime_per_month",
    "holiday_in_between",
    "is_sunday_in_between",
]

TARGET_COL = "is_delayed"


# ---------------------------------------------------------------------------
# Runtime safety helpers
# ---------------------------------------------------------------------------
def _force_xgb_cpu_predictor(model: Any) -> None:
    """
    Avoid CUDA/CPU inference mismatch warnings at runtime.
    This keeps inference on CPU even if the model was trained on CUDA.
    """
    try:
        module_name = model.__class__.__module__.lower()
    except Exception:
        module_name = ""

    if "xgboost" not in module_name:
        return

    try:
        model.set_params(device="cpu")
    except Exception:
        pass

    try:
        booster = model.get_booster()
        booster.set_param({"device": "cpu"})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def _resolve_dataset_columns(train_path: Path) -> Tuple[list[str], dict[str, str]]:
    """
    Resolve required columns from CSV header while tolerating accidental
    leading/trailing whitespace in source column names.
    """
    required = FEATURE_COLS + [TARGET_COL]
    header = pd.read_csv(train_path, nrows=0).columns.tolist()

    # Map stripped names -> original names present in file.
    stripped_to_original: dict[str, str] = {}
    for col in header:
        key = str(col).strip()
        if key and key not in stripped_to_original:
            stripped_to_original[key] = col

    missing = [col for col in required if col not in stripped_to_original]
    if missing:
        raise ValueError(
            "Training dataset is missing required column(s): "
            + ", ".join(missing)
        )

    usecols = [stripped_to_original[col] for col in required]
    rename_map = {stripped_to_original[col]: col for col in required}
    return usecols, rename_map


def _load_training_frame(train_path: Path) -> pd.DataFrame:
    """
    Load training dataframe with canonical feature names.
    """
    usecols, rename_map = _resolve_dataset_columns(train_path)
    df = pd.read_csv(train_path, usecols=usecols)
    df.rename(columns=rename_map, inplace=True)
    return df


def load_training_data(train_path: Path = TRAIN_PATH) -> pd.DataFrame:
    """
    Public loader for delay training data with canonical column names.
    """
    cols_to_load = FEATURE_COLS + [TARGET_COL]
    df = _load_training_frame(train_path)
    df.dropna(subset=cols_to_load, inplace=True)
    return df


def _get_model():
    """Return the best available classifier."""
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators      = 300,
            max_depth         = 8,
            learning_rate     = 0.1,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            tree_method       = "hist",    # GPU-compatible histogram method
            device            = "cuda",    # Use CUDA GPU
            eval_metric       = "logloss",
            random_state      = 42,
            n_jobs            = -1,
            verbosity         = 1,
        )
        print("  [delay_model] Using XGBoost with CUDA GPU")
        return model, "xgboost_cuda"
    except (ImportError, Exception) as e:
        print(f"  [delay_model] XGBoost/CUDA not available ({e}), using RandomForest CPU")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators = 150,
            max_depth    = 12,
            n_jobs       = -1,
            random_state = 42,
        )
        return model, "random_forest_cpu"


def train_and_save(
    train_path: Path = TRAIN_PATH,
    model_path: Path = MODEL_PATH,
) -> Any:
    """
    Load the full training dataset, train the model, save to disk.

    Returns the trained model object.
    """
    print(f"  [delay_model] Loading training data from:\n    {train_path}")
    t0 = time.time()

    # Load only the columns we need (faster I/O on large file)
    df = load_training_data(train_path)

    print(f"  [delay_model] Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"  [delay_model] Class balance: {df[TARGET_COL].value_counts().to_dict()}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)

    model, model_type = _get_model()

    print(f"  [delay_model] Training ({model_type}) on {len(X):,} samples â€¦")
    t1 = time.time()
    model.fit(X, y)
    _force_xgb_cpu_predictor(model)
    elapsed = time.time() - t1
    print(f"  [delay_model] Training complete in {elapsed:.1f}s")

    # Evaluate on training set (quick sanity check)
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    acc    = accuracy_score(y, y_pred)
    auc    = roc_auc_score(y, y_prob)
    print(f"  [delay_model] Train accuracy: {acc:.4f} | AUC: {auc:.4f}")

    # Save model and metadata together
    artifact = {
        "model":       model,
        "model_type":  model_type,
        "features":    FEATURE_COLS,
        "train_rows":  len(X),
        "train_acc":   round(acc, 4),
        "train_auc":   round(auc, 4),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"  [delay_model] Model saved to {model_path}")

    return artifact


def load_or_train(
    train_path: Path = TRAIN_PATH,
    model_path: Path = MODEL_PATH,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Load the model from disk if it exists, else train and save.

    Parameters
    ----------
    force_retrain : bool â€” set True to re-train even if model file exists

    Returns
    -------
    dict with keys: model, model_type, features, train_rows, train_acc, train_auc
    """
    if model_path.exists() and not force_retrain:
        print(f"  [delay_model] Loading cached model from {model_path}")
        artifact = joblib.load(model_path)
        _force_xgb_cpu_predictor(artifact.get("model"))
        print(f"  [delay_model] Loaded ({artifact['model_type']}) | "
              f"Train acc: {artifact['train_acc']} | AUC: {artifact['train_auc']}")
        return artifact

    return train_and_save(train_path, model_path)


def load_model(
    model_path: Path = MODEL_PATH,
) -> Dict[str, Any]:
    """
    Load an already-trained delay model artifact.
    Raises FileNotFoundError when model is absent.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Delay model not found at {model_path}. "
            "Train it first with: python train_models.py"
        )

    artifact = joblib.load(model_path)
    _force_xgb_cpu_predictor(artifact.get("model"))
    if "model" not in artifact:
        raise ValueError(f"Invalid delay model artifact at {model_path}: missing 'model'")
    if "features" not in artifact:
        artifact["features"] = FEATURE_COLS
    return artifact


def predict_delay_proba(
    artifact:          Dict[str, Any],
    distance_m:        float = 500_000,
    shipment_weight_g: float = 200,
    sla:               int   = 1,
    pickup_metro:      int   = 1,
    pickup_non_metro:  int   = 0,
    drop_metro:        int   = 1,
    drop_non_metro:    int   = 0,
    cp_delay_q:        float = 0.15,
    cp_ontime_q:       float = 0.85,
    cp_delay_m:        float = 0.15,
    cp_ontime_m:       float = 0.85,
    holiday:           int   = 0,
    is_sunday:         int   = 0,
) -> float:
    """
    Predict the probability of delay for a given route/shipment profile.

    All parameters have sensible defaults so you can call it with just the
    most important ones (e.g. distance_m, cp_delay_q).

    Returns
    -------
    float in [0, 1] â€” probability of delay
    """
    model = artifact["model"]
    features = np.array([[
        distance_m,
        shipment_weight_g,
        sla,
        pickup_metro,
        pickup_non_metro,
        drop_metro,
        drop_non_metro,
        cp_delay_q,
        cp_ontime_q,
        cp_delay_m,
        cp_ontime_m,
        holiday,
        is_sunday,
    ]], dtype=np.float32)

    proba = model.predict_proba(features)[0][1]
    return float(proba)


def estimate_node_delay(
    artifact: Dict[str, Any],
    G,
    node: str,
) -> float:
    """
    Estimate delay probability for a graph node based on its incoming
    edges' distance and average cargo weight.

    Returns a float in [0, 1].
    """
    import networkx as nx

    # Collect incoming edge stats
    in_edges = list(G.in_edges(node, data=True))
    if not in_edges:
        # Use node's tier as a proxy (higher tier = downstream = lower delay risk)
        tier = G.nodes[node].get("tier", 3)
        return max(0.05, 0.4 - tier * 0.05)

    # Average distance and weight of incoming shipments
    distances = [d.get("distance_m", 500_000) for _, _, d in in_edges]
    weights   = [d.get("avg_weight_kg", 200)  for _, _, d in in_edges]

    avg_distance = float(np.mean(distances))
    avg_weight   = float(np.mean(weights))

    return predict_delay_proba(
        artifact,
        distance_m        = avg_distance,
        shipment_weight_g = avg_weight * 1000,  # kg â†’ g
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    artifact = load_or_train()
    # Test a quick prediction
    p = predict_delay_proba(artifact, distance_m=1_000_000, cp_delay_q=0.25)
    print(f"\nTest prediction (long route, high delay rate): {p:.3f}")
    p2 = predict_delay_proba(artifact, distance_m=100_000, cp_delay_q=0.05)
    print(f"Test prediction (short route, low delay rate): {p2:.3f}")

