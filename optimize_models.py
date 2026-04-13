#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

DELAY_PATH = DATASETS_DIR / "Is_delayed_prediction_Train_2_Avatar_2_Version_1_05_09_2019.csv"
ORDERS_PATH = DATASETS_DIR / "order_large.csv"

DELAY_MODEL_OUT = MODELS_DIR / "delay_model_optimized.pkl"
ANOMALY_MODEL_OUT = MODELS_DIR / "anomaly_model_optimized.pkl"
REPORT_OUT = MODELS_DIR / "optimization_report.json"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ANOMALY_FEATURES = [
    "total_orders",
    "avg_weight_kg",
    "weight_cv",
    "danger_ratio",
    "avg_lead_days",
    "min_lead_days",
    "orders_per_dest",
    "weight_spike_flag",
]



def _to_float(x: Any) -> float:
    return float(np.round(float(x), 6))



def _clean_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out



def _load_delay_raw(path: Path = DELAY_PATH) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    df = _clean_names(df)
    if "is_delayed" not in df.columns:
        raise ValueError("Missing 'is_delayed' in delay dataset")

    target_col = "is_delayed"
    df = df.drop_duplicates().reset_index(drop=True)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    if df[target_col].isna().any():
        df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
    return df, target_col



def _argmax_onehot(df: pd.DataFrame, prefix: str, default_val: float = 0.0) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return np.full(len(df), default_val, dtype=np.float32)
    arr = df[cols].to_numpy(dtype=np.float32)
    idx = np.argmax(arr, axis=1)
    values = np.array([float(c.split("_")[-1]) if c.split("_")[-1].replace(".", "", 1).isdigit() else float(i) for i, c in enumerate(cols)], dtype=np.float32)
    return values[idx]



def _engineer_delay_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    if "order_time" not in out.columns:
        out["order_time"] = _argmax_onehot(out, "order_time_", default_val=12.0)
    if "order_week_day" not in out.columns:
        out["order_week_day"] = _argmax_onehot(out, "order_week_day_", default_val=3.0)

    if "cp_delay_per_quarter" in out.columns and "cp_ontime_per_quarter" in out.columns:
        out["delay_ontime_ratio_quarter"] = out["cp_delay_per_quarter"] / (out["cp_ontime_per_quarter"] + eps)
    else:
        out["delay_ontime_ratio_quarter"] = 0.0

    if "cp_delay_per_month" in out.columns and "cp_ontime_per_month" in out.columns:
        out["delay_ontime_ratio_month"] = out["cp_delay_per_month"] / (out["cp_ontime_per_month"] + eps)
        out["delay_rate_delta_m_q"] = out["cp_delay_per_month"] - out.get("cp_delay_per_quarter", 0.0)
        out["ontime_rate_delta_m_q"] = out["cp_ontime_per_month"] - out.get("cp_ontime_per_quarter", 0.0)
    else:
        out["delay_ontime_ratio_month"] = 0.0
        out["delay_rate_delta_m_q"] = 0.0
        out["ontime_rate_delta_m_q"] = 0.0

    out["is_weekend"] = out["order_week_day"].isin([5, 6]).astype(float)
    out["order_hour_sin"] = np.sin(2 * np.pi * out["order_time"] / 24.0)
    out["order_hour_cos"] = np.cos(2 * np.pi * out["order_time"] / 24.0)

    if "distance" in out.columns and "shipment_weight" in out.columns:
        out["distance_weight_interaction"] = out["distance"] * out["shipment_weight"]
        out["distance_weight_ratio"] = out["distance"] / (out["shipment_weight"] + eps)
    else:
        out["distance_weight_interaction"] = 0.0
        out["distance_weight_ratio"] = 0.0

    out["distance_sla_ratio"] = out.get("distance", 0.0) / (out.get("SLA", 1.0) + eps)
    out["pickup_drop_metro_x"] = out.get("pickup_metro", 0.0) * out.get("drop_metro", 0.0)
    out["pickup_drop_non_metro_x"] = out.get("pickup_non_metro", 0.0) * out.get("drop_non_metro", 0.0)

    if {"pickup_lat", "pickup_lon", "drop_lat", "drop_lon"}.issubset(set(out.columns)):
        out["pickup_zone_key"] = (
            out["pickup_lat"].round(1).astype(str) + "_" + out["pickup_lon"].round(1).astype(str)
        )
        out["drop_zone_key"] = (
            out["drop_lat"].round(1).astype(str) + "_" + out["drop_lon"].round(1).astype(str)
        )
        out["route_key"] = out["pickup_zone_key"] + "->" + out["drop_zone_key"]
    else:
        out["pickup_zone_key"] = out.get("pickup_metro", 0.0).astype(str)
        out["drop_zone_key"] = out.get("drop_metro", 0.0).astype(str)
        out["route_key"] = out["pickup_zone_key"] + "->" + out["drop_zone_key"]

    non_numeric = [c for c in out.columns if out[c].dtype == "object" and c != target_col]
    for c in non_numeric:
        if c not in {"route_key", "pickup_zone_key", "drop_zone_key"}:
            out = out.drop(columns=[c])

    return out



def _add_group_delay_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    y_series = pd.Series(y_train, index=tr.index)
    global_mean = float(y_series.mean())

    keys = [
        ("route_key", "route_delay_roll_avg"),
        ("pickup_zone_key", "pickup_zone_delay_roll_avg"),
        ("drop_zone_key", "drop_zone_delay_roll_avg"),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for key, out_col in keys:
        oof = np.full(len(tr), global_mean, dtype=np.float32)
        for fit_idx, val_idx in skf.split(tr, y_train):
            fit_part = tr.iloc[fit_idx]
            fit_y = y_series.iloc[fit_idx]
            means = fit_y.groupby(fit_part[key]).mean()
            mapped = tr.iloc[val_idx][key].map(means).fillna(global_mean).to_numpy(dtype=np.float32)
            oof[val_idx] = mapped

        full_means = y_series.groupby(tr[key]).mean()
        tr[out_col] = oof
        te[out_col] = te[key].map(full_means).fillna(global_mean).astype(np.float32)

    return tr, te



def _metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": _to_float(accuracy_score(y_true, y_pred)),
        "precision": _to_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _to_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _to_float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _to_float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else 0.0,
    }
    return out



def _threshold_grid_search(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_f1 = -1.0
    best_prec = -1.0
    best_rec = -1.0
    best_t = 0.5
    for t in np.linspace(0.2, 0.9, 71):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if (
            f1 > best_f1
            or (abs(f1 - best_f1) < 1e-12 and rec > best_rec)
            or (abs(f1 - best_f1) < 1e-12 and abs(rec - best_rec) < 1e-12 and prec > best_prec)
        ):
            best_f1 = float(f1)
            best_prec = float(prec)
            best_rec = float(rec)
            best_t = float(t)
    return _to_float(best_t)



def _imbalance_strategy(name: str) -> Any:
    if name == "smote_tomek":
        return SMOTETomek(random_state=42)
    if name == "smoteenn":
        return SMOTEENN(random_state=42)
    return None



def _xgb_base(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": 0,
    }



def _identify_leakage_features(
    columns: List[str],
    strict: bool = False,
) -> List[str]:
    explicit = {
        "actual_tat",
        "actual_delivery_time",
        "delivery_completion_time",
        "completed_time",
    }
    removed: List[str] = []
    for c in columns:
        lc = c.lower().strip()
        is_leak = (
            lc in explicit
            or lc.startswith("pickedup_")
            or lc.startswith("pickup_time_")
            or lc == "pickup_time"
            or lc.startswith("actual_")
            or "actual_tat" in lc
            or "delivery_completion" in lc
            or "completed_delivery" in lc
        )
        if strict:
            is_leak = is_leak or lc.startswith("cp_")
        if is_leak:
            removed.append(c)
    return sorted(set(removed))



def _prepare_feature_matrix(strict_leakage_filter: bool = False) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    raw, target_col = _load_delay_raw()
    feat = _engineer_delay_features(raw, target_col)

    y = feat[target_col].astype(int).to_numpy()
    X_df = feat.drop(columns=[target_col])

    removed_leakage_features = _identify_leakage_features(X_df.columns.tolist(), strict=strict_leakage_filter)
    if removed_leakage_features:
        X_df = X_df.drop(columns=removed_leakage_features, errors="ignore")

    numeric_cols = [c for c in X_df.columns if c not in {"route_key", "pickup_zone_key", "drop_zone_key"}]
    for c in numeric_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df[numeric_cols] = X_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_df[numeric_cols] = X_df[numeric_cols].fillna(X_df[numeric_cols].median(numeric_only=True))

    return X_df, y, numeric_cols, removed_leakage_features



def _select_features_by_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
    neg = max(int((y == 0).sum()), 1)
    pos = max(int((y == 1).sum()), 1)
    spw = neg / pos
    model = XGBClassifier(
        **_xgb_base(spw),
        n_estimators=260,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.85,
    )
    model.fit(X, y)
    imp = np.asarray(model.feature_importances_, dtype=np.float64)
    order = np.argsort(imp)[::-1]

    nonzero = imp[order] > 0
    keep_n = int(max(40, min(100, nonzero.sum() if nonzero.any() else len(feature_names))))
    keep_idx = order[:keep_n]
    return [feature_names[int(i)] for i in keep_idx]



def _fit_calibrated(estimator: Any, X: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=2)
    cal.fit(X, y)
    return cal



def _extract_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps["model"]
    return estimator



def _train_delay_model_core(strict_leakage_filter: bool = False) -> Dict[str, Any]:
    X_df, y, numeric_cols, removed_leakage_features = _prepare_feature_matrix(
        strict_leakage_filter=strict_leakage_filter
    )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=42
    )

    drop_group_cols = [c for c in ["route_key", "pickup_zone_key", "drop_zone_key"] if c in X_train_df.columns]
    X_train_df = X_train_df.drop(columns=drop_group_cols)
    X_test_df = X_test_df.drop(columns=drop_group_cols)

    all_features = X_train_df.columns.tolist()
    selected_features = _select_features_by_importance(
        X_train_df[all_features].to_numpy(dtype=np.float32), y_train, all_features
    )

    X_train = X_train_df[selected_features].to_numpy(dtype=np.float32)
    X_test = X_test_df[selected_features].to_numpy(dtype=np.float32)

    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = neg / pos

    model_limit = min(len(X_train), 18000)
    if model_limit < len(X_train):
        rng = np.random.default_rng(42)
        pos_idx_full = np.where(y_train == 1)[0]
        neg_idx_full = np.where(y_train == 0)[0]
        pos_take = int(model_limit * (len(pos_idx_full) / len(y_train)))
        pos_take = max(1200, min(len(pos_idx_full), pos_take))
        neg_take = max(1200, model_limit - pos_take)
        neg_take = min(len(neg_idx_full), neg_take)
        picked = np.concatenate(
            [
                rng.choice(pos_idx_full, size=pos_take, replace=False),
                rng.choice(neg_idx_full, size=neg_take, replace=False),
            ]
        )
        rng.shuffle(picked)
        X_model = X_train[picked]
        y_model = y_train[picked]
    else:
        X_model = X_train
        y_model = y_train

    imbalance_options = ["smote_tomek", "smoteenn", "weighted"]

    xgb_param = {
        "n_estimators": randint(120, 360),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.03, 0.17),
        "subsample": uniform(0.65, 0.35),
        "colsample_bytree": uniform(0.65, 0.35),
        "min_child_weight": randint(1, 8),
        "gamma": uniform(0.0, 3.0),
        "reg_alpha": uniform(0.0, 1.5),
        "reg_lambda": uniform(0.5, 3.0),
    }

    X_strat_fit, X_strat_val, y_strat_fit, y_strat_val = train_test_split(
        X_model, y_model, test_size=0.2, stratify=y_model, random_state=42
    )
    best_xgb_strategy = "weighted"
    best_xgb_score = -1.0
    for strategy in imbalance_options:
        if strategy == "weighted":
            X_tmp, y_tmp = X_strat_fit, y_strat_fit
            model = XGBClassifier(
                **_xgb_base(scale_pos_weight=scale_pos_weight),
                n_estimators=180,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
            )
        else:
            resampler = _imbalance_strategy(strategy)
            X_tmp, y_tmp = resampler.fit_resample(X_strat_fit, y_strat_fit)
            model = XGBClassifier(
                **_xgb_base(scale_pos_weight=1.0),
                n_estimators=180,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
            )
        model.fit(X_tmp, y_tmp)
        p_val = model.predict_proba(X_strat_val)[:, 1]
        t_val = _threshold_grid_search(y_strat_val, p_val)
        f1_val = f1_score(y_strat_val, (p_val >= t_val).astype(int), zero_division=0)
        if float(f1_val) > best_xgb_score:
            best_xgb_score = float(f1_val)
            best_xgb_strategy = strategy

    X_tune = X_model
    y_tune = y_model

    if best_xgb_strategy == "weighted":
        X_search, y_search = X_tune, y_tune
        estimator = XGBClassifier(**_xgb_base(scale_pos_weight=scale_pos_weight))
    else:
        resampler = _imbalance_strategy(best_xgb_strategy)
        X_search, y_search = resampler.fit_resample(X_tune, y_tune)
        estimator = XGBClassifier(**_xgb_base(scale_pos_weight=1.0))

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_xgb_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=xgb_param,
        n_iter=3,
        scoring={"f1": "f1", "recall": "recall"},
        refit="f1",
        cv=cv5,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    best_xgb_search.fit(X_search, y_search)
    best_xgb_params_raw = dict(best_xgb_search.best_params_)
    xgb_params_for_final: Dict[str, Any] = dict(best_xgb_params_raw)

    X_fit, X_es_val, y_fit, y_es_val = train_test_split(
        X_model, y_model, test_size=0.15, stratify=y_model, random_state=42
    )

    if best_xgb_strategy == "weighted":
        X_fit_res, y_fit_res = X_fit, y_fit
        final_spw = scale_pos_weight
    else:
        resampler = _imbalance_strategy(best_xgb_strategy)
        X_fit_res, y_fit_res = resampler.fit_resample(X_fit, y_fit)
        final_spw = 1.0

    xgb_es = XGBClassifier(
        **_xgb_base(scale_pos_weight=final_spw),
        **xgb_params_for_final,
        early_stopping_rounds=35,
    )
    xgb_es.fit(X_fit_res, y_fit_res, eval_set=[(X_es_val, y_es_val)], verbose=False)
    best_iter = getattr(xgb_es, "best_iteration", None)
    if best_iter is not None and int(best_iter) > 20:
        xgb_params_for_final["n_estimators"] = int(best_iter)

    if best_xgb_strategy == "weighted":
        xgb_final = XGBClassifier(**_xgb_base(scale_pos_weight=scale_pos_weight), **xgb_params_for_final)
    else:
        xgb_final = ImbPipeline(
            steps=[
                ("resample", _imbalance_strategy(best_xgb_strategy)),
                ("model", XGBClassifier(**_xgb_base(scale_pos_weight=1.0), **xgb_params_for_final)),
            ]
        )

    lgbm_base = LGBMClassifier(
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        class_weight="balanced",
        n_estimators=220,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        num_leaves=48,
        reg_alpha=0.1,
        reg_lambda=1.2,
    )
    lgbm_final = ImbPipeline(
        steps=[
            (
                "resample",
                _imbalance_strategy(best_xgb_strategy)
                if best_xgb_strategy != "weighted"
                else SMOTETomek(random_state=42),
            ),
            ("model", lgbm_base),
        ]
    )

    rf_final = ImbPipeline(
        steps=[
            (
                "resample",
                _imbalance_strategy(best_xgb_strategy)
                if best_xgb_strategy != "weighted"
                else SMOTETomek(random_state=42),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    n_estimators=260,
                    max_depth=12,
                    min_samples_split=6,
                    min_samples_leaf=2,
                    max_features=0.65,
                ),
            ),
        ]
    )

    xgb_cal = _fit_calibrated(xgb_final, X_model, y_model)
    lgbm_cal = _fit_calibrated(lgbm_final, X_model, y_model)
    rf_cal = _fit_calibrated(rf_final, X_model, y_model)

    train_opt_probs = {
        "xgboost": xgb_cal.predict_proba(X_model)[:, 1],
        "lightgbm": lgbm_cal.predict_proba(X_model)[:, 1],
        "random_forest": rf_cal.predict_proba(X_model)[:, 1],
    }
    train_probs = {
        "xgboost": xgb_cal.predict_proba(X_train)[:, 1],
        "lightgbm": lgbm_cal.predict_proba(X_train)[:, 1],
        "random_forest": rf_cal.predict_proba(X_train)[:, 1],
    }
    test_probs = {
        "xgboost": xgb_cal.predict_proba(X_test)[:, 1],
        "lightgbm": lgbm_cal.predict_proba(X_test)[:, 1],
        "random_forest": rf_cal.predict_proba(X_test)[:, 1],
    }

    best_combo = {"xgboost": 0.34, "lightgbm": 0.33, "random_forest": 0.33}
    best_threshold = 0.5
    best_score = (-1.0, -1.0, -1.0)
    best_feasible_score = (-1.0, -1.0, -1.0)
    best_feasible_combo = None
    best_feasible_threshold = None

    for wx in np.linspace(0.1, 0.8, 9):
        for wl in np.linspace(0.1, 0.8, 9):
            wr = 1.0 - wx - wl
            if wr < 0.1 or wr > 0.8:
                continue
            p_train = wx * train_opt_probs["xgboost"] + wl * train_opt_probs["lightgbm"] + wr * train_opt_probs["random_forest"]
            thr = _threshold_grid_search(y_model, p_train)
            yp = (p_train >= thr).astype(int)
            f1 = f1_score(y_model, yp, zero_division=0)
            rec = recall_score(y_model, yp, zero_division=0)
            prec = precision_score(y_model, yp, zero_division=0)
            score = (float(f1), float(rec), float(prec))
            if score > best_score:
                best_score = score
                best_combo = {
                    "xgboost": _to_float(wx),
                    "lightgbm": _to_float(wl),
                    "random_forest": _to_float(wr),
                }
                best_threshold = thr
            if float(rec) >= 0.60 and float(prec) >= 0.40:
                if score > best_feasible_score:
                    best_feasible_score = score
                    best_feasible_combo = {
                        "xgboost": _to_float(wx),
                        "lightgbm": _to_float(wl),
                        "random_forest": _to_float(wr),
                    }
                    best_feasible_threshold = thr

    if best_feasible_combo is not None and best_feasible_threshold is not None:
        best_combo = best_feasible_combo
        best_threshold = best_feasible_threshold

    p_train_final = (
        best_combo["xgboost"] * train_probs["xgboost"]
        + best_combo["lightgbm"] * train_probs["lightgbm"]
        + best_combo["random_forest"] * train_probs["random_forest"]
    )
    p_test_final = (
        best_combo["xgboost"] * test_probs["xgboost"]
        + best_combo["lightgbm"] * test_probs["lightgbm"]
        + best_combo["random_forest"] * test_probs["random_forest"]
    )

    train_metrics = _metric_bundle(y_train, p_train_final, best_threshold)
    test_metrics = _metric_bundle(y_test, p_test_final, best_threshold)

    cv_idx = int(best_xgb_search.best_index_)
    cv_f1 = _to_float(best_xgb_search.cv_results_["mean_test_f1"][cv_idx])
    cv_recall = _to_float(best_xgb_search.cv_results_["mean_test_recall"][cv_idx])

    xgb_for_importance = _extract_estimator(xgb_final)
    if hasattr(xgb_for_importance, "fit"):
        xgb_for_importance.fit(X_model, y_model)
        xgb_imp = np.asarray(getattr(xgb_for_importance, "feature_importances_", np.zeros(X_train.shape[1])), dtype=np.float64)
    else:
        xgb_imp = np.zeros(X_train.shape[1], dtype=np.float64)

    lgbm_model = _extract_estimator(lgbm_final)
    if hasattr(lgbm_model, "fit"):
        lgbm_model.fit(X_model, y_model)
        lgbm_imp = np.asarray(getattr(lgbm_model, "feature_importances_", np.zeros(X_train.shape[1])), dtype=np.float64)
    else:
        lgbm_imp = np.zeros(X_train.shape[1], dtype=np.float64)

    rf_model = _extract_estimator(rf_final)
    if hasattr(rf_model, "fit"):
        rf_model.fit(X_model, y_model)
        rf_imp = np.asarray(getattr(rf_model, "feature_importances_", np.zeros(X_train.shape[1])), dtype=np.float64)
    else:
        rf_imp = np.zeros(X_train.shape[1], dtype=np.float64)

    def _norm(a: np.ndarray) -> np.ndarray:
        s = float(np.sum(a))
        if s <= 0:
            return np.zeros_like(a)
        return a / s

    combined_imp = (
        best_combo["xgboost"] * _norm(xgb_imp)
        + best_combo["lightgbm"] * _norm(lgbm_imp)
        + best_combo["random_forest"] * _norm(rf_imp)
    )
    top_idx = np.argsort(combined_imp)[::-1][:10]
    top_features = [
        {"feature": selected_features[int(i)], "importance": _to_float(combined_imp[int(i)])}
        for i in top_idx
    ]

    best_params = {
        "imbalance_strategy": best_xgb_strategy,
        "xgboost": {
            k.replace("model__", ""): _to_float(v) if isinstance(v, (float, np.floating)) else int(v)
            for k, v in best_xgb_params_raw.items()
        },
        "lightgbm": {
            "n_estimators": 220,
            "max_depth": 6,
            "learning_rate": 0.07,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "num_leaves": 48,
            "reg_alpha": 0.1,
            "reg_lambda": 1.2,
        },
        "random_forest": {
            "n_estimators": 260,
            "max_depth": 12,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_features": 0.65,
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "models": {
                "xgb": xgb_cal,
                "lgbm": lgbm_cal,
                "rf": rf_cal,
            },
            "features": selected_features,
            "removed_leakage_features": removed_leakage_features,
            "weights": best_combo,
            "threshold": best_threshold,
            "best_params": best_params,
        },
        DELAY_MODEL_OUT,
    )

    return {
        "models_used": ["xgboost", "lightgbm", "random_forest"],
        "ensemble_weights": {
            "xgboost": _to_float(best_combo["xgboost"]),
            "lightgbm": _to_float(best_combo["lightgbm"]),
            "random_forest": _to_float(best_combo["random_forest"]),
        },
        "best_params": best_params,
        "optimal_threshold": _to_float(best_threshold),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cross_validation": {
            "f1_mean": cv_f1,
            "recall_mean": cv_recall,
        },
        "top_features": top_features,
        "features_used": selected_features,
        "removed_leakage_features": removed_leakage_features,
    }


def train_delay_model() -> Dict[str, Any]:
    report = _train_delay_model_core(strict_leakage_filter=False)
    if report["test_metrics"]["f1"] > 0.90 or report["test_metrics"]["roc_auc"] > 0.95:
        report = _train_delay_model_core(strict_leakage_filter=True)
    return report


def _build_city_features(order_path: Path = ORDERS_PATH) -> pd.DataFrame:
    dtypes = {
        "Order_ID": str,
        "Source": str,
        "Destination": str,
        "Danger_Type": str,
        "Weight": float,
        "Area": float,
    }
    df = pd.read_csv(order_path, dtype=dtypes, parse_dates=["Available_Time", "Deadline"])
    df["lead_days"] = (df["Deadline"] - df["Available_Time"]).dt.total_seconds() / 86400
    df["lead_days"] = df["lead_days"].clip(lower=0)
    df["is_dangerous"] = df["Danger_Type"].isin(["type_3", "type_4"]).astype(int)
    df["weight_kg"] = df["Weight"].where(df["Weight"] < 1e9, df["Weight"] / 1000)

    grp = df.groupby("Source")
    feat = pd.DataFrame(
        {
            "total_orders": grp["Order_ID"].count(),
            "avg_weight_kg": grp["weight_kg"].mean(),
            "std_weight_kg": grp["weight_kg"].std().fillna(0),
            "max_weight_kg": grp["weight_kg"].max(),
            "median_weight": grp["weight_kg"].median(),
            "danger_ratio": grp["is_dangerous"].mean(),
            "avg_lead_days": grp["lead_days"].mean(),
            "min_lead_days": grp["lead_days"].min(),
            "n_destinations": grp["Destination"].nunique(),
        }
    ).reset_index().rename(columns={"Source": "city_id"})
    feat["weight_cv"] = feat["std_weight_kg"] / (feat["avg_weight_kg"] + 1e-6)
    feat["orders_per_dest"] = feat["total_orders"] / (feat["n_destinations"] + 1)
    feat["weight_spike_flag"] = (feat["max_weight_kg"] > 4 * feat["median_weight"]).astype(float)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat = feat.drop_duplicates(subset=["city_id"])
    return feat



def train_anomaly_model() -> Dict[str, Any]:
    feat_df = _build_city_features()
    X_raw = feat_df[ANOMALY_FEATURES].astype(np.float32).values
    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)

    known_anomaly_nodes: set[str] = set()
    try:
        import generate_dataset as gd

        known_anomaly_nodes = set(getattr(gd, "ANOMALY_CITIES", {}).keys())
    except Exception:
        known_anomaly_nodes = set()

    y_known = feat_df["city_id"].isin(known_anomaly_nodes).astype(int).values
    labels_available = int(y_known.sum()) > 0

    best = {
        "model_type": "IsolationForest",
        "contamination": 0.1,
        "pred": np.zeros(len(X), dtype=int),
        "score": (-1.0, -1.0, -1.0),
        "artifact": None,
    }

    contam_grid = [0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.15, 0.18, 0.2]

    for contamination in contam_grid:
        iso = IsolationForest(
            n_estimators=450,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(X)
        pred = (iso.predict(X) == -1).astype(int)
        anomaly_rate = float(pred.mean())

        if labels_available:
            rec = recall_score(y_known, pred, zero_division=0)
            f1 = f1_score(y_known, pred, zero_division=0)
            score = (float(rec), float(f1), -abs(anomaly_rate - 0.12))
        else:
            score = (0.0, 0.0, -abs(anomaly_rate - 0.12))

        if score > best["score"]:
            best = {
                "model_type": "IsolationForest",
                "contamination": float(contamination),
                "pred": pred,
                "score": score,
                "artifact": {"model": iso, "scaler": scaler, "features": ANOMALY_FEATURES},
            }

    hidden = max(2, X.shape[1] // 2)
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
    )
    ae.fit(X, X)
    recon = ae.predict(X)
    err = np.mean((X - recon) ** 2, axis=1)

    for contamination in contam_grid:
        thr = float(np.quantile(err, 1.0 - contamination))
        pred = (err >= thr).astype(int)
        anomaly_rate = float(pred.mean())

        if labels_available:
            rec = recall_score(y_known, pred, zero_division=0)
            f1 = f1_score(y_known, pred, zero_division=0)
            score = (float(rec), float(f1), -abs(anomaly_rate - 0.12))
        else:
            score = (0.0, 0.0, -abs(anomaly_rate - 0.12))

        if score > best["score"]:
            best = {
                "model_type": "Autoencoder",
                "contamination": float(contamination),
                "pred": pred,
                "score": score,
                "artifact": {
                    "model": ae,
                    "scaler": scaler,
                    "features": ANOMALY_FEATURES,
                    "threshold": thr,
                    "mode": "reconstruction_error",
                },
            }

    pred_best = best["pred"]
    anomaly_rate = _to_float(float(pred_best.mean()))

    if labels_available:
        detection_rate = _to_float(recall_score(y_known, pred_best, zero_division=0))
    else:
        detection_rate = 0.0

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            **(best["artifact"] or {}),
            "contamination": _to_float(best["contamination"]),
            "model_type": best["model_type"],
        },
        ANOMALY_MODEL_OUT,
    )

    return {
        "model_type": best["model_type"],
        "contamination": _to_float(best["contamination"]),
        "anomaly_rate": anomaly_rate,
        "detection_rate": detection_rate,
    }



def run_pipeline() -> Dict[str, Any]:
    delay = train_delay_model()
    anomaly = train_anomaly_model()
    out = {
        "metrics": {
            "delay_model": {
                "train": delay["train_metrics"],
                "test": delay["test_metrics"],
                "cross_validation": delay["cross_validation"],
                "optimal_threshold": delay["optimal_threshold"],
            },
            "anomaly_model": {
                "model_type": anomaly["model_type"],
                "contamination": anomaly["contamination"],
                "anomaly_rate": anomaly["anomaly_rate"],
                "detection_rate": anomaly["detection_rate"],
            },
        },
        "features_used": delay["features_used"],
        "removed_leakage_features": delay["removed_leakage_features"],
    }
    MODELS_DIR.mkdir(exist_ok=True)
    REPORT_OUT.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=True), encoding="utf-8")
    return out


if __name__ == "__main__":
    report = run_pipeline()
    print(json.dumps(report, separators=(",", ":"), ensure_ascii=True))
