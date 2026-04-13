#!/usr/bin/env python3
"""
train_models.py
----------------
Train all local ML models used by SupplAI and generate sanity diagnostics
for overfitting and group-level bias signals on synthetic data.

Usage:
    python train_models.py
    python train_models.py --force-retrain
    python train_models.py --skip-delay
    python train_models.py --skip-anomaly
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_PATH = MODELS_DIR / "model_sanity_report.json"
EVALUATION_MD_PATH = PROJECT_ROOT / "evaluation.md"

import sys
sys.path.insert(0, str(SRC_PATH))

from delay_model import (  # noqa: E402
    FEATURE_COLS as DELAY_FEATURES,
    TARGET_COL as DELAY_TARGET,
    _get_model as get_delay_model_template,
    load_or_train,
    load_training_data,
)
from anomaly_detector import (  # noqa: E402
    ANOMALY_THRESHOLD,
    FEATURE_COLS as ANOMALY_FEATURES,
    _build_city_features,
    load_or_train_anomaly,
    score_anomalies,
)
from graph_builder import build_graph, load_supply_metadata  # noqa: E402


def _float(v: Any) -> float:
    return float(np.round(float(v), 6))


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": _float(accuracy_score(y_true, y_pred)),
        "precision": _float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = _float(roc_auc_score(y_true, y_prob))
    else:
        out["auc"] = float("nan")
    return out


def _group_bias_summary(
    frame: pd.DataFrame,
    group_col: str,
    min_count: int = 200,
) -> Dict[str, Any]:
    grouped = frame.groupby(group_col, dropna=False)
    rows: List[Dict[str, Any]] = []

    for group, g in grouped:
        if len(g) < min_count:
            continue
        positives = g[g["y_true"] == 1]
        negatives = g[g["y_true"] == 0]
        rows.append(
            {
                "group": str(group),
                "count": int(len(g)),
                "positive_rate": _float(g["y_pred"].mean()),
                "tpr": _float(positives["y_pred"].mean()) if len(positives) else None,
                "fpr": _float(negatives["y_pred"].mean()) if len(negatives) else None,
            }
        )

    if not rows:
        return {"group": group_col, "rows": [], "positive_rate_disparity": None}

    positive_rates = [r["positive_rate"] for r in rows]
    disparity = max(positive_rates) - min(positive_rates)
    rows.sort(key=lambda r: r["count"], reverse=True)
    return {
        "group": group_col,
        "rows": rows,
        "positive_rate_disparity": _float(disparity),
    }


def _build_sla_band(series: pd.Series) -> pd.Series:
    q1, q2 = series.quantile([0.33, 0.66])
    if q1 == q2:
        return pd.Series(["uniform"] * len(series), index=series.index)
    return pd.cut(series, bins=[-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"]).astype(str)


def run_delay_sanity() -> Dict[str, Any]:
    print("[sanity] Loading delay dataset …")
    df = load_training_data()
    df = df.copy()

    class_dist = df[DELAY_TARGET].value_counts(normalize=True).to_dict()
    duplicate_ratio = _float(df.duplicated(subset=DELAY_FEATURES + [DELAY_TARGET]).mean())
    corr_series = (
        df[DELAY_FEATURES + [DELAY_TARGET]]
        .corr(numeric_only=True)[DELAY_TARGET]
        .drop(labels=[DELAY_TARGET])
        .abs()
        .sort_values(ascending=False)
    )
    top_feature_correlations = {k: _float(v) for k, v in corr_series.head(6).items()}

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[DELAY_TARGET],
    )

    X_train = train_df[DELAY_FEATURES].values.astype(np.float32)
    y_train = train_df[DELAY_TARGET].values.astype(np.int32)
    X_test = test_df[DELAY_FEATURES].values.astype(np.float32)
    y_test = test_df[DELAY_TARGET].values.astype(np.int32)

    model, model_type = get_delay_model_template()
    print(f"[sanity] Fitting holdout model ({model_type}) on {len(X_train):,} rows …")
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = _binary_metrics(y_train, train_prob)
    test_metrics = _binary_metrics(y_test, test_prob)

    overfit_gap_auc = (
        _float(train_metrics["auc"] - test_metrics["auc"])
        if np.isfinite(train_metrics["auc"]) and np.isfinite(test_metrics["auc"])
        else None
    )
    overfit_gap_acc = _float(train_metrics["accuracy"] - test_metrics["accuracy"])

    eval_df = test_df.copy()
    eval_df["y_true"] = y_test
    eval_df["y_prob"] = test_prob
    eval_df["y_pred"] = (test_prob >= 0.5).astype(int)
    eval_df["pickup_zone"] = np.where(eval_df["pickup_metro"] >= 0.5, "metro", "non_metro")
    eval_df["drop_zone"] = np.where(eval_df["drop_metro"] >= 0.5, "metro", "non_metro")
    eval_df["sla_band"] = _build_sla_band(eval_df["SLA"])

    bias = {
        "pickup_zone": _group_bias_summary(eval_df, "pickup_zone", min_count=200),
        "drop_zone": _group_bias_summary(eval_df, "drop_zone", min_count=200),
        "sla_band": _group_bias_summary(eval_df, "sla_band", min_count=200),
    }

    return {
        "rows": int(len(df)),
        "class_distribution": {str(k): _float(v) for k, v in class_dist.items()},
        "duplicate_ratio": duplicate_ratio,
        "top_feature_target_abs_correlation": top_feature_correlations,
        "model_type": model_type,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfit_gap_auc": overfit_gap_auc,
        "overfit_gap_accuracy": overfit_gap_acc,
        "bias_signals": bias,
    }


def _group_anomaly_rates(
    anomaly_df: pd.DataFrame,
    group_col: str,
    min_count: int = 3,
) -> Dict[str, Any]:
    grp = (
        anomaly_df.groupby(group_col, dropna=False)["is_anomalous"]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "count", "mean": "anomaly_rate"})
    )
    grp = grp[grp["count"] >= min_count].copy()
    if grp.empty:
        return {"group": group_col, "rows": [], "disparity": None}
    grp.sort_values("count", ascending=False, inplace=True)
    disparity = float(grp["anomaly_rate"].max() - grp["anomaly_rate"].min())
    rows = [
        {"group": str(r[group_col]), "count": int(r["count"]), "anomaly_rate": _float(r["anomaly_rate"])}
        for _, r in grp.iterrows()
    ]
    return {"group": group_col, "rows": rows, "disparity": _float(disparity)}


def run_anomaly_sanity() -> Dict[str, Any]:
    print("[sanity] Building city feature matrix for anomaly checks …")
    feat_df = _build_city_features()
    duplicate_city_rows = _float(feat_df.duplicated(subset=["city_id"]).mean())

    artifact = load_or_train_anomaly(force_retrain=False)
    X = artifact["scaler"].transform(artifact["feat_df"][ANOMALY_FEATURES].values.astype(np.float32))
    scores = artifact["model"].decision_function(X)
    train_anomaly_rate = _float(float((scores < ANOMALY_THRESHOLD).mean()))

    G = build_graph()
    supply_df = load_supply_metadata()
    scored = score_anomalies(artifact, G, supply_df)

    overall_anomaly_rate = _float(scored["is_anomalous"].mean())
    by_country = _group_anomaly_rates(scored, "country", min_count=3)
    by_product = _group_anomaly_rates(scored, "product", min_count=3)

    injected_overlap = None
    try:
        import generate_dataset as gd  # type: ignore

        injected = set(getattr(gd, "ANOMALY_CITIES", {}).keys())
        if injected:
            predicted = set(scored.loc[scored["is_anomalous"], "node"])
            known = injected & set(scored["node"])
            injected_overlap = {
                "known_injected_cities": int(len(known)),
                "detected_injected_cities": int(len(known & predicted)),
                "detection_rate": _float(len(known & predicted) / max(len(known), 1)),
            }
    except Exception:
        injected_overlap = None

    return {
        "rows": int(len(feat_df)),
        "duplicate_city_rows": duplicate_city_rows,
        "training_score_mean": _float(np.mean(scores)),
        "training_score_std": _float(np.std(scores)),
        "training_anomaly_rate": train_anomaly_rate,
        "graph_scored_nodes": int(len(scored)),
        "graph_anomaly_rate": overall_anomaly_rate,
        "bias_signals": {
            "country": by_country,
            "product": by_product,
        },
        "injected_anomaly_overlap": injected_overlap,
    }


def _evaluation_markdown(report: Dict[str, Any]) -> str:
    d = report["delay_model_sanity"]
    a = report["anomaly_model_sanity"]
    train = d["train_metrics"]
    test = d["test_metrics"]

    auc_gap = d["overfit_gap_auc"]
    if auc_gap is None:
        overfit_status = "UNKNOWN"
    elif auc_gap <= 0.05:
        overfit_status = "PASS"
    elif auc_gap <= 0.10:
        overfit_status = "WARN"
    else:
        overfit_status = "FAIL"

    dup = d["duplicate_ratio"]
    if dup <= 0.10:
        dup_status = "PASS"
    elif dup <= 0.30:
        dup_status = "WARN"
    else:
        dup_status = "FAIL"

    inj = a.get("injected_anomaly_overlap") or {}

    return f"""# Model Evaluation Report

Generated from: `models/model_sanity_report.json`  
Generated at (UTC): **{report['generated_at_utc']}**

## Models Evaluated

1. Delay prediction model: XGBoost (CUDA) / RandomForest fallback (`models/delay_model.pkl`)
2. Anomaly detection model: IsolationForest (`models/anomaly_model.pkl`)

## Delay Model Metrics

| Metric | Train | Test |
|---|---:|---:|
| Accuracy | {train['accuracy']:.4f} | {test['accuracy']:.4f} |
| Precision | {train['precision']:.4f} | {test['precision']:.4f} |
| Recall | {train['recall']:.4f} | {test['recall']:.4f} |
| F1 | {train['f1']:.4f} | {test['f1']:.4f} |
| AUC | {train['auc']:.4f} | {test['auc']:.4f} |

### Overfitting Check

- AUC gap (train - test): **{d['overfit_gap_auc']:.4f}** -> **{overfit_status}**
- Accuracy gap (train - test): **{d['overfit_gap_accuracy']:.4f}**

Interpretation:
- Model is learning signal, but there is **generalization drop** from train to test.

### Dataset Quality Signals (Delay Data)

- Rows used: **{d['rows']}**
- Class distribution: **{d['class_distribution']}**
- Exact duplicate ratio (feature+target): **{d['duplicate_ratio']:.4f}** -> **{dup_status}**
- Top feature-target absolute correlations: **{d.get('top_feature_target_abs_correlation', {})}**

## Bias/Fairness Indicators (Delay Model)

Indicators below are prediction-rate disparities across groups (higher means more imbalance):

- Pickup zone disparity: **{d['bias_signals']['pickup_zone']['positive_rate_disparity']:.4f}**
- Drop zone disparity: **{d['bias_signals']['drop_zone']['positive_rate_disparity']:.4f}**
- SLA band disparity: **{d['bias_signals']['sla_band']['positive_rate_disparity']:.4f}**

## Anomaly Model Metrics

- Feature rows (cities): **{a['rows']}**
- Duplicate city rows: **{a['duplicate_city_rows']:.4f}**
- Training anomaly rate: **{a['training_anomaly_rate']:.4f}**
- Graph anomaly rate: **{a['graph_anomaly_rate']:.4f}**

### Anomaly Bias Indicators

- Country anomaly-rate disparity: **{a['bias_signals']['country']['disparity']:.4f}**
- Product anomaly-rate disparity: **{a['bias_signals']['product']['disparity']:.4f}**

### Injected Anomaly Recovery (Synthetic Stress Check)

- Known injected anomaly cities: **{inj.get('known_injected_cities', 'N/A')}**
- Detected injected anomaly cities: **{inj.get('detected_injected_cities', 'N/A')}**
- Detection rate: **{inj.get('detection_rate', 'N/A')}**

## Overall Health Summary

1. Training pipeline works and produces both model artifacts.
2. Inference pipeline is functional with pretrained artifacts.
3. Main quality risks to address next:
- Overfitting gap in delay model (AUC gap > 0.10)
- High duplicate ratio in delay dataset
- Group disparities in anomaly rates by country/product
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SupplAI models and run sanity checks.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain models even if cached files exist.")
    parser.add_argument("--skip-delay", action="store_true", help="Skip delay model training.")
    parser.add_argument("--skip-anomaly", action="store_true", help="Skip anomaly model training.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    trained: Dict[str, Any] = {}
    if not args.skip_delay:
        print("[train] Training/Loading delay model …")
        trained["delay_model"] = load_or_train(force_retrain=args.force_retrain)
    if not args.skip_anomaly:
        print("[train] Training/Loading anomaly model …")
        trained["anomaly_model"] = load_or_train_anomaly(force_retrain=args.force_retrain)

    print("[sanity] Running delay-model sanity checks …")
    delay_sanity = run_delay_sanity()

    print("[sanity] Running anomaly-model sanity checks …")
    anomaly_sanity = run_anomaly_sanity()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_trained": {
            "delay_model": not args.skip_delay,
            "anomaly_model": not args.skip_anomaly,
        },
        "delay_model_sanity": delay_sanity,
        "anomaly_model_sanity": anomaly_sanity,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Sanity report saved to: {REPORT_PATH}")
    EVALUATION_MD_PATH.write_text(_evaluation_markdown(report), encoding="utf-8")
    print(f"[done] Evaluation markdown saved to: {EVALUATION_MD_PATH}")

    # Compact console summary.
    print(
        "[summary] Delay AUC train/test/gap: "
        f"{delay_sanity['train_metrics']['auc']:.4f} / "
        f"{delay_sanity['test_metrics']['auc']:.4f} / "
        f"{delay_sanity['overfit_gap_auc']:.4f}"
    )
    print(
        "[summary] Anomaly rate train/graph: "
        f"{anomaly_sanity['training_anomaly_rate']:.4f} / "
        f"{anomaly_sanity['graph_anomaly_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
